import { arms, colors } from "../../lib/theme";
import { byArm } from "../../lib/transforms";
import type { Metric, Model, Summary, Reference } from "../../api/types";
import type { SeriesOption } from "echarts";
import type { PlotOption } from "./Plot";
import {
  axisTip,
  categoryAxis,
  errorBars,
  formatPoint,
  grid,
  itemTip,
  radarWhiskers,
  valueAxis,
} from "./options";
export type UtilityView = "line" | "bar" | "area" | "radar" | "slope" | "heatmap" | "small";
export const utilityViews: Record<UtilityView, string> = {
  line: "Line",
  bar: "Grouped bar",
  area: "Area",
  radar: "Radar",
  slope: "Slope",
  heatmap: "Heatmap",
  small: "Small multiples",
};
const colorFor = (arm: string) => arms[arm]?.color ?? colors.slate;
const labelFor = (arm: string) => arms[arm]?.label ?? arm;
export function utilityOption(
  rows: Summary[],
  metric: Metric,
  selected: string[],
  ref: Reference | undefined,
  model: Model,
  type: UtilityView,
): PlotOption {
  const grouped = byArm(rows, metric).filter((group) => selected.includes(group.arm));
  const sizes = [...new Set(rows.map((r) => r.seed_size))].sort((a, b) => a - b);
  const domain = type === "slope" && sizes.length > 1 ? [sizes[0], sizes.at(-1)!] : sizes;
  const sizeValues = (values: (typeof grouped)[number]["values"]) =>
    domain.map((size) => values.find((v) => v.size === size));
  const reference = ref?.trtr_full?.[model]?.[metric];
  const random = metric === "pr_auc" ? ref?.random_guess_pr_auc : undefined;
  if (type === "heatmap")
    return {
      grid: { ...grid, left: 100 },
      xAxis: categoryAxis(domain),
      yAxis: {
        ...categoryAxis(grouped.map((g) => labelFor(g.arm))),
        axisLabel: { width: 84, overflow: "truncate", color: colors.muted },
      },
      tooltip: {
        ...itemTip,
        formatter: (p: unknown) => {
          const {
            value: [x, y, mean, sd, n],
          } = p as { value: number[] };
          return `${labelFor(grouped[y].arm)} · ${domain[x]} rows\n${formatPoint(mean, sd, n)}`;
        },
      },
      visualMap: {
        show: false,
        min: 0,
        max: 1,
        dimension: 2,
        inRange: { color: [colors.softIndigo, colors.indigo] },
      },
      series: [
        {
          id: "heatmap",
          type: "heatmap",
          data: grouped.flatMap(({ values }, i) =>
            sizeValues(values).flatMap((v, j) => (v ? [[j, i, v.mean, v.std, v.n]] : [])),
          ),
          label: {
            show: true,
            color: colors.ink,
            backgroundColor: colors.surface,
            padding: 3,
            borderRadius: 3,
            formatter: (p) => {
              const data = p.value as number[];
              return `${data[2].toFixed(2)}\n±${data[3].toFixed(2)}`;
            },
          },
          itemStyle: { borderColor: colors.surface, borderWidth: 2 },
        },
      ],
    };
  const radarMax = Math.max(1, ...grouped.flatMap((g) => g.values.map((v) => v.mean + v.std)));
  if (type === "radar")
    return {
      radar: {
        radius: "66%",
        indicator: domain.map((size) => ({ name: String(size), max: radarMax })),
        axisName: { color: colors.muted },
      },
      tooltip: {
        ...itemTip,
        formatter: (p: unknown) => {
          const { seriesName } = p as { seriesName: string };
          const group = grouped.find((g) => labelFor(g.arm) === seriesName);
          return group
            ? `${seriesName}\n${group.values.map((v) => `${v.size} rows: ${formatPoint(v.mean, v.std, v.n)}`).join("\n")}`
            : seriesName;
        },
      },
      series: grouped.flatMap<SeriesOption>(({ arm, values }) => {
        const points = sizeValues(values);
        if (points.some((p) => !p)) return [];
        return [
          {
            id: arm,
            type: "radar",
            name: labelFor(arm),
            itemStyle: { color: colorFor(arm) },
            lineStyle: { width: 2 },
            data: [{ value: points.map((p) => p!.mean) }],
            universalTransition: true,
          },
          radarWhiskers(
            `${arm}-radar-whiskers`,
            colorFor(arm),
            points.map((p) => p!),
            radarMax,
          ),
        ];
      }),
    };
  const bars = type === "bar";
  const series: SeriesOption[] = grouped.flatMap(({ arm, values }, i) => {
    const points = sizeValues(values),
      color = colorFor(arm);
    const main: SeriesOption = {
      id: arm,
      name: labelFor(arm),
      type: bars ? "bar" : "line",
      data: points.map((v) => (v ? { value: v.mean, std: v.std, n: v.n } : null)),
      itemStyle: { color },
      symbolSize: 7,
      universalTransition: true,
      ...(bars
        ? { barMaxWidth: 20, barGap: 0, animationDelay: (i: number) => Math.min(i * 35, 180) }
        : {
            lineStyle: { color, width: 2 },
            areaStyle: type === "area" ? { color, opacity: 0.09 } : undefined,
          }),
    };
    if (bars || type === "slope")
      return [
        main,
        errorBars(
          `${arm}-error`,
          color,
          points.flatMap((v, j) => (v ? [[j, v.mean - v.std, v.mean + v.std]] : [])),
          bars ? i - (grouped.length - 1) / 2 : 0,
          grouped.length,
        ),
      ];
    return [
      {
        id: `${arm}-lower`,
        type: "line",
        stack: `band-${arm}`,
        stackStrategy: "all",
        silent: true,
        tooltip: { show: false },
        lineStyle: { opacity: 0 },
        symbol: "none",
        data: points.map((v) => (v ? v.mean - v.std : null)),
      },
      {
        id: `${arm}-upper`,
        type: "line",
        stack: `band-${arm}`,
        stackStrategy: "all",
        silent: true,
        tooltip: { show: false },
        lineStyle: { opacity: 0 },
        areaStyle: { color, opacity: 0.14 },
        symbol: "none",
        data: points.map((v) => (v ? v.std * 2 : null)),
      },
      main,
    ];
  });
  if (reference !== undefined && selected.includes("TRTR-full"))
    series.push({
      id: "reference",
      name: "TRTR-full reference",
      type: "line",
      lineStyle: { color: colors.slate, type: "dashed" },
      itemStyle: { color: colors.slate },
      symbol: "none",
      data: domain.map(() => reference),
    });
  if (random !== undefined)
    series.push({
      id: "random",
      name: "Random-guess PR-AUC",
      type: "line",
      lineStyle: { color: colors.muted, type: "dotted" },
      itemStyle: { color: colors.muted },
      symbol: "none",
      data: domain.map(() => random),
    });
  return {
    grid,
    xAxis: categoryAxis(domain),
    yAxis: valueAxis,
    tooltip: {
      ...axisTip,
      formatter: (p: unknown) => {
        const items = p as {
          axisValue: string;
          seriesName: string;
          data?: number | { value: number; std: number; n: number };
        }[];
        return [
          `${items[0]?.axisValue ?? ""} rows`,
          ...items.flatMap((i) =>
            typeof i.data === "number"
              ? `${i.seriesName}: ${i.data.toFixed(3)}`
              : i.data?.n !== undefined
                ? `${i.seriesName}: ${formatPoint(i.data.value, i.data.std, i.data.n)}`
                : [],
          ),
        ].join("\n");
      },
    },
    series,
  };
}
