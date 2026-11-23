import type { CustomSeriesOption, EChartsOption } from "echarts";
import { colors } from "../../lib/theme";
import { formatNumber } from "../../lib/transforms";
export const grid = { left: 48, right: 20, top: 24, bottom: 44 };
export const categoryAxis = (labels: (string | number)[]) => ({
  type: "category" as const,
  data: labels,
  axisTick: { show: false },
  axisLine: { lineStyle: { color: colors.border } },
  axisLabel: { color: colors.muted, hideOverlap: true },
});
export const valueAxis = {
  type: "value" as const,
  axisLabel: { color: colors.muted },
  splitLine: { lineStyle: { color: colors.border } },
};
export const legend = { bottom: 2, textStyle: { color: colors.muted } };
export const axisTip = { trigger: "axis" as const, confine: true, renderMode: "richText" as const };
export const itemTip = { ...axisTip, trigger: "item" as const };
export const formatPoint = (mean: number, sd: number, n?: number) =>
  `${formatNumber(mean, 3)} ± ${formatNumber(sd, 3)}${n === undefined ? "" : ` (n=${n})`}`;
export function errorBars(
  id: string,
  color: string,
  values: number[][],
  offset = 0,
  count = 1,
): CustomSeriesOption {
  return {
    id,
    type: "custom",
    silent: true,
    tooltip: { show: false },
    data: values,
    encode: { x: 0, y: [1, 2] },
    z: 9,
    renderItem: (_, api) => {
      const low = api.coord([api.value(0), api.value(1)]),
        high = api.coord([api.value(0), api.value(2)]);
      const band = (api.size?.([1, 0]) as number[] | undefined)?.[0] ?? 0;
      const x = low[0] + offset * Math.min(20, (band * 0.8) / count);
      const cap = 4;
      return {
        type: "group",
        children: [
          [x, low[1], x, high[1]],
          [x - cap, low[1], x + cap, low[1]],
          [x - cap, high[1], x + cap, high[1]],
        ].map(([x1, y1, x2, y2]) => ({
          type: "line",
          shape: { x1, y1, x2, y2 },
          style: { stroke: color, lineWidth: 1.5 },
        })),
      };
    },
  };
}
export function comparisonOption(
  labels: (number | string)[],
  real: number[],
  synthetic: number[],
  color: string,
  mode: string,
  realLabel = "Real",
): EChartsOption {
  const syntheticLabel = "Synthetic";
  const both = [real, synthetic];
  const names = [realLabel, syntheticLabel],
    palette = [colors.slate, color];
  const cumulative = (counts: number[]) => {
    const sum = counts.reduce((a, b) => a + b, 0);
    let total = 0;
    return counts.map((c) => (sum ? (total += c) / sum : 0));
  };
  const peak = Math.max(...both.flat(), 1);
  return {
    color: palette,
    grid: { ...grid, bottom: 65 },
    tooltip: axisTip,
    legend,
    xAxis: categoryAxis(labels),
    yAxis: {
      ...valueAxis,
      max: mode === "ecdf" ? 1 : undefined,
      axisLabel: { show: mode !== "ridge", formatter: mode === "ecdf" ? "{value}" : "" },
    },
    series: both.map((counts, i) => {
      const common = { id: names[i], name: names[i], universalTransition: true };
      return mode === "ecdf" || mode === "ridge"
        ? {
            ...common,
            type: "line",
            data: mode === "ecdf" ? cumulative(counts) : counts.map((c) => c / peak + i * 1.15),
            step: mode === "ecdf" ? "end" : false,
            smooth: mode === "ridge",
            showSymbol: false,
            areaStyle: mode === "ridge" ? { origin: i * 1.15, opacity: 0.24 } : undefined,
          }
        : {
            ...common,
            type: "bar",
            data: counts,
            barGap: mode === "overlay" ? "-100%" : "15%",
            itemStyle: { opacity: mode === "overlay" ? 0.55 : 1, borderRadius: [3, 3, 0, 0] },
          };
    }),
  };
}

// ECharts radar angles start at twelve o'clock and increase counterclockwise.
export function radarWhiskers(
  id: string,
  color: string,
  points: { mean: number; std: number }[],
  max: number,
): CustomSeriesOption {
  return {
    id,
    type: "custom",
    coordinateSystem: "none",
    silent: true,
    tooltip: { show: false },
    clip: false,
    z: 8,
    data: points.map(({ mean, std }, i) => [i, Math.max(0, mean - std), mean + std]),
    renderItem: (_, api) => {
      const angle = Math.PI / 2 + (Number(api.value(0)) * Math.PI * 2) / points.length;
      const dx = Math.cos(angle),
        dy = -Math.sin(angle);
      const scale = (Math.min(api.getWidth(), api.getHeight()) * 0.33) / max;
      const cx = api.getWidth() / 2,
        cy = api.getHeight() / 2;
      const a = Number(api.value(1)) * scale,
        b = Number(api.value(2)) * scale;
      const low = [cx + dx * a, cy + dy * a],
        high = [cx + dx * b, cy + dy * b];
      const cap = 2.5;
      return {
        type: "group",
        children: [
          [...low, ...high],
          [low[0] - dy * cap, low[1] + dx * cap, low[0] + dy * cap, low[1] - dx * cap],
          [high[0] - dy * cap, high[1] + dx * cap, high[0] + dy * cap, high[1] - dx * cap],
        ].map(([x1, y1, x2, y2]) => ({
          type: "line",
          shape: { x1, y1, x2, y2 },
          style: { stroke: color, lineWidth: 1.2, opacity: 0.65 },
        })),
      };
    },
  };
}
