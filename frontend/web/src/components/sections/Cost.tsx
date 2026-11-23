import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Pause, Play, RotateCcw } from "lucide-react";
import { api } from "../../api/client";
import { colors } from "../../lib/theme";
import { generatorColor, generatorLabel, seedSizes } from "../../lib/study";
import { formatNumber } from "../../lib/transforms";
import { Reveal, SectionHeading } from "../ui/Primitives";
import { Control, QueryState } from "../ui/QueryState";
import { ResearchChart } from "../charts/ResearchChart";
import {
  axisTip,
  categoryAxis,
  errorBars,
  formatPoint,
  grid,
  legend,
  valueAxis,
} from "../charts/options";
import type { SeriesOption } from "echarts";
export function Cost() {
  const q = useQuery({ queryKey: ["runtime"], queryFn: api.runtime });
  const seeds = useQuery({ queryKey: ["seeds"], queryFn: api.seeds });
  const rows = q.data ?? [];
  const [picked, setSize] = useState<number>(),
    [random, setRun] = useState<number>(),
    [cursor, setCursor] = useState<number>(),
    [playing, setPlaying] = useState(false);
  const sizes = seeds.data?.sizes ?? seedSizes,
    runs = seeds.data?.random_seeds ?? [];
  const size = picked ?? sizes.at(-1)!,
    run = random ?? runs[0];
  const loss = useQuery({
    queryKey: ["loss", size, run],
    queryFn: () => api.loss(size, run),
    enabled: run !== undefined,
  });
  const steps = loss.data?.step ?? [];
  const length = steps.length,
    end = cursor ?? length;
  useEffect(() => {
    if (!playing || !length) return;
    const tick = window.setInterval(() => {
      setCursor((value) => Math.min((value ?? 0) + Math.max(1, Math.ceil(length / 35)), length));
    }, 32);
    return () => clearInterval(tick);
  }, [playing, length]);
  useEffect(() => {
    if (playing && end >= length) setPlaying(false);
  }, [playing, end, length]);
  function reset() {
    setPlaying(false);
    setCursor(undefined);
  }
  const names = [...new Set(rows.map((r) => r.generator))];
  const axis = [...new Set(rows.map((r) => r.seed_size))].sort((a, b) => a - b);
  const series: SeriesOption[] = names.flatMap((generator) => {
    const color = generatorColor(generator),
      r = rows.filter((row) => row.generator === generator);
    return [
      {
        id: generator,
        name: generatorLabel(generator),
        type: "line",
        data: axis.map((size) => {
          const p = r.find((x) => x.seed_size === size);
          return p ? { value: p.seconds_mean, std: p.seconds_std } : null;
        }),
        itemStyle: { color },
        lineStyle: { color },
        universalTransition: true,
      },
      errorBars(
        `${generator}-sd`,
        color,
        r.map((p) => [
          axis.indexOf(p.seed_size),
          Math.max(0, p.seconds_mean - p.seconds_std),
          p.seconds_mean + p.seconds_std,
        ]),
      ),
    ];
  });
  const ct = rows.filter((r) => /ctgan/i.test(r.generator)).map((r) => r.seconds_mean);
  return (
    <section id="cost" className="section soft-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="08 / COST & FAIRNESS"
            title="Every experiment has a cost."
            description="Look at runtime alongside model quality. CTGAN used a fixed step budget, so its training effort doesn’t grow automatically with seed size."
          />
        </Reveal>
        <div className="split-charts">
          <ResearchChart
            title="Runtime by generator"
            subtitle="Mean seconds across runs; whiskers show ±1 standard deviation."
            summary={
              ct.length > 1
                ? `CTGAN mean runtimes ranged from ${formatNumber(Math.min(...ct), 1)} to ${formatNumber(Math.max(...ct), 1)} seconds across tested sizes. Runtime alone does not prove the budget was fixed.`
                : "Runtime across seed sizes, in seconds."
            }
            option={
              rows.length
                ? {
                    legend,
                    grid: { ...grid, bottom: 64 },
                    xAxis: categoryAxis(axis),
                    yAxis: valueAxis,
                    tooltip: {
                      ...axisTip,
                      formatter: (p: unknown) =>
                        (p as { seriesName: string; data: { value: number; std: number } | null }[])
                          .flatMap((v) =>
                            v.data
                              ? `${v.seriesName}: ${formatPoint(v.data.value, v.data.std)} s`
                              : [],
                          )
                          .join("\n"),
                    },
                    series,
                  }
                : undefined
            }
          >
            {!rows.length && (
              <QueryState query={q} message="Runtime results are not supplied in mock mode." />
            )}
          </ResearchChart>
          <ResearchChart
            title="CTGAN training trace"
            subtitle="Replay the reported generator and discriminator losses."
            summary="Both loss curves are reported against training step. Loss alone does not establish synthetic-data quality."
            option={
              length && loss.data
                ? {
                    color: [colors.indigo, colors.rose],
                    animation: false,
                    legend,
                    grid: { ...grid, bottom: 62 },
                    xAxis: categoryAxis(steps),
                    yAxis: valueAxis,
                    tooltip: axisTip,
                    series: [
                      {
                        id: "g",
                        name: "Generator",
                        type: "line",
                        showSymbol: false,
                        data: loss.data.generator_loss.slice(0, end),
                      },
                      {
                        id: "d",
                        name: "Discriminator",
                        type: "line",
                        showSymbol: false,
                        data: loss.data.discriminator_loss.slice(0, end),
                      },
                    ],
                  }
                : undefined
            }
          >
            <div className="filter-strip inline-controls">
              <Control
                label="Training size"
                value={size}
                options={sizes}
                onChange={(v) => {
                  setSize(+v);
                  reset();
                }}
              />
              <Control
                label="Run seed"
                value={run ?? ""}
                options={runs}
                onChange={(v) => {
                  setRun(+v);
                  reset();
                }}
              />
              <button
                className="icon-button"
                aria-label={
                  playing
                    ? "Pause loss trace"
                    : end === length
                      ? "Replay loss trace"
                      : "Play loss trace"
                }
                disabled={!length}
                onClick={() => {
                  if (end === length) setCursor(0);
                  setPlaying(!playing);
                }}
              >
                {playing ? (
                  <Pause size={15} />
                ) : end === length ? (
                  <RotateCcw size={15} />
                ) : (
                  <Play size={15} />
                )}
              </button>
            </div>
            {!length && (
              <QueryState query={loss} message="No CTGAN loss trace is available for this run." />
            )}
          </ResearchChart>
        </div>
      </div>
    </section>
  );
}
