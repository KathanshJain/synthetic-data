import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Fingerprint, ShieldCheck } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import { api } from "../../api/client";
import { useExperiment } from "../../hooks/useExperiment";
import { colors } from "../../lib/theme";
import { dcrCaveat, dcrThreshold, generatorColor, generatorLabel } from "../../lib/study";
import { formatNumber } from "../../lib/transforms";
import type { Criteria, Fidelity } from "../../api/types";
import { Reveal, SectionHeading, transition } from "../ui/Primitives";
import { Control, QueryState } from "../ui/QueryState";
import { GaugeCard } from "../cards/GaugeCard";
import { ResearchChart } from "../charts/ResearchChart";
import {
  categoryAxis,
  comparisonOption,
  grid,
  itemTip,
  legend,
  valueAxis,
} from "../charts/options";
import type { PlotOption } from "../charts/Plot";
function ExperimentControls({ selection: s }: { selection: ReturnType<typeof useExperiment> }) {
  return (
    <>
      <Control
        label="Generator"
        value={s.generator}
        options={s.generatorOptions.map((value) => ({ value, label: generatorLabel(value) }))}
        onChange={s.setGenerator}
      />
      <Control
        label="Seed size"
        value={s.size}
        options={s.sizes}
        onChange={(v) => s.setSize(Number(v))}
      />
      <Control
        label="Random seed"
        value={s.run ?? ""}
        options={s.runs}
        onChange={(v) => s.setRun(Number(v))}
      />
    </>
  );
}
function ruleDetails(
  criteria: Criteria | undefined,
  generator: string,
  size: number,
  id: string,
  aliases: string[] = [],
) {
  const keys = [id, ...aliases];
  const rule = criteria?.rules.find((r) => keys.includes(r.id));
  const row = criteria?.evaluations.find((e) => e.generator === generator && e.seed_size === size);
  return { threshold: rule?.threshold, passed: rule ? row?.passed[rule.id] : undefined };
}
const metricFields = [
  ["ks_mean", "Mean KS", ["ks"]],
  ["tvd_mean", "Mean TVD", ["tvd"]],
  ["corr_diff", "Correlation difference", ["correlation_difference"]],
  ["coverage", "Category coverage", ["category_coverage"]],
] as const;
function ColumnTable({ rows, color }: { rows: Fidelity["per_column"]; color: string }) {
  const [sort, setSort] = useState<{ key: "column" | "metric" | "value"; asc: boolean }>({
    key: "value",
    asc: false,
  });
  const reduced = useReducedMotion();
  const max = Math.max(1, ...rows.map((r) => r.value));
  const sorted = [...rows].sort((a, b) => {
    const x = a[sort.key],
      y = b[sort.key];
    const d =
      typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y));
    return sort.asc ? d : -d;
  });
  return (
    <div className="card scoreboard">
      <div className="chart-header">
        <div>
          <h3>Per-column fidelity</h3>
          <p>Use any column header to sort the measurements.</p>
        </div>
      </div>
      <div className="score-scroll">
        <table>
          <thead>
            <tr>
              {(["column", "metric", "value"] as const).map((key) => (
                <th
                  key={key}
                  aria-sort={sort.key === key ? (sort.asc ? "ascending" : "descending") : undefined}
                >
                  <button
                    className="table-sort"
                    onClick={() => setSort({ key, asc: sort.key !== key || !sort.asc })}
                  >
                    {key} {sort.key === key ? (sort.asc ? "↑" : "↓") : "↕"}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={`${r.column}-${r.metric}-${i}`}>
                <th scope="row">{r.column}</th>
                <td>{r.metric}</td>
                <td>
                  <div className="table-measure">
                    <span className="gauge-track">
                      <motion.i
                        style={{ background: color }}
                        initial={reduced ? false : { width: 0 }}
                        whileInView={{ width: `${(Math.max(0, r.value) / max) * 100}%` }}
                        viewport={{ once: true }}
                        transition={transition}
                      />
                    </span>
                    {formatNumber(r.value, 3)}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
function matrixOption(
  columns: string[],
  matrix: number[][],
  difference: boolean,
  color: string,
): PlotOption {
  return {
    grid: { left: 62, right: 12, top: 22, bottom: 56 },
    tooltip: itemTip,
    xAxis: {
      ...categoryAxis(columns),
      axisLabel: { rotate: 35, hideOverlap: true, width: 66, overflow: "truncate" },
    },
    yAxis: {
      ...categoryAxis(columns),
      inverse: true,
      axisLabel: { hideOverlap: true, width: 54, overflow: "truncate" },
    },
    visualMap: {
      show: false,
      min: difference ? 0 : -1,
      max: difference ? Math.max(1, ...matrix.flat()) : 1,
      dimension: 2,
      inRange: {
        color: difference ? [colors.surface, color] : [colors.softRose, colors.surface, color],
      },
    },
    series: [
      {
        id: "correlation",
        type: "heatmap",
        data: matrix.flatMap((row, y) =>
          row.flatMap((v, x) => (Number.isFinite(v) ? [[x, y, v]] : [])),
        ),
        itemStyle: { borderColor: colors.surface, borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: colors.ink } },
        tooltip: {
          formatter: (p: unknown) => {
            const [x, y, v] = (p as { value: number[] }).value;
            return `${columns[x]} × ${columns[y]}\n${formatNumber(v, 3)}`;
          },
        },
      },
    ],
  };
}
export function FidelityStudio() {
  const f = useQuery({ queryKey: ["fidelity"], queryFn: api.fidelity });
  const rules = useQuery({ queryKey: ["criteria"], queryFn: api.criteria });
  const overview = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const s = useExperiment(f.data?.overall);
  const current = f.data?.overall.find(
    (r) => r.generator === s.generator && r.seed_size === s.size,
  );
  const per =
    f.data?.per_column.filter((r) => r.generator === s.generator && r.seed_size === s.size) ?? [];
  const columns = [
    ...new Set([
      ...(overview.data?.features.map((f) => f.name) ?? []),
      ...per.map((r) => r.column),
    ]),
  ];
  const [picked, setColumn] = useState(""),
    [view, setView] = useState("overlay"),
    [opacity, setOpacity] = useState(55);
  const column = columns.includes(picked) ? picked : (columns[0] ?? "");
  const key = [s.generator, s.size, s.run] as const;
  const dist = useQuery({
    queryKey: ["compare-dist", column, ...key],
    queryFn: () => api.compare(column, s.generator, s.size, s.run!),
    enabled: !!column && s.run !== undefined,
  });
  const corr = useQuery({
    queryKey: ["compare-corr", ...key],
    queryFn: () => api.correlation(s.generator, s.size, s.run!),
    enabled: s.run !== undefined,
  });
  const pca = useQuery({
    queryKey: ["compare-pca", ...key],
    queryFn: () => api.projection(s.generator, s.size, s.run!),
    enabled: s.run !== undefined,
  });
  const tone = generatorColor(s.generator);
  const categorical = Boolean(dist.data?.categories);
  const options = [
    { value: "overlay", label: "Overlay histogram" },
    { value: "grouped", label: "Side-by-side bars" },
    ...(!categorical
      ? [
          { value: "ridge", label: "Ridgeline" },
          { value: "ecdf", label: "Binned ECDF" },
        ]
      : []),
  ];
  const mode = options.some((o) => o.value === view) ? view : "grouped";
  return (
    <section id="fidelity" className="section soft-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="05 / FIDELITY STUDIO"
            title="Looks similar. But how similar?"
            description="Compare distributions, relationships, and category coverage. Similar averages can hide very different records."
          />
        </Reveal>
        <div className="filter-strip">
          <ExperimentControls selection={s} />
        </div>
        <div className="metric-grid">
          {metricFields.map(([id, label, aliases]) => (
            <GaugeCard
              key={id}
              title={label}
              color={tone}
              value={current?.[id]}
              detail={
                current?.stds[id] !== undefined
                  ? `SD ${formatNumber(current.stds[id], 3)} · Across runs`
                  : "Across random seeds"
              }
              {...ruleDetails(rules.data, s.generator, s.size, id, [...aliases])}
            />
          ))}
        </div>
        {!current && (
          <div className="section-stack card">
            <QueryState
              query={f}
              message="No aggregate fidelity measurements were supplied for this selection."
            />
          </div>
        )}
        {per.length > 0 && <ColumnTable rows={per} color={tone} />}
        <div className="section-stack">
          <ResearchChart
            title="Real vs. synthetic distribution"
            subtitle="Choose an individual feature and compare the real and generated counts."
            summary={
              mode === "ecdf"
                ? "Cumulative proportions calculated from the reported bins, not from raw records."
                : mode === "ridge"
                  ? "Vertically offset shapes, scaled to the same peak for visual comparison."
                  : "Two distributions measured using the same reported bins or categories."
            }
            option={
              dist.data && (dist.data.real.length > 0 || dist.data.synthetic.length > 0)
                ? comparisonOption(
                    dist.data.bins ?? dist.data.categories ?? [],
                    dist.data.real,
                    dist.data.synthetic,
                    tone,
                    mode,
                  )
                : undefined
            }
          >
            <div className="filter-strip inline-controls">
              <Control label="Column" value={column} options={columns} onChange={setColumn} />
              <Control
                label="Distribution view"
                value={mode}
                options={options}
                onChange={setView}
              />
            </div>
            {!dist.data?.real.length && !dist.data?.synthetic.length && (
              <QueryState
                query={dist}
                message="Select a feature and a random seed when comparison data is available."
              />
            )}
          </ResearchChart>
        </div>
        <div className="subhead">
          <div>
            <span className="section-kicker">PAIRWISE RELATIONSHIPS</span>
            <h3>Correlation, side by side</h3>
          </div>
        </div>
        {corr.data?.columns.length ? (
          <div className="three-cards">
            {(["real", "synthetic", "diff"] as const).map((field, i) => (
              <ResearchChart
                key={field}
                title={["Real", "Synthetic", "Absolute difference"][i]}
                subtitle="Features on both axes"
                summary={
                  field === "diff"
                    ? "Absolute correlation difference for each feature pair."
                    : `Correlation matrix for ${field} records.`
                }
                height={235}
                option={matrixOption(
                  corr.data.columns,
                  corr.data[field],
                  field === "diff",
                  field === "real" ? colors.slate : tone,
                )}
              />
            ))}
          </div>
        ) : (
          <div className="card">
            <QueryState query={corr} message="Correlation matrices are unavailable for this run." />
          </div>
        )}
        <div className="section-stack">
          <ResearchChart
            title="A two-dimensional view"
            subtitle="PCA projection of real and synthetic records."
            summary={
              pca.data
                ? `PC1 and PC2 explain ${pca.data.explained_variance
                    .slice(0, 2)
                    .map((v) => `${formatNumber(v * 100, 1)}%`)
                    .join(" and ")} of the variance, respectively.`
                : "No projection points are available for this run."
            }
            option={
              pca.data && (pca.data.real.length || pca.data.synthetic.length)
                ? {
                    grid: { ...grid, bottom: 62 },
                    legend,
                    tooltip: itemTip,
                    xAxis: { ...valueAxis, name: "PC1", nameLocation: "middle", nameGap: 25 },
                    yAxis: { ...valueAxis, name: "PC2" },
                    series: [
                      {
                        id: "real",
                        name: "Real",
                        type: "scatter",
                        data: pca.data.real,
                        symbolSize: 5,
                        itemStyle: { color: colors.slate, opacity: 0.55 },
                      },
                      {
                        id: "synthetic",
                        name: "Synthetic",
                        type: "scatter",
                        data: pca.data.synthetic,
                        symbolSize: 5,
                        itemStyle: { color: tone, opacity: opacity / 100 },
                      },
                    ],
                  }
                : undefined
            }
          >
            <label className="slider-control">
              Synthetic point opacity{" "}
              <input
                type="range"
                min={0}
                max={100}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
              <output>{opacity}%</output>
            </label>
            {!pca.data?.real.length && !pca.data?.synthetic.length && (
              <QueryState
                query={pca}
                message="PCA points appear when a projection is available for the selected run."
              />
            )}
          </ResearchChart>
        </div>
      </div>
    </section>
  );
}
export function PrivacyRadar() {
  const q = useQuery({ queryKey: ["privacy"], queryFn: api.privacy });
  const rules = useQuery({ queryKey: ["criteria"], queryFn: api.criteria });
  const s = useExperiment(q.data);
  const row = q.data?.find((r) => r.generator === s.generator && r.seed_size === s.size);
  const hist = useQuery({
    queryKey: ["dcr", s.generator, s.size, s.run],
    queryFn: () => api.privacyHistogram(s.generator, s.size, s.run!),
    enabled: s.run !== undefined,
  });
  const tone = generatorColor(s.generator),
    rule = ruleDetails(rules.data, s.generator, s.size, "dcr_ratio");
  return (
    <section id="privacy" className="section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="06 / PRIVACY RADAR"
            title="Similar shouldn't mean identical."
            description="Distance to closest record (DCR) compares how close synthetic and holdout records are to the training data. Exact copies are checked separately."
          />
        </Reveal>
        <div className="filter-strip">
          <ExperimentControls selection={s} />
        </div>
        <div className="caveat privacy-caveat">
          <ShieldCheck size={16} />
          {dcrCaveat}
        </div>
        <div className="metric-grid">
          <GaugeCard
            title="Synthetic median DCR"
            color={tone}
            value={row?.dcr_median_synth}
            detail="Synthetic → training"
          />
          <GaugeCard
            title="Holdout median DCR"
            color={colors.slate}
            value={row?.dcr_median_holdout}
            detail="Holdout → training"
          />
          <GaugeCard
            title="Synthetic exact-copy rate"
            color={tone}
            value={row?.exact_rate_synth}
            detail="Synthetic → training"
            percent
          />
          <GaugeCard
            title="Holdout exact-copy rate"
            color={colors.slate}
            value={row?.exact_rate_holdout}
            detail="Holdout → training"
            percent
          />
        </div>
        {!row && q.error && (
          <div className="card section-stack">
            <QueryState query={q} message="Aggregate privacy measurements aren’t available." />
          </div>
        )}
        <div className="section-stack privacy-charts">
          <ResearchChart
            title="Distance to closest record"
            subtitle="Compare synthetic and holdout distances for one run."
            summary={dcrCaveat}
            option={
              hist.data?.bins.length
                ? comparisonOption(
                    hist.data.bins,
                    hist.data.holdout,
                    hist.data.synthetic,
                    tone,
                    "overlay",
                    "Real holdout",
                  )
                : undefined
            }
          >
            {!hist.data?.bins.length && (
              <QueryState
                query={hist}
                message="The DCR histogram is unavailable for this selection."
              />
            )}
          </ResearchChart>
          <div className="privacy-side">
            <GaugeCard
              title="DCR ratio"
              color={tone}
              value={row?.dcr_ratio}
              threshold={rule.threshold ?? dcrThreshold}
              passed={
                rule.passed ?? (row ? row.dcr_ratio >= (rule.threshold ?? dcrThreshold) : undefined)
              }
              detail="Synthetic median / holdout median"
            />
            <div className="card privacy-explainer">
              <Fingerprint size={21} />
              <strong>Read the low end</strong>
              <p>
                Short distances can flag possible memorization. They do not establish whether any
                specific record can be identified.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
