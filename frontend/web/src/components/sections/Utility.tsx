import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity } from "lucide-react";
import { api } from "../../api/client";
import type { Model, Metric } from "../../api/types";
import { arms, colors } from "../../lib/theme";
import { formatNumber, ratio, utilityInsights } from "../../lib/transforms";
import { generators, generatorLabel, generatorColor, utilityThreshold } from "../../lib/study";
import { Empty, Reveal, SectionHeading } from "../ui/Primitives";
import { Control, QueryState } from "../ui/QueryState";
import { ResearchChart } from "../charts/ResearchChart";
import { ArmLegend } from "../charts/ArmLegend";
import { utilityOption, utilityViews, type UtilityView } from "../charts/UtilityChart";
import { GaugeCard } from "../cards/GaugeCard";
const metrics: Record<Metric, string> = {
  pr_auc: "PR-AUC",
  roc_auc: "ROC-AUC",
  f1: "F1 score",
  precision: "Precision",
  recall: "Recall",
  accuracy: "Accuracy",
};
export function Utility() {
  const [model, setModel] = useState<Model>("xgboost"),
    [metric, setMetric] = useState<Metric>("pr_auc"),
    [view, setView] = useState<UtilityView>("line");
  const [selected, setSelected] = useState(Object.keys(arms));
  const [picked, setPicked] = useState<number>();
  const q = useQuery({ queryKey: ["summary", model], queryFn: () => api.summary(model) });
  const ref = useQuery({ queryKey: ["reference"], queryFn: api.reference });
  const rows = q.data ?? [];
  const sizes = [
    ...new Set(rows.filter((r) => r.arm.startsWith("TSTR-")).map((r) => r.seed_size)),
  ].sort((a, b) => a - b);
  const size = sizes.includes(picked ?? -1) ? picked : sizes.at(-1);
  const insight = utilityInsights(rows, metric);
  const draw = rows.length > 0 && selected.length > 0;
  const detail =
    view === "radar"
      ? "Radial whiskers show the mean ±1 standard deviation. Arms with incomplete size coverage are omitted."
      : view === "heatmap"
        ? "Each cell shows the mean and its standard deviation. Whiskers below show the same intervals."
        : "Each score is shown with its mean ±1 standard deviation. Hover for the number of runs.";
  return (
    <section id="utility" className="section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="04 / UTILITY LAB"
            title="Do the models actually learn?"
            description="Realistic-looking data is one thing. Useful data is another. Here's where we put it to the test."
          />
        </Reveal>
        <div className="utility-toolbar">
          <div className="toggle" aria-label="Classifier">
            {(["xgboost", "random_forest"] as const).map((m) => (
              <button
                className={model === m ? "on" : ""}
                aria-pressed={model === m}
                key={m}
                onClick={() => setModel(m)}
              >
                {m === "xgboost" ? "XGBoost" : "Random Forest"}
              </button>
            ))}
          </div>
          <div className="toolbar-right">
            <Control
              label="Metric"
              value={metric}
              options={Object.entries(metrics).map(([value, label]) => ({ value, label }))}
              onChange={(v) => setMetric(v as Metric)}
            />
            <Control
              label="View"
              value={view}
              options={Object.entries(utilityViews).map(([value, label]) => ({ value, label }))}
              onChange={(v) => setView(v as UtilityView)}
            />
          </div>
        </div>
        <ResearchChart
          title="Model performance by seed size"
          subtitle={detail}
          summary={`Utility comparison for ${metrics[metric]} by seed size. ${draw ? "Uncertainty shown as ±1 standard deviation." : "Measurements unavailable for the selected view."}`}
          height={340}
          option={
            draw && view !== "small"
              ? utilityOption(rows, metric, selected, ref.data, model, view)
              : undefined
          }
        >
          <ArmLegend
            selected={selected}
            onToggle={(arm) =>
              setSelected((s) => (s.includes(arm) ? s.filter((a) => a !== arm) : [...s, arm]))
            }
          />
          {draw && ref.data && (
            <p className="reference-key">
              {selected.includes("TRTR-full") && "Dashed slate: TRTR-full reference."}{" "}
              {metric === "pr_auc" && "Dotted: random-guess PR-AUC."}
            </p>
          )}
          {!rows.length ? (
            <QueryState
              query={q}
              message="The study’s performance measurements aren’t in this workspace. Connect the research API to compare arms."
            />
          ) : !selected.length ? (
            <Empty
              title="No arms selected"
              message="Select an experiment arm above to see its results."
            />
          ) : view === "small" ? (
            <div className="split-charts">
              {selected
                .filter((a) => rows.some((r) => r.arm === a) || (a === "TRTR-full" && ref.data))
                .map((a) => (
                  <ResearchChart
                    key={a}
                    title={arms[a].label}
                    subtitle="Seed size · model score"
                    summary={`Mean and uncertainty for ${arms[a].label}, where reported.`}
                    height={180}
                    option={utilityOption(rows, metric, [a], ref.data, model, "line")}
                  />
                ))}
            </div>
          ) : null}
        </ResearchChart>
        {draw && view === "heatmap" && (
          <div className="section-stack">
            <ResearchChart
              title="Intervals behind the heatmap"
              subtitle="Whiskers show the mean ±1 standard deviation."
              summary="Uncertainty for the arms shown above."
              option={utilityOption(rows, metric, selected, undefined, model, "bar")}
            />
          </div>
        )}
        <div className="subhead">
          <div>
            <span className="section-kicker">NORMALIZED TO REAL DATA</span>
            <h3>Utility ratio</h3>
          </div>
          <Control
            label="Seed size"
            value={size ?? ""}
            options={sizes}
            onChange={(v) => setPicked(Number(v))}
          />
        </div>
        <div className="split-charts">
          {generators.map((g) => {
            const full = ref.data?.trtr_full[model].pr_auc;
            const synth = rows.find((r) => r.arm === `TSTR-${g}` && r.seed_size === size)
              ?.metric_means.pr_auc;
            const value =
              full === undefined || synth === undefined
                ? undefined
                : (ratio(synth, full) ?? undefined);
            return (
              <GaugeCard
                key={g}
                title={generatorLabel(g)}
                value={value}
                threshold={utilityThreshold}
                passed={value === undefined ? undefined : value >= utilityThreshold}
                color={generatorColor(g)}
                detail="PR-AUC / TRTR-full PR-AUC"
              />
            );
          })}
        </div>
        <div className="utility-callout">
          <span className="callout-icon">
            <Activity size={20} />
          </span>
          <div>
            <strong>What the data says</strong>
            {insight.best.length ? (
              <>
                <p>
                  Highest mean by size (excluding the full-data reference):{" "}
                  {insight.best
                    .map(
                      (b) =>
                        `${b.size}: ${arms[b.arm]?.label ?? b.arm} (${formatNumber(b.mean, 3)})`,
                    )
                    .join(" · ")}
                  .
                </p>
                {insight.flattest && (
                  <p>
                    CTGAN’s smallest adjacent change: {insight.flattest.start} →{" "}
                    {insight.flattest.end} rows ({formatNumber(insight.flattest.change, 3)}). This
                    is descriptive, not a test for a plateau.
                  </p>
                )}
                {insight.gaps.map((g) => (
                  <p key={g.arm}>
                    {arms[g.arm].label} minus TRTR-seed at {g.size} rows: {formatNumber(g.delta, 3)}
                    .
                  </p>
                ))}
              </>
            ) : (
              <p>
                Best arms, CTGAN’s smallest gain or loss, and the gap to real seeds will appear once
                measurements are available.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
