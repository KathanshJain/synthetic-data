import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  BarChart3,
  Braces,
  Database,
  Fingerprint,
  Layers3,
  PieChart,
  X,
} from "lucide-react";
import { api } from "../../api/client";
import { colors } from "../../lib/theme";
import { StatCard } from "../cards/StatCard";
import { ResearchChart } from "../charts/ResearchChart";
import { categoryAxis, grid, itemTip, valueAxis, axisTip } from "../charts/options";
import type { PlotOption } from "../charts/Plot";
import { Reveal, SectionHeading, transition } from "../ui/Primitives";
import { Control, QueryState } from "../ui/QueryState";
function Mini({
  values,
  ring = false,
  horizontal = false,
  line = false,
}: {
  values: number[];
  ring?: boolean;
  horizontal?: boolean;
  line?: boolean;
}) {
  const reduced = useReducedMotion();
  const max = Math.max(...values, 1);
  const total = values.reduce((a, b) => a + b, 0);
  const portion = total ? values[0] / total : 0;
  if (line && values.length > 1) {
    const low = Math.min(...values),
      span = Math.max(1, max - low);
    const d = values
      .map(
        (v, i) =>
          `${i ? "L" : "M"} ${4 + (i / (values.length - 1)) * 58} ${27 - ((v - low) / span) * 19}`,
      )
      .join(" ");
    return (
      <svg className="stat-sparkline" viewBox="0 0 66 33" aria-hidden="true">
        <motion.path
          d={d}
          fill="none"
          stroke={colors.indigo}
          strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray="67"
          initial={reduced ? false : { strokeDashoffset: "67" }}
          whileInView={{ strokeDashoffset: "0" }}
          viewport={{ once: true }}
          transition={transition}
        />
      </svg>
    );
  }
  if (horizontal)
    return (
      <span className="stat-mini-hbars" aria-hidden="true">
        {(values.length ? values.slice(0, 4) : [0, 0, 0]).map((v, i) => (
          <motion.i
            key={i}
            style={{ width: reduced ? `${Math.max(7, (v / max) * 100)}%` : undefined }}
            initial={reduced ? false : { width: 0 }}
            whileInView={{ width: `${Math.max(7, (v / max) * 100)}%` }}
            viewport={{ once: true }}
            transition={{ ...transition, delay: reduced ? 0 : i * 0.06 }}
          />
        ))}
      </span>
    );
  return ring ? (
    <svg className="stat-mini-ring" viewBox="0 0 48 48" aria-hidden="true">
      <circle cx="24" cy="24" r="17" fill="none" stroke={colors.softIndigo} strokeWidth="7" />
      <motion.circle
        cx="24"
        cy="24"
        r="17"
        fill="none"
        stroke={total ? colors.indigo : colors.border}
        strokeWidth="7"
        strokeLinecap="round"
        transform="rotate(-90 24 24)"
        initial={false}
        whileInView={{ strokeDasharray: `${portion * 106.8} 107` }}
        style={{ strokeDasharray: `${reduced ? portion * 106.8 : 0} 107` }}
        viewport={{ once: true }}
        transition={transition}
      />
    </svg>
  ) : (
    <span className="stat-mini-bars" aria-hidden="true">
      {(values.length ? values : [0, 0, 0, 0]).map((v, i) => (
        <motion.i
          key={i}
          initial={false}
          style={{ height: `${reduced ? Math.max(8, (v / max) * 100) : 8}%` }}
          whileInView={{ height: `${Math.max(8, (v / max) * 100)}%` }}
          viewport={{ once: true }}
          transition={{ ...transition, delay: reduced ? 0 : Math.min(0.2, i * 0.04) }}
        />
      ))}
    </span>
  );
}
export function DataSnapshot() {
  const query = useQuery({ queryKey: ["overview"], queryFn: api.overview });
  const overview = query.data;
  const [column, setColumn] = useState(""),
    [open, setOpen] = useState<string>();
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = dialog.current;
    if (open && d && !d.open) d.showModal();
    if (!open && d?.open) d.close();
  }, [open]);
  const fields = overview?.features ?? [];
  const selected = fields.some((f) => f.name === column) ? column : (fields[0]?.name ?? "");
  const dist = useQuery({
    queryKey: ["distribution", selected],
    queryFn: () => api.distribution(selected),
    enabled: !!selected,
  });
  const mostMissing = fields
    .slice()
    .sort((a, b) => b.missing_pct - a.missing_pct)
    .slice(0, 7);
  const categories = fields
    .filter((f) => /categor|bool|object|string/i.test(f.type))
    .sort((a, b) => b.unique - a.unique)
    .slice(0, 7);
  const rates = overview ? [1 - overview.positive_rate_dev, overview.positive_rate_dev] : [];
  const cards = [
    {
      id: "rows",
      label: "Rows",
      value: overview?.rows_clean,
      detail: "Raw → clean → development and test",
      icon: <Database size={18} />,
      labels: ["Raw", "Clean", "Development", "Locked test"],
      values: overview
        ? [overview.rows_raw, overview.rows_clean, overview.dev_rows, overview.test_rows]
        : [],
      wide: true,
    },
    {
      id: "features",
      label: "Features",
      value: overview?.n_features,
      detail: "Numerical and categorical",
      icon: <Braces size={18} />,
      labels: ["Numerical", "Categorical"],
      values: overview ? [overview.n_numerical, overview.n_categorical] : [],
      ring: true,
    },
    {
      id: "balance",
      label: "Class balance",
      value: overview ? (1 - overview.positive_rate_dev) * 100 : undefined,
      detail: "Negative share in the development pool",
      icon: <PieChart size={18} />,
      labels: ["Negative", "Positive"],
      values: rates.map((v) => v * 100),
      ring: true,
      suffix: "%",
      digits: 1,
    },
    {
      id: "missing",
      label: "Missing values",
      value: overview ? fields.filter((f) => f.missing_pct > 0).length : undefined,
      detail: "Features with missing entries",
      icon: <BarChart3 size={18} />,
      labels: mostMissing.map((f) => f.name),
      values: mostMissing.map((f) => f.missing_pct),
      horizontal: true,
    },
    {
      id: "duplicates",
      label: "Duplicates removed",
      value: overview?.duplicates_removed,
      detail: "Raw vs. cleaned row count",
      icon: <Layers3 size={18} />,
      labels: ["Raw", "After deduplication"],
      values: overview ? [overview.rows_raw, overview.rows_raw - overview.duplicates_removed] : [],
      line: true,
    },
    {
      id: "positive",
      label: "Positive rate",
      value: overview ? overview.positive_rate_dev * 100 : undefined,
      detail: "Development vs. locked test",
      icon: <Activity size={18} />,
      labels: ["Development", "Locked test"],
      values: overview ? [overview.positive_rate_dev * 100, overview.positive_rate_test * 100] : [],
      ring: true,
      suffix: "%",
      digits: 1,
    },
    {
      id: "cardinality",
      label: "Categorical cardinality",
      value: categories[0]?.unique,
      detail: "Most distinct values in a categorical feature",
      icon: <Fingerprint size={18} />,
      labels: categories.map((f) => f.name),
      values: categories.map((f) => f.unique),
      horizontal: true,
    },
  ];
  const active = cards.find((c) => c.id === open);
  const drawerOption: PlotOption | undefined = active?.values.length
    ? active.ring && active.id !== "positive"
      ? {
          color: [colors.indigo, colors.teal],
          tooltip: itemTip,
          legend: { bottom: 8 },
          series: [
            {
              type: "pie",
              radius: ["48%", "74%"],
              data: active.values.map((value, i) => ({ value, name: active.labels[i] })),
              label: { show: false },
            },
          ],
        }
      : {
          grid: { ...grid, left: active.horizontal ? 95 : grid.left },
          tooltip: axisTip,
          xAxis: active.horizontal ? valueAxis : categoryAxis(active.labels),
          yAxis: active.horizontal
            ? {
                ...categoryAxis(active.labels),
                inverse: true,
                axisLabel: { width: 82, overflow: "truncate" },
              }
            : valueAxis,
          series: [
            {
              type: active.line ? "line" : "bar",
              data: active.values,
              itemStyle: { color: colors.indigo },
              universalTransition: true,
              realtimeSort: active.id === "cardinality",
            },
          ],
        }
    : undefined;
  return (
    <section id="data" className="section soft-section">
      <div className="container">
        <Reveal>
          <SectionHeading
            eyebrow="01 / THE DATA"
            title="It starts with the real thing."
            description="Before generating anything new, we get to know the data we already have."
          />
        </Reveal>
        {query.isLoading ? (
          <div
            className="skeleton chart-skeleton"
            role="status"
            aria-label="Loading dataset overview"
          />
        ) : (
          <>
            <div className="snapshot-grid">
              {cards.map((c) => (
                <StatCard
                  key={c.id}
                  {...c}
                  onOpen={() => setOpen(c.id)}
                  preview={
                    <Mini
                      values={
                        c.id === "positive" && overview
                          ? [overview.positive_rate_dev, 1 - overview.positive_rate_dev]
                          : c.values
                      }
                      ring={c.ring}
                      horizontal={c.horizontal}
                      line={c.line}
                    />
                  }
                />
              ))}
            </div>
            {!overview && (
              <div className="card section-stack">
                <QueryState
                  query={query}
                  message="Dataset counts and feature metadata were not included with the brief. No values have been substituted."
                />
              </div>
            )}
          </>
        )}
        <div className="section-stack">
          <ResearchChart
            title="Explore a column"
            subtitle="Distributions from the development pool only. The locked test set is excluded."
            summary={
              selected
                ? `Observed counts for ${selected} in the development pool.`
                : "Select a column when the development data is connected."
            }
            height={260}
            option={
              dist.data?.counts.length
                ? {
                    grid,
                    tooltip: axisTip,
                    xAxis: categoryAxis(dist.data.bins ?? dist.data.categories ?? []),
                    yAxis: valueAxis,
                    series: [
                      {
                        id: "dev",
                        type: "bar",
                        data: dist.data.counts,
                        itemStyle: { color: colors.indigo },
                        universalTransition: true,
                      },
                    ],
                  }
                : undefined
            }
          >
            <div className="filter-strip inline-controls">
              <Control
                label="Dataset column"
                value={selected}
                options={fields.map((f) => f.name)}
                onChange={setColumn}
              />
            </div>
            {!dist.data?.counts.length && (
              <QueryState
                query={dist}
                message="Column distributions are not available in the supplied research fixture."
              />
            )}
          </ResearchChart>
        </div>
      </div>
      <dialog
        ref={dialog}
        className="snapshot-drawer"
        aria-labelledby="drawer-title"
        onClose={() => setOpen(undefined)}
        onClick={(e) => {
          if (e.target === dialog.current) setOpen(undefined);
        }}
      >
        {active && (
          <div className="drawer-inner">
            <div className="drawer-head">
              <div>
                <span className="section-kicker">DATA SNAPSHOT</span>
                <h2 id="drawer-title">{active.label}</h2>
              </div>
              <button
                autoFocus
                className="icon-button"
                aria-label="Close dataset details"
                onClick={() => setOpen(undefined)}
              >
                <X size={18} />
              </button>
            </div>
            <ResearchChart
              title={active.detail}
              subtitle="From the dataset overview"
              summary={
                active.values.length
                  ? active.labels.map((l, i) => `${l}: ${active.values[i]}`).join(" · ")
                  : "The source values are unavailable."
              }
              option={drawerOption}
            >
              {!drawerOption && (
                <QueryState query={query} message="The data for this chart hasn’t been supplied." />
              )}
            </ResearchChart>
          </div>
        )}
      </dialog>
    </section>
  );
}
