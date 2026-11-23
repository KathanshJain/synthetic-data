import type { Metric, Summary } from "../api/types";
export function byArm(rows: Summary[], metric: Metric) {
  return Object.entries(
    rows.reduce<Record<string, Summary[]>>((acc, row) => {
      (acc[row.arm] ||= []).push(row);
      return acc;
    }, {}),
  ).map(([arm, values]) => ({
    arm,
    values: values
      .slice()
      .sort((a, b) => a.seed_size - b.seed_size)
      .map((row) => ({
        size: row.seed_size,
        mean: row.metric_means[metric],
        std: row.metric_stds[metric],
        n: row.n_runs,
      })),
  }));
}
export function formatNumber(value: number | undefined, digits = 0) {
  return value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      }).format(value);
}
export function ratio(numerator: number, denominator: number) {
  return denominator === 0 ? null : numerator / denominator;
}

export function utilityInsights(rows: Summary[], metric: Metric) {
  const grouped = byArm(rows, metric);
  const sizes = [...new Set(rows.map((r) => r.seed_size))].sort((a, b) => a - b);
  const means = (arm: string) => grouped.find((g) => g.arm === arm)?.values ?? [];
  const best = sizes.flatMap((size) => {
    const candidates = grouped
      .filter((g) => g.arm !== "TRTR-full")
      .flatMap((g) =>
        g.values.filter((v) => v.size === size).map((v) => ({ arm: g.arm, mean: v.mean, size })),
      );
    return candidates.sort((a, b) => b.mean - a.mean).slice(0, 1);
  });
  const ctgan = means("TSTR-CTGAN");
  const flattest = ctgan
    .slice(1)
    .map((v, i) => ({ start: ctgan[i].size, end: v.size, change: v.mean - ctgan[i].mean }))
    .sort((a, b) => Math.abs(a.change) - Math.abs(b.change))[0];
  const gaps = ["TSTR-CTGAN", "TSTR-GaussianCopula"].flatMap((arm) => {
    const a = means(arm);
    const b = means("TRTR-seed");
    const match = a.filter((v) => b.some((k) => k.size === v.size)).at(-1);
    return match
      ? [{ arm, size: match.size, delta: match.mean - b.find((v) => v.size === match.size)!.mean }]
      : [];
  });
  return { best, flattest, gaps };
}
