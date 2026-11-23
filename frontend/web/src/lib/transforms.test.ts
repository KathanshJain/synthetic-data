import { describe, it, expect } from "vitest";
import { byArm, formatNumber, ratio, utilityInsights } from "./transforms";
import type { Metrics, Summary } from "../api/types";
const metrics = (n: number) =>
  ({ pr_auc: n, roc_auc: n, f1: n, precision: n, recall: n, accuracy: n }) as Metrics;
describe("data transforms", () => {
  it("sorts each arm by seed size and keeps uncertainty", () => {
    const rows = [
      {
        arm: "a",
        seed_size: 500,
        metric_means: metrics(0.7),
        metric_stds: metrics(0.1),
        n_runs: 3,
      },
      {
        arm: "a",
        seed_size: 100,
        metric_means: metrics(0.5),
        metric_stds: metrics(0.2),
        n_runs: 3,
      },
    ] as Summary[];
    expect(byArm(rows, "pr_auc")[0].values.map((v) => [v.size, v.std])).toEqual([
      [100, 0.2],
      [500, 0.1],
    ]);
  });
  it("handles absent and zero denominator", () => {
    expect(formatNumber(undefined)).toBe("—");
    expect(ratio(1, 0)).toBeNull();
    expect(ratio(1, 2)).toBe(0.5);
  });
});

describe("utility insights", () => {
  it("excludes the full baseline, compares like sizes, and finds the smallest absolute gain or loss", () => {
    const row = (arm: string, seed_size: number, mean: number): Summary => ({
      arm,
      seed_size,
      metric_means: metrics(mean),
      metric_stds: metrics(0.02),
      n_runs: 3,
    });
    const result = utilityInsights(
      [
        row("TSTR-CTGAN", 100, 0.4),
        row("TSTR-CTGAN", 250, 0.42),
        row("TSTR-CTGAN", 500, 0.52),
        row("TRTR-seed", 100, 0.45),
        row("TRTR-seed", 500, 0.6),
        row("TRTR-full", 500, 0.95),
      ],
      "pr_auc",
    );
    expect(result.best.map((x) => x.arm)).toEqual(["TRTR-seed", "TSTR-CTGAN", "TRTR-seed"]);
    expect(result.flattest).toMatchObject({ start: 100, end: 250 });
    expect(result.gaps[0].delta).toBeCloseTo(-0.08);
  });
});
