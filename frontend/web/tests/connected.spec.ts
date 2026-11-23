import { expect, test } from "@playwright/test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { Criteria, Overview, Summary } from "../src/api/types";

// Artificial values for browser verification only; not shipped or used by screenshots.
const overview: Overview = {
  rows_raw: 145,
  duplicates_removed: 5,
  rows_clean: 140,
  dev_rows: 110,
  test_rows: 30,
  positive_rate_dev: 0.24,
  positive_rate_test: 0.23,
  n_features: 2,
  n_numerical: 1,
  n_categorical: 1,
  features: [
    { name: "quantity", type: "numerical", unique: 23, missing_pct: 0 },
    { name: "region", type: "categorical", unique: 4, missing_pct: 3 },
  ],
  locked: { sha256: "a".repeat(64), verified: true, created: "2025-01-01", n_rows: 30 },
};
const metrics = (v: number) => ({
  pr_auc: v,
  roc_auc: v,
  f1: v,
  precision: v,
  recall: v,
  accuracy: v,
});
const arms = ["TRTR-seed", "SMOTE-NC", "TSTR-CTGAN", "TSTR-GaussianCopula"];
const sizes = [100, 250, 500, 1000, 2000];
const rows: Summary[] = arms.flatMap((arm, i) =>
  sizes.map((seed_size, j) => ({
    arm,
    seed_size,
    metric_means: metrics(0.4 + i * 0.05 + j * 0.04),
    metric_stds: metrics(0.03),
    n_runs: 3,
  })),
);
const criteria: Criteria = {
  rules: [
    { id: "ks_mean", label: "KS", threshold: 0.2, direction: "≤" },
    { id: "dcr_ratio", label: "DCR ratio", threshold: 0.8, direction: "≥" },
  ],
  evaluations: [
    {
      generator: "CTGAN",
      seed_size: 500,
      values: { ks_mean: 0.19, dcr_ratio: 0.74 },
      passed: { ks_mean: true, dcr_ratio: false },
      all_passed: false,
    },
  ],
  min_useful_seed: null,
  explanation: "Browser test fixture only.",
};
const distribution = { type: "numerical", bins: [0, 5, 10], real: [3, 7, 5], synthetic: [5, 5, 5] };
const result = {
  class_balance: { real: { yes: 2, no: 1 }, synthetic: { yes: 9, no: 5 } },
  fidelity: { ks_mean: 0.19 },
  privacy: { dcr_ratio: 0.74 },
  per_column: { quantity: distribution },
};
const responses: Record<string, unknown> = {
  "/api/overview": overview,
  "/api/health": { status: "ok", version: "test", tests_passing: 6, tests_total: 6 },
  "/api/seeds": {
    sizes,
    random_seeds: [21, 42, 84],
    nested_ok: true,
    class_counts: { 500: { majority: 380, minority: 120 } },
  },
  "/api/dataset/distribution": { type: "numerical", bins: [0, 5, 10], counts: [3, 7, 5] },
  "/api/results/summary": rows,
  "/api/results/reference": {
    trtr_full: { xgboost: metrics(0.89), random_forest: metrics(0.85) },
    random_guess_pr_auc: 0.24,
  },
  "/api/results/criteria": criteria,
  "/api/results/fidelity": {
    overall: [
      {
        generator: "CTGAN",
        seed_size: 500,
        ks_mean: 0.19,
        tvd_mean: 0.14,
        chi2_mean: 1,
        corr_diff: 0.07,
        coverage: 0.91,
        stds: { ks_mean: 0.01 },
      },
    ],
    per_column: [
      { generator: "CTGAN", seed_size: 500, column: "quantity", metric: "KS", value: 0.2 },
      { generator: "CTGAN", seed_size: 500, column: "region", metric: "TVD", value: 0.15 },
    ],
  },
  "/api/compare/distribution": distribution,
  "/api/compare/correlation": {
    columns: ["quantity", "region"],
    real: [
      [1, 0.4],
      [0.4, 1],
    ],
    synthetic: [
      [1, 0.5],
      [0.5, 1],
    ],
    diff: [
      [0, 0.1],
      [0.1, 0],
    ],
  },
  "/api/compare/projection": {
    real: [
      [0, 0],
      [1, 1],
    ],
    synthetic: [
      [0, 0.1],
      [1, 0.6],
    ],
    explained_variance: [0.55, 0.22],
  },
  "/api/results/privacy": [
    {
      generator: "CTGAN",
      seed_size: 500,
      dcr_median_synth: 0.42,
      dcr_median_holdout: 0.55,
      dcr_ratio: 0.74,
      exact_rate_synth: 0,
      exact_rate_holdout: 0,
      p05_synth: 0.01,
      p05_holdout: 0.05,
    },
  ],
  "/api/results/privacy/histogram": { bins: [0, 1, 2], synthetic: [2, 4, 1], holdout: [1, 3, 5] },
  "/api/results/runtime": [
    { generator: "CTGAN", seed_size: 500, seconds_mean: 14, seconds_std: 0.5 },
  ],
  "/api/results/loss-curve": {
    step: [1, 2, 3, 4],
    generator_loss: [1, 0.8, 0.9, 0.6],
    discriminator_loss: [1, 1, 0.7, 0.5],
  },
  "/api/lab/upload": {
    dataset_id: "test",
    profile: {
      rows: 3,
      columns: ["target", "quantity"],
      types: { target: "categorical", quantity: "numerical" },
      missing: { target: 0, quantity: 0 },
      warnings: [],
      class_balance: { yes: 2, no: 1 },
    },
  },
  "/api/lab/run": { job_id: "test" },
  "/api/lab/jobs/test": {
    status: "completed",
    stage: "Finished",
    progress: 100,
    log: ["Finished evaluation"],
    result,
    error: null,
  },
};
test("connected mode renders every chart family and a completed job via polling fallback", async ({
  page,
}) => {
  test.skip(!process.env.TEST_CONNECTED, "Requires the optional connected-mode build");
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("https://research.test/api/**", (route) => {
    const path = new URL(route.request().url()).pathname;
    return path.endsWith("/events")
      ? route.abort()
      : route.fulfill({
          json: responses[path] ?? { message: "Not available" },
          status: path in responses ? 200 : 404,
          headers: {
            "access-control-allow-origin": "*",
            "access-control-allow-headers": "content-type",
            "access-control-allow-methods": "GET, POST, OPTIONS",
          },
        });
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(pathToFileURL(resolve("dist/connected/index.html")).href);
  await expect(page.locator("#vault").getByText("VERIFIED", { exact: true })).toBeVisible();
  await expect(page.locator("#data canvas").first()).toBeVisible();
  const utility = page.locator("#utility");
  for (const [i, mode] of ["line", "bar", "area", "radar", "slope", "heatmap", "small"].entries()) {
    await utility.getByLabel("View", { exact: true }).selectOption(mode);
    await expect(utility.locator("canvas").first()).toBeVisible();
    if (i === 5)
      await expect(
        utility.getByRole("heading", { name: "Intervals behind the heatmap" }),
      ).toBeVisible();
  }
  await expect(page.getByRole("button", { name: "Export chart as PNG" }).first()).toBeEnabled();
  await expect(page.locator("#fidelity canvas")).toHaveCount(5);
  await page.locator("#fidelity").getByLabel("Distribution view").selectOption("ecdf");
  await page
    .locator("#fidelity")
    .getByRole("button", { name: /^value/i })
    .click();
  await expect(page.locator("#privacy canvas")).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "Frozen criteria scoreboard" })).toBeVisible();
  await expect(page.locator("#cost canvas")).toHaveCount(2);
  await page.getByRole("button", { name: "Replay loss trace" }).click();
  await page
    .locator("#lab input[type=file]")
    .setInputFiles({
      name: "upload.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("target,quantity\nyes,1\nno,2\nyes,3"),
    });
  await page.getByRole("button", { name: "Run experiment" }).click();
  await expect(page.getByRole("link", { name: "Download CSV" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Feature comparison" })).toBeVisible();
  expect(errors).toEqual([]);
});
