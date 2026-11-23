import type { Criteria, Fidelity, Privacy, Runtime, Seeds, Summary } from "../api/types";
import { generators, seedSizes } from "../lib/study";

// These are the study design and aggregate conclusion stated in the brief, not reconstructed measurements.
export const criteria: Criteria = {
  rules: [],
  evaluations: generators.flatMap((generator) =>
    seedSizes.map((seed_size) => ({
      generator,
      seed_size,
      values: {},
      passed: {},
      all_passed: false,
    })),
  ),
  min_useful_seed: null,
  explanation:
    "0 of 10 generator and seed-size combinations passed every frozen rule. The per-rule scores were not supplied.",
};
export const seeds: Seeds = {
  sizes: seedSizes,
  random_seeds: [],
  nested_ok: true,
  class_counts: {},
};
export const summary: Summary[] = [];
export const fidelity: Fidelity = { overall: [], per_column: [] };
export const privacy: Privacy[] = [];
export const runtime: Runtime[] = [];
export const fixtures = {
  "/results/criteria": criteria,
  "/seeds": seeds,
  "/results/summary": summary,
  "/results/fidelity": fidelity,
  "/results/privacy": privacy,
  "/results/runtime": runtime,
};
