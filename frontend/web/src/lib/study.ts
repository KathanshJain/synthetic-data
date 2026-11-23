import { arms, colors } from "./theme";
export const seedSizes = [100, 250, 500, 1000, 2000];
export const generators = ["CTGAN", "GaussianCopula"];
export const generatorLabel = (name: string) => name.replace(/GaussianCopula/i, "Gaussian Copula");
export const generatorColor = (name: string) =>
  arms[`TSTR-${name}`]?.color ?? (/gaussian/i.test(name) ? colors.teal : colors.indigo);
export const dcrCaveat = "DCR is a warning indicator, not a privacy guarantee.";
export const utilityThreshold = 0.9;
export const dcrThreshold = 0.8;

export const frozenRules: { id: string; label: string; threshold?: number; direction?: string }[] =
  [
    { id: "utility_ratio", label: "Utility ratio", threshold: utilityThreshold, direction: "≥" },
    { id: "ks_mean", label: "KS" },
    { id: "tvd_mean", label: "TVD" },
    { id: "dcr_ratio", label: "DCR ratio", threshold: dcrThreshold, direction: "≥" },
    { id: "exact_rate_synth", label: "Exact-copy rate" },
  ];
