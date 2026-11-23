export const colors = {
  background: "#FAFAF7",
  surface: "#FFFFFF",
  border: "#E7E5E0",
  ink: "#1B2430",
  muted: "#5B6572",
  indigo: "#4F46E5",
  teal: "#0EA5A4",
  amber: "#F59E0B",
  rose: "#F43F5E",
  sky: "#38BDF8",
  slate: "#64748B",
  emerald: "#10B981",
  softIndigo: "#EFEEFF",
  softTeal: "#E4F7F4",
  softAmber: "#FFF6DF",
  softRose: "#FFF0F2",
};
export const arms: Record<string, { label: string; color: string }> = {
  "TRTR-full": { label: "TRTR · full", color: colors.slate },
  "TRTR-seed": { label: "TRTR · seed", color: colors.sky },
  "SMOTE-NC": { label: "SMOTE-NC", color: colors.amber },
  "TSTR-CTGAN": { label: "TSTR · CTGAN", color: colors.indigo },
  "TSTR-GaussianCopula": { label: "TSTR · Gaussian Copula", color: colors.teal },
};
export function installTheme() {
  Object.entries(colors).forEach(([key, value]) =>
    document.documentElement.style.setProperty(
      `--${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`,
      value,
    ),
  );
}
