import { arms } from "../../lib/theme";
export function ArmLegend({
  selected,
  onToggle,
}: {
  selected?: string[];
  onToggle?: (arm: string) => void;
}) {
  return (
    <div className="legend">
      {Object.entries(arms).map(([key, arm]) => (
        <button
          key={key}
          type="button"
          className={
            selected && !selected.includes(key) ? "legend-item muted-legend" : "legend-item"
          }
          onClick={() => onToggle?.(key)}
          aria-pressed={selected ? selected.includes(key) : undefined}
        >
          <i style={{ background: arm.color }} />
          {arm.label}
        </button>
      ))}
    </div>
  );
}
