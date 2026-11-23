import { Download, Info } from "lucide-react";
import type { ReactNode } from "react";
export function ChartCard({
  title,
  subtitle,
  children,
  summary,
  onExport,
}: {
  title: string;
  subtitle: string;
  children: ReactNode;
  summary: string;
  onExport?: () => void;
}) {
  return (
    <div className="card chart-card">
      <div className="chart-header">
        <div>
          <h3>
            {title}{" "}
            <span tabIndex={0} title={subtitle} aria-label={subtitle}>
              <Info size={14} />
            </span>
          </h3>
          <p>{subtitle}</p>
        </div>
        <button
          className="icon-button"
          aria-label="Export chart as PNG"
          disabled={!onExport}
          onClick={onExport}
        >
          <Download size={17} />
        </button>
      </div>
      {children}
      <p className="chart-summary">{summary}</p>
    </div>
  );
}
