import { motion, useReducedMotion } from "framer-motion";
import { formatNumber } from "../../lib/transforms";
import { transition } from "../ui/Primitives";
export function GaugeCard({
  title,
  value,
  color,
  threshold,
  passed,
  detail,
  percent = false,
}: {
  title: string;
  value?: number;
  color: string;
  threshold?: number;
  passed?: boolean;
  detail?: string;
  percent?: boolean;
}) {
  const reduced = useReducedMotion();
  const scale = Math.max(1, value ?? 0, threshold ?? 0);
  const width = `${(Math.max(0, value ?? 0) / scale) * 100}%`;
  const shown = (n?: number) =>
    n === undefined ? "—" : percent ? `${formatNumber(n * 100, 1)}%` : formatNumber(n, 3);
  return (
    <article className="card gauge-card" style={{ "--series": color } as React.CSSProperties}>
      <div className="gauge-label">
        <h3>{title}</h3>
        {passed !== undefined && (
          <span className={passed ? "score-pass" : "score-fail"}>{passed ? "Pass" : "Fail"}</span>
        )}
      </div>
      <strong>{shown(value)}</strong>
      <div className="gauge-track" aria-hidden="true">
        {value !== undefined &&
          (reduced ? (
            <i style={{ width }} />
          ) : (
            <motion.i
              initial={{ width: 0 }}
              whileInView={{ width }}
              viewport={{ once: true }}
              transition={transition}
            />
          ))}
        {threshold !== undefined && (
          <b style={{ left: `${Math.min(99, (threshold / scale) * 100)}%` }} />
        )}
      </div>
      <p>
        {detail}
        {threshold !== undefined && ` · Threshold ${shown(threshold)}`}
      </p>
    </article>
  );
}
