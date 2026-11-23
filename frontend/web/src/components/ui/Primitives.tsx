import { Info, RefreshCw, ArrowUpRight } from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
export const transition = {
  duration: 0.7,
  ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
};
export function Reveal({ children, className = "" }: { children: ReactNode; className?: string }) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      data-reveal
      initial={{ opacity: 0, y: reduced ? 0 : 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={transition}
    >
      {children}
    </motion.div>
  );
}
export function SectionHeading({
  eyebrow,
  title,
  description,
}: {
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">
          <span className="eyebrow-dot" />
          {eyebrow}
        </div>
        <h2>{title}</h2>
      </div>
      <p>{description}</p>
    </div>
  );
}
export function Empty({
  title = "Waiting for the data",
  message = "The research data for this view hasn’t been supplied yet.",
  retry,
}: {
  title?: string;
  message?: string;
  retry?: () => void;
}) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Info size={18} />
      </div>
      <strong>{title}</strong>
      <p>{message}</p>
      {retry && (
        <button className="text-button" onClick={retry}>
          <RefreshCw size={13} /> Try again
        </button>
      )}
    </div>
  );
}
export function ArrowLink({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="arrow-link" href={href}>
      {children}
      <ArrowUpRight size={16} />
    </a>
  );
}
