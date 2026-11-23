import type { ReactNode } from "react";
import { animate, motion, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { ArrowUpRight } from "lucide-react";
import { formatNumber } from "../../lib/transforms";
import { transition } from "../ui/Primitives";
export function Count({
  value,
  digits = 0,
  suffix = "",
}: {
  value?: number;
  digits?: number;
  suffix?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true });
  const reduced = useReducedMotion();
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    if (!seen || value === undefined || reduced) return;
    const control = animate(0, value, { ...transition, onUpdate: setAnimated });
    return () => control.stop();
  }, [seen, value, reduced]);
  return (
    <span className="count-number" ref={ref}>
      {formatNumber(value === undefined || reduced ? value : animated, digits)}
      {value !== undefined ? suffix : ""}
    </span>
  );
}
export function StatCard({
  label,
  value,
  detail,
  icon,
  preview,
  onOpen,
  digits,
  suffix,
  wide,
}: {
  label: string;
  value?: number;
  detail: string;
  icon: ReactNode;
  preview?: ReactNode;
  onOpen: () => void;
  digits?: number;
  suffix?: string;
  wide?: boolean;
}) {
  const reduced = useReducedMotion();
  return (
    <motion.button
      type="button"
      aria-haspopup="dialog"
      className={`stat-card card stat-interactive${wide ? " stat-wide" : ""}`}
      onClick={onOpen}
      onPointerMove={(e) => {
        if (reduced || e.pointerType !== "mouse") return;
        const rect = e.currentTarget.getBoundingClientRect();
        e.currentTarget.style.setProperty("--spot-x", `${e.clientX - rect.left}px`);
        e.currentTarget.style.setProperty("--spot-y", `${e.clientY - rect.top}px`);
      }}
    >
      <span className="stat-card-top">
        <span>{label}</span>
        <span className="stat-icon">{icon}</span>
      </span>
      <span className="stat-body">
        <span className="stat-number">
          <Count value={value} digits={digits} suffix={suffix} />
        </span>
        {preview}
      </span>
      <span className="stat-detail">{detail}</span>
      <ArrowUpRight className="stat-expand" size={15} />
    </motion.button>
  );
}
