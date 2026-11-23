import { useInView, useReducedMotion } from "framer-motion";
import { lazy, Suspense, useRef } from "react";
import type { EChartsOption, EChartsType } from "echarts";
import { colors } from "../../lib/theme";
export type PlotOption = EChartsOption;
const Renderer = lazy(() => import("./PlotRenderer"));
export function Plot({
  option,
  label,
  height = 300,
  onReady,
}: {
  option: PlotOption;
  label: string;
  height?: number;
  onReady?: (chart: EChartsType) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "80px" });
  const reduced = useReducedMotion();
  return (
    <div className="plot" ref={ref} role="img" aria-label={label} style={{ height }}>
      {(inView || reduced) && (
        <Suspense fallback={<div className="skeleton plot-skeleton" />}>
          <Renderer
            onChartReady={onReady}
            replaceMerge={["series", "grid", "xAxis", "yAxis", "radar", "visualMap"]}
            style={{ height: "100%", width: "100%" }}
            option={{
              textStyle: { fontFamily: "Inter, sans-serif", color: colors.muted },
              animation: !reduced,
              animationDuration: 700,
              animationDurationUpdate: 550,
              tooltip: { confine: true, renderMode: "richText" },
              ...option,
            }}
          />
        </Suspense>
      )}
    </div>
  );
}
