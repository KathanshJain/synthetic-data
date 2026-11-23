import { useState } from "react";
import type { EChartsType } from "echarts";
import type { ReactNode } from "react";
import { colors } from "../../lib/theme";
import { ChartCard } from "./ChartCard";
import { Plot, type PlotOption } from "./Plot";
export function ResearchChart({
  title,
  subtitle,
  summary,
  option,
  height,
  children,
}: {
  title: string;
  subtitle: string;
  summary: string;
  option?: PlotOption;
  height?: number;
  children?: ReactNode;
}) {
  const [chart, setChart] = useState<EChartsType>();
  return (
    <ChartCard
      title={title}
      subtitle={subtitle}
      summary={summary}
      onExport={
        option && chart && !chart.isDisposed()
          ? () => {
              const a = document.createElement("a");
              a.href = chart.getDataURL({
                type: "png",
                pixelRatio: 2,
                backgroundColor: colors.surface,
              });
              a.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.png`;
              a.click();
            }
          : undefined
      }
    >
      {children}
      {option && <Plot option={option} label={summary} height={height} onReady={setChart} />}
    </ChartCard>
  );
}
