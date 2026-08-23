import { formatPercent } from "@/utils/formatters";
import { cn } from "@/utils/helpers";

export function SimilarityBadge({ score }: { score: number }) {
  const tone = score >= 0.8 ? "bg-emerald-500/15 text-secondary" : score >= 0.5 ? "bg-amber-500/15 text-accent" : "bg-slate-500/15";
  return <span className={cn("rounded-full px-2.5 py-0.5 text-xs font-semibold", tone)}>{formatPercent(score)}</span>;
}
