import type { MetricCardProps } from "@/types/component.types";
import { cn } from "@/utils/helpers";

export function MetricCard({ label, value, hint, trend, icon }: MetricCardProps) {
  return (
    <article className="glass-card p-5 transition hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-3">
        <p className="subtle text-sm font-medium">{label}</p>
        {icon ? <span className="text-primary">{icon}</span> : null}
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight">{value}</p>
      <div className="mt-2 flex items-center gap-2 text-xs">
        {typeof trend === "number" ? (
          <span className={cn(trend >= 0 ? "text-secondary" : "text-rose-400")}>
            {trend >= 0 ? "+" : ""}
            {trend.toFixed(1)}%
          </span>
        ) : null}
        {hint ? <span className="subtle">{hint}</span> : null}
      </div>
    </article>
  );
}
