import { MetricCard } from "@/components/common/MetricCard";
import { formatCompact, formatIDR, formatR2 } from "@/utils/formatters";
import type { BusinessMetrics } from "@/types/api.types";

export function KPIStats({ metrics }: { metrics: BusinessMetrics }) {
  return (
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Overview metrics">
      <MetricCard label="Predictions today" value={formatCompact(metrics.predictions_today ?? 0)} hint="POST /predict" trend={4.2} />
      <MetricCard label="Model accuracy (R²)" value={formatR2(metrics.model_r2 ?? 0)} hint="Prometheus model_r2_score" />
      <MetricCard label="Active users" value={formatCompact(metrics.business_active_users)} hint="Unique customers" trend={1.8} />
      <MetricCard label="Revenue impact" value={formatIDR(metrics.business_revenue_impact)} hint="From sales dataset" trend={6.4} />
    </section>
  );
}
