import type { DamageDetectResponse } from "@/types/api.types";
import { formatIDR, formatPercent } from "@/utils/formatters";
import { cn } from "@/utils/helpers";

export function DetectionResults({ result }: { result: DamageDetectResponse | null }) {
  if (!result) {
    return <div className="glass-card p-6 subtle">Run detection to see class, severity, cost, and QA status.</div>;
  }
  return (
    <div className="glass-card grid gap-4 p-6 sm:grid-cols-2">
      <Stat label="Damage type" value={result.damage_type.replace(/_/g, " ")} />
      <Stat label="Severity" value={result.severity} tone={result.severity} />
      <Stat label="Confidence" value={formatPercent(result.confidence)} />
      <Stat label="Repair estimate" value={formatIDR(result.repair_cost_estimate)} />
      <Stat label="Repair days" value={String(result.repair_days)} />
      <Stat label="QA status" value={result.qa_status} />
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <p className="subtle text-xs uppercase tracking-wide">{label}</p>
      <p
        className={cn(
          "mt-1 text-lg font-semibold capitalize",
          tone === "High" && "text-rose-400",
          tone === "Medium" && "text-accent",
          tone === "Low" && "text-secondary",
        )}
      >
        {value}
      </p>
    </div>
  );
}
