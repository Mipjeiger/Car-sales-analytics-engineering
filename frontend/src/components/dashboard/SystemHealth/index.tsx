import { AIRFLOW_URL, GRAFANA_URL, MLFLOW_URL, PROMETHEUS_URL } from "@/utils/constants";
import { useWebSocket } from "@/hooks/useWebSocket";
import type { ServiceHealth } from "@/types/common.types";
import { cn } from "@/utils/helpers";

export function SystemHealth({ apiUp }: { apiUp: boolean }) {
  const { status } = useWebSocket(true);
  const services: ServiceHealth[] = [
    { name: "FastAPI", status: apiUp ? "up" : "down", url: "http://localhost:8000/health" },
    { name: "MLflow", status: "degraded", url: MLFLOW_URL },
    { name: "Prometheus", status: "degraded", url: PROMETHEUS_URL },
    { name: "Grafana", status: "degraded", url: GRAFANA_URL },
    { name: "Airflow", status: "degraded", url: AIRFLOW_URL },
    { name: "Live events (WS)", status: status === "open" ? "up" : status === "error" ? "down" : "degraded" },
  ];

  return (
    <section className="glass-card p-5">
      <h2 className="text-lg font-semibold">System health</h2>
      <p className="subtle mt-1 text-sm">FastAPI /health is live-checked. Other services are linked from docker-compose ports.</p>
      <ul className="mt-4 grid gap-3 sm:grid-cols-2">
        {services.map((s) => (
          <li key={s.name} className="flex items-center justify-between rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)" }}>
            <div>
              <p className="text-sm font-medium">{s.name}</p>
              {s.url ? (
                <a className="subtle text-xs underline" href={s.url} target="_blank" rel="noreferrer">
                  Open
                </a>
              ) : null}
            </div>
            <span
              className={cn(
                "rounded-full px-2 py-0.5 text-xs font-medium",
                s.status === "up" && "bg-emerald-500/15 text-secondary",
                s.status === "degraded" && "bg-amber-500/15 text-accent",
                s.status === "down" && "bg-rose-500/15 text-rose-400",
              )}
            >
              {s.status}
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}
