import { KPIStats } from "@/components/dashboard/KPIStats";
import { ActivityFeed } from "@/components/dashboard/ActivityFeed";
import { SystemHealth } from "@/components/dashboard/SystemHealth";
import { useAnalytics } from "@/hooks/useAnalytics";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";
import { Link } from "react-router-dom";

export default function Dashboard() {
  const { metrics, apiUp, isLoading, error } = useAnalytics();

  return (
    <div className="space-y-6">
      <div className="rounded-3xl bg-hero-gradient p-8 text-white shadow-glass">
        <p className="text-sm uppercase tracking-[0.25em] text-indigo-200">Overview</p>
        <h1 className="mt-2 text-3xl font-semibold">Car Sales Intelligence</h1>
        <p className="mt-2 max-w-2xl text-indigo-100">
          Predictions, visual search, damage QA, and LLM assistance on top of FastAPI, MLflow, Airflow, and Prometheus.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <Link className="btn-primary" to="/search">
            Upload image for search
          </Link>
          <Link className="btn-ghost border-white/20 text-white" to="/chat">
            Start chat
          </Link>
        </div>
      </div>
      {isLoading ? <LoadingSpinner label="Loading metrics" /> : <KPIStats metrics={metrics} />}
      {error ? <p className="text-sm text-amber-400">API metrics unavailable: {error}. Showing zeros until FastAPI is up.</p> : null}
      <div className="grid gap-4 xl:grid-cols-2">
        <ActivityFeed />
        <SystemHealth apiUp={apiUp} />
      </div>
    </div>
  );
}
