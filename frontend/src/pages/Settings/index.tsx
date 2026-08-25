import { API_BASE_URL, GRAFANA_URL, MLFLOW_URL } from "@/utils/constants";
import { useTheme } from "@/hooks/useTheme";
import { useAppSelector } from "@/store/hooks";

export default function SettingsPage() {
  const { theme, toggle } = useTheme();
  const user = useAppSelector((s) => s.auth.user);

  return (
    <div className="space-y-6">
      <h1 className="page-title">Settings</h1>
      <section className="glass-card space-y-3 p-5">
        <h2 className="font-semibold">Appearance</h2>
        <button className="btn-primary" onClick={toggle}>
          Switch to {theme === "dark" ? "light" : "dark"}
        </button>
      </section>
      <section className="glass-card space-y-2 p-5 text-sm">
        <h2 className="font-semibold">Session</h2>
        <p>{user?.email} · {user?.role}</p>
        <p className="subtle">API base: {API_BASE_URL}</p>
        <p>
          <a className="text-primary underline" href={MLFLOW_URL} target="_blank" rel="noreferrer">
            MLflow
          </a>
          {" · "}
          <a className="text-primary underline" href={GRAFANA_URL} target="_blank" rel="noreferrer">
            Grafana
          </a>
        </p>
      </section>
    </div>
  );
}
