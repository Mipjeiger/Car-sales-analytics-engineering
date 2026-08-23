/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_WS_URL: string;
  readonly VITE_MLFLOW_URL: string;
  readonly VITE_GRAFANA_URL: string;
  readonly VITE_AIRFLOW_URL: string;
  readonly VITE_PROMETHEUS_URL: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
