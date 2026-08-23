export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";
export const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL ?? "http://localhost:5003";
export const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL ?? "http://localhost:3001";
export const AIRFLOW_URL = import.meta.env.VITE_AIRFLOW_URL ?? "http://localhost:8080";
export const PROMETHEUS_URL = import.meta.env.VITE_PROMETHEUS_URL ?? "http://localhost:9090";
export const AUTH_LOGIN_URL = `${API_BASE}/auth/login`;

export const SALES_FEATURE_FIELDS = [
  "day_of_week",
  "week_of_year",
  "season",
  "cost",
  "gross_sales",
  "profit",
  "rolling_mean_7",
  "rolling_std_7",
  "rolling_max_7",
  "quantity",
  "model",
  "price_band",
] as const;

export const QUANTITY_FEATURE_FIELDS = [
  "gender",
  "income_customer",
  "dealer_name",
  "dealer_region",
  "company",
  "model",
  "color",
  "body_style",
  "price",
  "discount",
  "day_of_week",
  "season",
  "week_of_year",
  "price_band",
  "sales",
] as const;

export const BODY_TYPES = ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Truck"];

export const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "grid" },
  { to: "/search", label: "Visual Search", icon: "search" },
  { to: "/analytics", label: "Analytics", icon: "chart" },
  { to: "/chat", label: "Chat Assistant", icon: "chat" },
  { to: "/damage", label: "Damage Detection", icon: "shield" },
  { to: "/models", label: "Model Registry", icon: "cube", adminOnly: true },
  { to: "/settings", label: "Settings", icon: "settings" },
] as const;
