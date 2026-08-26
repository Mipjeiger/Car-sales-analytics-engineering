// USE RELATIVE PATHS ONLY - No hardcoded URLs!
export const API_BASE_URL = '/api';  
export const WS_BASE_URL = '/ws';

export const AUTH_LOGIN_PATH = "/auth/login";
export const AUTH_LOGIN_URL = `${API_BASE_URL}${AUTH_LOGIN_PATH}`;

export const AUTH_TOKEN_KEY = "car_sales_access_token";
export const AUTH_ROLE_KEY = "car_sales_role";
export const AUTH_EMAIL_KEY = "car_sales_email";
export const METRICS_URL = "/metrics";

export const API_URLS = {
  login: `${API_BASE_URL}/auth/login`,
  register: `${API_BASE_URL}/auth/register`,
  logout: `${API_BASE_URL}/auth/logout`,
  user: `${API_BASE_URL}/user`,
  dashboard: `${API_BASE_URL}/dashboard`,
  analytics: `${API_BASE_URL}/analytics`,
  chat: `${API_BASE_URL}/chat`,
  damage: `${API_BASE_URL}/damage`,
  metrics: `${API_BASE_URL}/metrics`,
  models: `${API_BASE_URL}/predict/models`,
  settings: `${API_BASE_URL}/settings`,
  search: `${API_BASE_URL}/search`,
} as const;

// WebSocket URL
export const WS_URL = '/ws';
  
// External services (these can stay as is since they're not proxied)
export const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL ?? "http://localhost:5003";
export const GRAFANA_URL = import.meta.env.VITE_GRAFANA_URL ?? "http://localhost:3001";
export const AIRFLOW_URL = import.meta.env.VITE_AIRFLOW_URL ?? "http://localhost:8080";
export const PROMETHEUS_URL = import.meta.env.VITE_PROMETHEUS_URL ?? "http://localhost:9090";

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

export const getMetricsUrl = () => `${API_BASE_URL}${METRICS_URL}`;