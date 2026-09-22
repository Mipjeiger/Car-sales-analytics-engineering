export const API_BASE_URL = '';

// Auth endpoints - matches /auth router
export const AUTH_LOGIN_PATH = "/auth/login";
export const AUTH_LOGIN_URL = "/auth/login";
export const AUTH_TOKEN_KEY = "car_sales_access_token";
export const AUTH_ROLE_KEY = "car_sales_role";
export const AUTH_EMAIL_KEY = "car_sales_email";

// API URLs - match FastAPI routes
export const API_URLS = {
  login: '/auth/login',
  register: '/auth/register',
  logout: '/auth/logout',
  user: '/api/user',  // If this doesn't exist, change it
  dashboard: '/api/dashboard',
  analytics: '/api/analytics',
  chat: '/chat', 
  damage: '/api/damage',
  metrics: '/metrics', 
  models: '/predict/models',  
  settings: '/api/settings',
  search: '/search',  
  health: '/health',  
  business_metrics: '/business-metrics', 
} as const;

// WebSocket URL - matches /ws router
export const WS_URL = '/ws';

// External services (these are not proxied)
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

// Helper functions
export const getMetricsUrl = () => '/metrics';
export const getBusinessMetricsUrl = () => '/business-metrics';
export const getModelsUrl = () => '/predict/models';
export const getHealthUrl = () => '/health';