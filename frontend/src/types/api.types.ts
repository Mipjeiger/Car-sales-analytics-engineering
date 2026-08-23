export interface PredictRequest {
  model_type: "sales" | "quantity" | "both";
  sales_model_name?: string;
  quantity_model_name?: string;
  model_name?: string;
  features: Record<string, string | number>;
}

export interface PredictResponse {
  model_type: string;
  predicted_sales?: number | null;
  sales_confidence?: number | null;
  predicted_quantity?: number | null;
  quantity_confidence?: number | null;
}

export interface AvailableModelsResponse {
  sales_models: string[];
  quantity_models: string[];
}

export interface ChatRequest {
  message: string;
  session_id?: string;
  context?: Record<string, unknown>;
}

export interface ChatResponse {
  response: string;
  intent: string;
  entities: Record<string, unknown>;
  session_id: string;
  timestamp: string;
}

export interface SearchResult {
  brand: string;
  path: string;
  similarity: number;
  rank: number;
  model?: string;
  price?: number;
  bodyType?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  query_image: string;
}

export interface BrandsResponse {
  brands: string[];
  total: number;
}

export interface SearchStatsResponse {
  total_images: number;
  feature_dimension: number;
  brands: number;
}

export interface DamageDetectResponse {
  damage_type: string;
  severity: "None" | "Low" | "Medium" | "High";
  confidence: number;
  repair_cost_estimate: number;
  repair_days: number;
  qa_status: "Pass" | "Fail" | "Rework";
  similar_cases?: SimilarDamageCase[];
}

export interface SimilarDamageCase {
  id: string;
  damage_type: string;
  severity: string;
  similarity: number;
  image_url?: string;
}

export interface BusinessMetrics {
  business_revenue_impact: number;
  business_active_users: number;
  business_dropped_impact: number;
  business_quantity_sold: number;
  business_kpi: Record<string, number>;
  predictions_today?: number;
  model_r2?: number;
}

export interface ModelRegistryEntry {
  name: string;
  type: "sales" | "quantity" | "vision" | "llm" | "damage";
  version: string;
  stage: "Staging" | "Production" | "Archived" | "None";
  r2?: number;
  rmse?: number;
  updatedAt: string;
}
