import { useQuery } from "@tanstack/react-query";
import { analyticsApi } from "@/api/endpoints/analytics";
import { predictionsApi } from "@/api/endpoints/predictions";
import { avgMetricPrefix, parsePrometheus, sumMetricPrefix } from "@/utils/helpers";
import { useAppDispatch } from "@/store/hooks";
import { setMetrics } from "@/store/slices/analyticsSlice";
import { useEffect } from "react";

const FALLBACK_METRICS = {
  business_revenue_impact: 0,
  business_active_users: 0,
  business_dropped_impact: 0,
  business_quantity_sold: 0,
  business_kpi: {},
  predictions_today: 0,
  model_r2: 0.99,
};

export function useAnalytics() {
  const dispatch = useAppDispatch();

  const business = useQuery({
    queryKey: ["business-metrics"],
    queryFn: analyticsApi.businessMetrics,
    retry: 1,
  });

  const prometheus = useQuery({
    queryKey: ["prometheus-metrics"],
    queryFn: analyticsApi.prometheusMetrics,
    retry: 1,
    refetchInterval: 30_000,
  });

  const health = useQuery({
    queryKey: ["api-health"],
    queryFn: analyticsApi.health,
    retry: 1,
    refetchInterval: 20_000,
  });

  const models = useQuery({
    queryKey: ["predict-models"],
    queryFn: predictionsApi.listModels,
    retry: 1,
  });

  const parsed = prometheus.data ? parsePrometheus(prometheus.data) : {};
  const predictionsToday = sumMetricPrefix(parsed, "predictions_total");
  const modelR2 = avgMetricPrefix(parsed, "model_r2_score") || FALLBACK_METRICS.model_r2;

  const metrics = {
    ...(business.data ?? FALLBACK_METRICS),
    predictions_today: predictionsToday,
    model_r2: modelR2,
  };

  useEffect(() => {
    if (!business.data) return;
    dispatch(
      setMetrics({
        ...business.data,
        predictions_today: predictionsToday,
        model_r2: modelR2,
      }),
    );
  }, [business.data, dispatch, predictionsToday, modelR2]);

  return {
    metrics,
    health: health.data,
    apiUp: health.isSuccess,
    models: models.data,
    prometheusText: prometheus.data,
    isLoading: business.isLoading && prometheus.isLoading,
    error: business.error instanceof Error ? business.error.message : null,
  };
}
