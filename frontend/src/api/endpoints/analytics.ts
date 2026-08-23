import { apiClient } from "@/api/client";
import type { BusinessMetrics } from "@/types/api.types";

export const analyticsApi = {
  prometheusMetrics: async () => {
    const { data } = await apiClient.get<string>("/metrics", { responseType: "text" });
    return data;
  },
  businessMetrics: async () => {
    const { data } = await apiClient.get<BusinessMetrics>("/business-metrics");
    return data;
  },
  health: async () => {
    const { data } = await apiClient.get<{ status: string; message: string }>("/health");
    return data;
  },
};
