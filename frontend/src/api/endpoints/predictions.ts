import { apiClient } from "@/api/client";
import type { AvailableModelsResponse, PredictRequest, PredictResponse } from "@/types/api.types";

export const predictionsApi = {
  listModels: async () => {
    const { data } = await apiClient.get<AvailableModelsResponse>("/predict/models");
    return data;
  },
  predict: async (payload: PredictRequest) => {
    const { data } = await apiClient.post<PredictResponse>("/predict/", payload);
    return data;
  },
};
