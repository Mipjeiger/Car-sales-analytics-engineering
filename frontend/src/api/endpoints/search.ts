import { apiClient } from "@/api/client";
import type { BrandsResponse, SearchResponse, SearchStatsResponse } from "@/types/api.types";

export const searchApi = {
  similar: async (file: File, k = 5) => {
    const form = new FormData();
    form.append("file", file);
    const { data } = await apiClient.post<SearchResponse>("/search/similar", form, {
      params: { k },
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },
  brands: async () => {
    const { data } = await apiClient.get<BrandsResponse>("/search/brands");
    return data;
  },
  stats: async () => {
    const { data } = await apiClient.get<SearchStatsResponse>("/search/stats");
    return data;
  },
};
