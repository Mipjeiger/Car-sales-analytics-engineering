import { useMutation, useQuery } from "@tanstack/react-query";
import { predictionsApi } from "@/api/endpoints/predictions";
import { useAppDispatch } from "@/store/hooks";
import { addPrediction } from "@/store/slices/analyticsSlice";
import type { PredictRequest } from "@/types/api.types";

export function useAvailableModels() {
  return useQuery({
    queryKey: ["predict-models"],
    queryFn: predictionsApi.listModels,
    staleTime: 60_000,
  });
}

export function usePredict() {
  const dispatch = useAppDispatch();
  return useMutation({
    mutationFn: (payload: PredictRequest) => predictionsApi.predict(payload),
    onSuccess: (data) => dispatch(addPrediction(data)),
  });
}
