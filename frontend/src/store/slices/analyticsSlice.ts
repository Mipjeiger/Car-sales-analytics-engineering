import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { BusinessMetrics, PredictResponse } from "@/types/api.types";

interface AnalyticsState {
  metrics: BusinessMetrics | null;
  predictions: PredictResponse[];
}

const initialState: AnalyticsState = {
  metrics: null,
  predictions: [],
};

const analyticsSlice = createSlice({
  name: "analytics",
  initialState,
  reducers: {
    setMetrics(state, action: PayloadAction<BusinessMetrics>) {
      state.metrics = action.payload;
    },
    addPrediction(state, action: PayloadAction<PredictResponse>) {
      state.predictions.unshift(action.payload);
      state.predictions = state.predictions.slice(0, 20);
    },
  },
});

export const { setMetrics, addPrediction } = analyticsSlice.actions;
export default analyticsSlice.reducer;
