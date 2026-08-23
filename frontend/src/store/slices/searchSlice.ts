import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { SearchResult } from "@/types/api.types";
import type { SearchFilters } from "@/types/common.types";

interface SearchState {
  results: SearchResult[];
  selected: SearchResult[];
  queryImage: string | null;
  isLoading: boolean;
  filters: SearchFilters;
}

const initialState: SearchState = {
  results: [],
  selected: [],
  queryImage: null,
  isLoading: false,
  filters: { brand: "all", minPrice: 0, maxPrice: 500000, bodyType: "all" },
};

const searchSlice = createSlice({
  name: "search",
  initialState,
  reducers: {
    setLoading(state, action: PayloadAction<boolean>) {
      state.isLoading = action.payload;
    },
    setResults(state, action: PayloadAction<{ results: SearchResult[]; queryImage: string | null }>) {
      state.results = action.payload.results;
      state.queryImage = action.payload.queryImage;
    },
    setFilters(state, action: PayloadAction<Partial<SearchFilters>>) {
      state.filters = { ...state.filters, ...action.payload };
    },
    toggleCompare(state, action: PayloadAction<SearchResult>) {
      const exists = state.selected.find((c) => c.path === action.payload.path && c.rank === action.payload.rank);
      if (exists) {
        state.selected = state.selected.filter((c) => !(c.path === action.payload.path && c.rank === action.payload.rank));
      } else if (state.selected.length < 3) {
        state.selected.push(action.payload);
      }
    },
    clearCompare(state) {
      state.selected = [];
    },
  },
});

export const { setLoading, setResults, setFilters, toggleCompare, clearCompare } = searchSlice.actions;
export default searchSlice.reducer;
