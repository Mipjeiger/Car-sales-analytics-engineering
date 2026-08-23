import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { User } from "@/types/common.types";

interface AuthState {
  user: User | null;
  token: string | null;
}

function readAuth(): AuthState | null {
  try {
    const stored = localStorage.getItem("csi_auth");
    return stored ? (JSON.parse(stored) as AuthState) : null;
  } catch {
    return null;
  }
}
const parsed = readAuth();

const initialState: AuthState = parsed ?? { user: null, token: null };

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginSuccess(state, action: PayloadAction<AuthState>) {
      state.user = action.payload.user;
      state.token = action.payload.token;
      localStorage.setItem("csi_token", action.payload.token ?? "");
      localStorage.setItem("csi_auth", JSON.stringify(action.payload));
    },
    logout(state) {
      state.user = null;
      state.token = null;
      localStorage.removeItem("csi_token");
      localStorage.removeItem("csi_auth");
    },
  },
});

export const { loginSuccess, logout } = authSlice.actions;
export default authSlice.reducer;
