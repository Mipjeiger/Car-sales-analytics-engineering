import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { User, UserRole } from "@/types/auth";
import {
  AUTH_TOKEN_KEY,
  AUTH_ROLE_KEY,
  AUTH_EMAIL_KEY,
} from "@/utils/constants";

interface AuthState {
  user: User | null;
  token: string | null;
}

const STORAGE_AUTH_KEY = "csi_auth";

const initialState: AuthState = readInitialState();

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginSuccess(state, action: PayloadAction<{ user: User; token: string }>) {
      state.user = action.payload.user;
      state.token = action.payload.token;

      localStorage.setItem(AUTH_TOKEN_KEY, action.payload.token);
      localStorage.setItem(AUTH_ROLE_KEY, action.payload.user.role);
      localStorage.setItem(AUTH_EMAIL_KEY, action.payload.user.email);
      localStorage.setItem(
        STORAGE_AUTH_KEY,
        JSON.stringify({ 
          user: action.payload.user, 
          token: action.payload.token 
        }),
      );
    },
    logout(state) {
      state.user = null;
      state.token = null;

      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_ROLE_KEY);
      localStorage.removeItem(AUTH_EMAIL_KEY);
      localStorage.removeItem(STORAGE_AUTH_KEY);
    },
  },
});

function readInitialState(): AuthState {
  try {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const role = localStorage.getItem(AUTH_ROLE_KEY);
    const email = localStorage.getItem(AUTH_EMAIL_KEY);

    if (token && role && email) {
      return {
        user: { 
          email, 
          role: role.toLowerCase() as UserRole,
          id: undefined,
          name: undefined
        },
        token,
      };
    }

    const storedAuth = localStorage.getItem(STORAGE_AUTH_KEY);
    if (storedAuth) {
      const parsed = JSON.parse(storedAuth) as AuthState;
      if (parsed.user && parsed.token) {
        return parsed;
      }
    }
  } catch (error) {
    console.error("Failed to parse auth state from localStorage:", error);
  }

  return { user: null, token: null };
}

export const { loginSuccess, logout } = authSlice.actions;
export default authSlice.reducer;