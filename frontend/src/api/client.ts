import axios, { AxiosError } from "axios";
import { API_BASE_URL, AUTH_TOKEN_KEY } from "@/utils/constants";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError<{ detail?: string }>) => {
    // Auto-clear storage if token is invalid or expired
    if (error.response?.status === 401) {
      localStorage.removeItem(AUTH_TOKEN_KEY);
    }

    const message =
      error.response?.data?.detail ||
      error.message ||
      "Request failed. Check that the FastAPI service is running on port API.";
    return Promise.reject(new Error(message));
  },
);

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}