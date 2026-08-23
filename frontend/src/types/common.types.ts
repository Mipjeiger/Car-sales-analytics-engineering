export type UserRole = "Admin" | "Analyst" | "Viewer";

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}

export type ThemeMode = "dark" | "light";

export interface SearchFilters {
  brand: string;
  minPrice: number;
  maxPrice: number;
  bodyType: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: string;
  timestamp: string;
}

export interface ServiceHealth {
  name: string;
  status: "up" | "down" | "degraded";
  url?: string;
}
