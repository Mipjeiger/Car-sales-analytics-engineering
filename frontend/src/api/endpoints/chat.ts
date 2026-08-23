import { apiClient } from "@/api/client";
import type { ChatRequest, ChatResponse } from "@/types/api.types";

export const chatApi = {
  send: async (payload: ChatRequest) => {
    const { data } = await apiClient.post<ChatResponse>("/chat/", payload);
    return data;
  },
  reset: async (sessionId: string) => {
    const { data } = await apiClient.post("/chat/reset", null, { params: { session_id: sessionId } });
    return data as { status: string; message: string };
  },
  sessions: async () => {
    const { data } = await apiClient.get("/chat/session_chat");
    return data as { status: string; session_count: number; sessions: unknown };
  },
  intents: async () => {
    const { data } = await apiClient.get<{ intents: string[]; description: string }>("/chat/intents");
    return data;
  },
};
