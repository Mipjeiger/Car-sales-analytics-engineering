import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import type { ChatMessage } from "@/types/common.types";

interface ChatState {
  messages: ChatMessage[];
  sessionId: string;
  entities: Record<string, unknown>;
  isTyping: boolean;
}

const initialState: ChatState = {
  messages: [],
  sessionId: crypto.randomUUID(),
  entities: {},
  isTyping: false,
};

const chatSlice = createSlice({
  name: "chat",
  initialState,
  reducers: {
    addMessage(state, action: PayloadAction<ChatMessage>) {
      state.messages.push(action.payload);
    },
    setSessionId(state, action: PayloadAction<string>) {
      state.sessionId = action.payload;
    },
    setEntities(state, action: PayloadAction<Record<string, unknown>>) {
      state.entities = action.payload;
    },
    setTyping(state, action: PayloadAction<boolean>) {
      state.isTyping = action.payload;
    },
    resetChat(state) {
      state.messages = [];
      state.entities = {};
      state.sessionId = crypto.randomUUID();
      state.isTyping = false;
    },
  },
});

export const { addMessage, setSessionId, setEntities, setTyping, resetChat } = chatSlice.actions;
export default chatSlice.reducer;
