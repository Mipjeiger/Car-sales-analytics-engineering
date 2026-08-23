import { configureStore } from "@reduxjs/toolkit";
import auth from "./slices/authSlice";
import search from "./slices/searchSlice";
import chat from "./slices/chatSlice";
import analytics from "./slices/analyticsSlice";

export const store = configureStore({
  reducer: { auth, search, chat, analytics },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
