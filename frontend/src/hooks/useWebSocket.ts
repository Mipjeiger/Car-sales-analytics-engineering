import { useEffect, useRef, useState } from "react";
import { WS_URL } from "@/utils/constants";

export function useWebSocket(enabled = true) {
  const [status, setStatus] = useState<"idle" | "open" | "closed" | "error">("idle");
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) return;
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen = () => setStatus("open");
      ws.onclose = () => setStatus("closed");
      ws.onerror = () => setStatus("error");
      ws.onmessage = (event) => setLastMessage(String(event.data));
      return () => ws.close();
    } catch {
      setStatus("error");
    }
  }, [enabled]);

  return { status, lastMessage };
}
