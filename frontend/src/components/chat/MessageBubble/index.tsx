import { cn } from "@/utils/helpers";
import type { ChatMessage } from "@/types/common.types";
import { formatDateTime } from "@/utils/formatters";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const mine = message.role === "user";
  return (
    <div className={cn("flex", mine ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-sm",
          mine ? "bg-primary text-white" : "glass-card",
        )}
      >
        <p>{message.content}</p>
        <p className={cn("mt-1 text-[11px]", mine ? "text-indigo-100" : "subtle")}>
          {message.intent ? `${message.intent} · ` : ""}
          {formatDateTime(message.timestamp)}
        </p>
      </div>
    </div>
  );
}
