import { useState, type FormEvent } from "react";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { QuickActions } from "@/components/chat/QuickActions";
import { useChat } from "@/hooks/useChat";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";

export function ChatInterface() {
  const { messages, send, isSending, isTyping, error, reset, intents } = useChat();
  const [draft, setDraft] = useState("");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!draft.trim()) return;
    const text = draft;
    setDraft("");
    try {
      await send(text);
    } catch {
      setDraft(text);
    }
  }

  return (
    <div className="glass-card flex h-[70vh] flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="text-xs subtle">Intents: {intents.slice(0, 6).join(", ") || "loading…"}</p>
        <button className="btn-ghost text-xs" onClick={() => reset()}>
          New session
        </button>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto pr-1">
        {messages.length === 0 ? (
          <p className="subtle py-10 text-center text-sm">Ask about inventory, pricing, or recommendations.</p>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} />)
        )}
        {isTyping ? <LoadingSpinner label="Assistant is typing…" /> : null}
      </div>
      {error ? <p className="mt-2 text-sm text-rose-400">{error}</p> : null}
      <QuickActions onPick={(m) => send(m)} disabled={isSending} />
      <form className="mt-3 flex gap-2" onSubmit={onSubmit}>
        <input
          className="flex-1 rounded-xl border bg-transparent px-3 py-2 text-sm"
          style={{ borderColor: "var(--border)" }}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a message"
          aria-label="Chat message"
        />
        <button className="btn-primary" disabled={isSending}>
          Send
        </button>
      </form>
    </div>
  );
}
