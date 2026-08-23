import { ChatInterface } from "@/components/chat/ChatInterface";
import { ContextPanel } from "@/components/chat/ContextPanel";
import { useAppSelector } from "@/store/hooks";

export default function ChatPage() {
  const entities = useAppSelector((s) => s.chat.entities);
  const recs = Object.entries(entities).filter(([k]) => k.toLowerCase().includes("car") || k.toLowerCase().includes("model"));

  return (
    <div className="space-y-6">
      <h1 className="page-title">Chat assistant</h1>
      <div className="grid gap-6 lg:grid-cols-[1fr_280px]">
        <ChatInterface />
        <div className="space-y-4">
          <ContextPanel entities={entities} />
          <aside className="glass-card p-5">
            <h2 className="font-semibold">Recommendations</h2>
            {recs.length === 0 ? (
              <p className="subtle mt-2 text-sm">Car suggestions from the chatbot will appear here.</p>
            ) : (
              <ul className="mt-2 space-y-2 text-sm">
                {recs.map(([k, v]) => (
                  <li key={k}>
                    {k}: {String(v)}
                  </li>
                ))}
              </ul>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
