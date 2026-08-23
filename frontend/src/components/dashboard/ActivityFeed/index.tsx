import { useAppSelector } from "@/store/hooks";
import { formatDateTime } from "@/utils/formatters";

export function ActivityFeed() {
  const predictions = useAppSelector((s) => s.analytics.predictions);
  const messages = useAppSelector((s) => s.chat.messages).slice(-5).reverse();

  return (
    <section className="glass-card p-5">
      <h2 className="text-lg font-semibold">Recent activity</h2>
      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <div>
          <p className="subtle mb-2 text-xs uppercase tracking-wide">Predictions</p>
          {predictions.length === 0 ? (
            <p className="subtle text-sm">No local predictions yet. Run one from Analytics.</p>
          ) : (
            <ul className="space-y-2">
              {predictions.slice(0, 5).map((p, i) => (
                <li key={i} className="rounded-xl border px-3 py-2 text-sm" style={{ borderColor: "var(--border)" }}>
                  {p.model_type}: sales {p.predicted_sales?.toFixed?.(0) ?? "—"} · qty {p.predicted_quantity?.toFixed?.(0) ?? "—"}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <p className="subtle mb-2 text-xs uppercase tracking-wide">Chat</p>
          {messages.length === 0 ? (
            <p className="subtle text-sm">No chat turns in this session.</p>
          ) : (
            <ul className="space-y-2">
              {messages.map((m) => (
                <li key={m.id} className="rounded-xl border px-3 py-2 text-sm" style={{ borderColor: "var(--border)" }}>
                  <span className="font-medium">{m.role}</span> · {m.content.slice(0, 80)}
                  <span className="subtle ml-2 text-xs">{formatDateTime(m.timestamp)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
