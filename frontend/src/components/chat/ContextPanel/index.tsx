export function ContextPanel({ entities }: { entities: Record<string, unknown> }) {
  const entries = Object.entries(entities);
  return (
    <aside className="glass-card p-5">
      <h2 className="font-semibold">Context</h2>
      <p className="subtle mt-1 text-sm">Entities parsed from the latest chat turn.</p>
      {entries.length === 0 ? (
        <p className="subtle mt-4 text-sm">No entities yet.</p>
      ) : (
        <dl className="mt-4 space-y-2 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="flex justify-between gap-3 rounded-xl border px-3 py-2" style={{ borderColor: "var(--border)" }}>
              <dt className="subtle capitalize">{key.replace(/_/g, " ")}</dt>
              <dd className="font-medium">{typeof value === "object" ? JSON.stringify(value) : String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </aside>
  );
}
