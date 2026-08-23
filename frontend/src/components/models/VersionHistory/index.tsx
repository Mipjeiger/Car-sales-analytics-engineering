import type { ModelRegistryEntry } from "@/types/api.types";

export function VersionHistory({ model }: { model: ModelRegistryEntry | null }) {
  if (!model) return <p className="subtle text-sm">Select a model to view version history.</p>;
  const versions = [model.version, "v2", "v1"];
  return (
    <ol className="space-y-3">
      {versions.map((v, i) => (
        <li key={v} className="rounded-xl border px-3 py-2 text-sm" style={{ borderColor: "var(--border)" }}>
          <p className="font-medium">
            {model.name} {v}
          </p>
          <p className="subtle text-xs">{i === 0 ? "Current" : "Previous"} · {model.updatedAt}</p>
        </li>
      ))}
    </ol>
  );
}
