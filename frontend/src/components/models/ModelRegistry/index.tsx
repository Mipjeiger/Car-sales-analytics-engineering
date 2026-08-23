import type { ModelRegistryEntry } from "@/types/api.types";
import { formatR2 } from "@/utils/formatters";

export function ModelRegistry({ models, onSelect }: { models: ModelRegistryEntry[]; onSelect: (m: ModelRegistryEntry) => void }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead className="subtle text-xs uppercase">
          <tr>
            <th className="pb-3">Name</th>
            <th className="pb-3">Type</th>
            <th className="pb-3">Version</th>
            <th className="pb-3">Stage</th>
            <th className="pb-3">R²</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr
              key={`${m.name}-${m.version}`}
              className="cursor-pointer border-t hover:bg-white/5"
              style={{ borderColor: "var(--border)" }}
              onClick={() => onSelect(m)}
            >
              <td className="py-3 font-medium">{m.name}</td>
              <td>{m.type}</td>
              <td>{m.version}</td>
              <td>{m.stage}</td>
              <td>{m.r2 != null ? formatR2(m.r2) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
