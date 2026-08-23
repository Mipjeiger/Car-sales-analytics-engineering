import type { SimilarDamageCase } from "@/types/api.types";
import { SimilarityBadge } from "@/components/search/SimilarityBadge";

export function SimilarCases({ cases }: { cases: SimilarDamageCase[] }) {
  if (!cases.length) {
    return <p className="subtle text-sm">No similar FAISS cases yet.</p>;
  }
  return (
    <ul className="grid gap-3 sm:grid-cols-2">
      {cases.map((c) => (
        <li key={c.id} className="glass-card p-4">
          <div className="flex items-center justify-between">
            <p className="font-medium capitalize">{c.damage_type.replace(/_/g, " ")}</p>
            <SimilarityBadge score={c.similarity} />
          </div>
          <p className="subtle mt-1 text-sm">Severity: {c.severity}</p>
        </li>
      ))}
    </ul>
  );
}
