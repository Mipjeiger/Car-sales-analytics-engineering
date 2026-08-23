import type { ModelRegistryEntry } from "@/types/api.types";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ModelComparison({ models }: { models: ModelRegistryEntry[] }) {
  const data = models.filter((m) => m.r2 != null).map((m) => ({ name: m.name, r2: m.r2, rmse: m.rmse ?? 0 }));
  if (!data.length) return <p className="subtle text-sm">No comparable metrics yet.</p>;
  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis dataKey="name" tick={{ fontSize: 11 }} />
          <YAxis domain={[0.9, 1]} />
          <Tooltip />
          <Bar dataKey="r2" fill="#6366f1" radius={8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
