import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";

const R2 = [
  { name: "XGBoost sales", r2: 0.991 },
  { name: "CatBoost sales", r2: 0.986 },
  { name: "XGBoost qty", r2: 0.998 },
  { name: "RF qty", r2: 0.972 },
];

const SCATTER = [
  { price: 18, qty: 40 },
  { price: 24, qty: 32 },
  { price: 31, qty: 22 },
  { price: 42, qty: 14 },
  { price: 55, qty: 9 },
  { price: 70, qty: 5 },
];

export function ModelPerformance() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="glass-card h-80 p-5">
        <h2 className="font-semibold">R² by model</h2>
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={R2}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="name" stroke="var(--muted)" tick={{ fontSize: 11 }} />
            <YAxis domain={[0.9, 1]} stroke="var(--muted)" />
            <Tooltip />
            <Bar dataKey="r2" fill="#34d399" radius={8} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="glass-card h-80 p-5">
        <h2 className="font-semibold">Price vs quantity</h2>
        <ResponsiveContainer width="100%" height="85%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="price" name="price" stroke="var(--muted)" />
            <YAxis dataKey="qty" name="qty" stroke="var(--muted)" />
            <Tooltip />
            <Scatter data={SCATTER} fill="#f59e0b" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
