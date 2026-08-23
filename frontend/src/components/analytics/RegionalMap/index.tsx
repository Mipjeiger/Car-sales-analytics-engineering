import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const REGIONS = [
  { region: "East", sales: 420 },
  { region: "West", sales: 380 },
  { region: "Central", sales: 290 },
  { region: "South", sales: 340 },
  { region: "North", sales: 210 },
];

export function RegionalMap() {
  return (
    <div className="glass-card h-80 p-5">
      <h2 className="font-semibold">Regional sales</h2>
      <p className="subtle mb-2 text-sm">Dealer region distribution (placeholder until /business-metrics includes geo).</p>
      <ResponsiveContainer width="100%" height="80%">
        <BarChart data={REGIONS} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis type="number" stroke="var(--muted)" />
          <YAxis type="category" dataKey="region" stroke="var(--muted)" width={80} />
          <Tooltip />
          <Bar dataKey="sales" fill="#6366f1" radius={8} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
