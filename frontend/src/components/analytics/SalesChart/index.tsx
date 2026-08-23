import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const SAMPLE = [
  { period: "Mon", revenue: 120, quantity: 18 },
  { period: "Tue", revenue: 160, quantity: 22 },
  { period: "Wed", revenue: 140, quantity: 19 },
  { period: "Thu", revenue: 210, quantity: 28 },
  { period: "Fri", revenue: 260, quantity: 34 },
  { period: "Sat", revenue: 310, quantity: 41 },
  { period: "Sun", revenue: 190, quantity: 24 },
];

export function SalesChart({ data = SAMPLE }: { data?: typeof SAMPLE }) {
  return (
    <div className="glass-card h-80 p-5">
      <h2 className="font-semibold">Revenue trend</h2>
      <p className="subtle mb-4 text-sm">Daily sales index (sample until warehouse queries are exposed)</p>
      <ResponsiveContainer width="100%" height="80%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="rev" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.7} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="period" stroke="var(--muted)" />
          <YAxis stroke="var(--muted)" />
          <Tooltip />
          <Area type="monotone" dataKey="revenue" stroke="#6366f1" fill="url(#rev)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
