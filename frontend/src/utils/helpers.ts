export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

export function parsePrometheus(text: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([0-9eE.+-]+)/);
    if (match) {
      out[match[1]] = Number(match[2]);
    }
  }
  return out;
}

export function sumMetricPrefix(metrics: Record<string, number>, prefix: string): number {
  return Object.entries(metrics).reduce((acc, [key, value]) => {
    if (key.startsWith(prefix) && !key.includes("_created")) {
      return acc + (Number.isFinite(value) ? value : 0);
    }
    return acc;
  }, 0);
}

export function avgMetricPrefix(metrics: Record<string, number>, prefix: string): number {
  const values = Object.entries(metrics)
    .filter(([key]) => key.startsWith(prefix) && !key.includes("_created") && !key.includes("_bucket"))
    .map(([, value]) => value)
    .filter((value) => Number.isFinite(value));
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function fileToObjectUrl(file: File): string {
  return URL.createObjectURL(file);
}

export function estimatePriceFromBrand(brand: string, similarity: number): number {
  const base: Record<string, number> = {
    Audi: 52000,
    BMW: 54000,
    Mercedes: 58000,
    Toyota: 28000,
    Honda: 26000,
    Hyundai: 24000,
    Mahindra: 22000,
    Rolls_Royce: 320000,
    Tesla: 45000,
  };
  const key = Object.keys(base).find((k) => brand.toLowerCase().includes(k.toLowerCase().replace("_", " ")) || brand.includes(k));
  const start = key ? base[key] : 30000;
  return Math.round(start * (0.85 + similarity * 0.3));
}
