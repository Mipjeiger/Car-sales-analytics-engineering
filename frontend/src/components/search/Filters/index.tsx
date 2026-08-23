import { BODY_TYPES } from "@/utils/constants";
import type { SearchFilters } from "@/types/common.types";

export function Filters({
  brands,
  filters,
  onChange,
}: {
  brands: string[];
  filters: SearchFilters;
  onChange: (next: Partial<SearchFilters>) => void;
}) {
  return (
    <aside className="glass-card space-y-4 p-5">
      <h2 className="font-semibold">Filters</h2>
      <label className="block text-sm">
        Brand
        <select
          className="mt-1 w-full rounded-xl border bg-transparent px-3 py-2"
          style={{ borderColor: "var(--border)" }}
          value={filters.brand}
          onChange={(e) => onChange({ brand: e.target.value })}
        >
          <option value="all">All brands</option>
          {brands.map((b) => (
            <option key={b} value={b}>
              {b.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </label>
      <label className="block text-sm">
        Body type
        <select
          className="mt-1 w-full rounded-xl border bg-transparent px-3 py-2"
          style={{ borderColor: "var(--border)" }}
          value={filters.bodyType}
          onChange={(e) => onChange({ bodyType: e.target.value })}
        >
          <option value="all">All types</option>
          {BODY_TYPES.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      </label>
      <label className="block text-sm">
        Max price (USD)
        <input
          type="range"
          min={5000}
          max={500000}
          step={1000}
          value={filters.maxPrice}
          onChange={(e) => onChange({ maxPrice: Number(e.target.value) })}
          className="mt-2 w-full"
        />
        <span className="subtle text-xs">${filters.maxPrice.toLocaleString()}</span>
      </label>
    </aside>
  );
}
