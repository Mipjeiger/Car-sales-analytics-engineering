import { NAV_ITEMS } from "@/utils/constants";
import type { UserRole } from "@/types/common.types";

export function ModelSelector({
  value,
  options,
  onChange,
  label = "Model",
}: {
  value: string;
  options: string[];
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <label className="block text-sm">
      {label}
      <select
        className="mt-1 w-full rounded-xl border bg-transparent px-3 py-2"
        style={{ borderColor: "var(--border)" }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        <option value="">Default</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

export function canAccess(path: string, role: UserRole) {
  const item = NAV_ITEMS.find((n) => n.to === path);
  if (!item) return true;
  return !("adminOnly" in item && item.adminOnly) || role === "Admin";
}
