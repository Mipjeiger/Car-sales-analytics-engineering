import type { CarCardProps } from "@/types/component.types";
import { SimilarityBadge } from "@/components/search/SimilarityBadge";
import { formatUSD } from "@/utils/formatters";
import { cn } from "@/utils/helpers";

export function CarCard({ car, selected, onSelect, onCompare }: CarCardProps) {
  const src = car.path.startsWith("http") ? car.path : undefined;
  return (
    <article
      className={cn("glass-card overflow-hidden transition hover:-translate-y-0.5", selected && "ring-2 ring-primary")}
      onClick={() => onSelect?.(car)}
    >
      <div className="flex h-36 items-center justify-center bg-gradient-to-br from-indigo-500/30 to-slate-900 text-4xl">
        {src ? <img src={src} alt={car.brand} className="h-full w-full object-cover" /> : <span aria-hidden>🚗</span>}
      </div>
      <div className="space-y-2 p-4">
        <div className="flex items-start justify-between gap-2">
          <div>
            <p className="font-semibold">{car.brand.replace(/_/g, " ")}</p>
            <p className="subtle text-sm">{car.model ?? `Rank #${car.rank}`}</p>
          </div>
          <SimilarityBadge score={car.similarity} />
        </div>
        <p className="text-sm font-medium">{car.price ? formatUSD(car.price) : "Price on request"}</p>
        <button
          className="btn-ghost w-full"
          onClick={(e) => {
            e.stopPropagation();
            onCompare?.(car);
          }}
        >
          {selected ? "Remove" : "Compare"}
        </button>
      </div>
    </article>
  );
}
