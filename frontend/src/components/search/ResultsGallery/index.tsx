import { CarCard } from "@/components/search/CarCard";
import type { SearchResult } from "@/types/api.types";

export function ResultsGallery({
  results,
  selected,
  onCompare,
}: {
  results: SearchResult[];
  selected: SearchResult[];
  onCompare: (car: SearchResult) => void;
}) {
  if (!results.length) {
    return <p className="subtle py-12 text-center">Upload a car photo to see similar inventory.</p>;
  }
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
      {results.map((car) => (
        <CarCard
          key={`${car.rank}-${car.path}`}
          car={car}
          selected={selected.some((s) => s.rank === car.rank && s.path === car.path)}
          onCompare={onCompare}
        />
      ))}
    </div>
  );
}
