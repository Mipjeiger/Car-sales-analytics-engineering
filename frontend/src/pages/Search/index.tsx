import { useState } from "react";
import { ImageUploader } from "@/components/search/ImageUploader";
import { ResultsGallery } from "@/components/search/ResultsGallery";
import { Filters } from "@/components/search/Filters";
import { useSearch } from "@/hooks/useSearch";
import { fileToObjectUrl } from "@/utils/helpers";
import { formatUSD } from "@/utils/formatters";
import { LoadingSpinner } from "@/components/common/LoadingSpinner";

export default function SearchPage() {
  const { results, selected, filters, brands, stats, isLoading, error, searchSimilar, setFilters, toggleCompare, clearCompare } =
    useSearch();
  const [preview, setPreview] = useState<string | null>(null);
  const [k, setK] = useState(8);

  const filtered = results.filter((r) => {
    if (filters.brand !== "all" && r.brand !== filters.brand) return false;
    if (r.price && r.price > filters.maxPrice) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      <header>
        <h1 className="page-title">Visual search</h1>
        <p className="subtle mt-1">
          POST /search/similar · {stats ? `${stats.total_images} indexed images, ${stats.brands} brands` : "stats loading"}
        </p>
      </header>
      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <div className="space-y-4">
          <Filters brands={brands} filters={filters} onChange={setFilters} />
          <label className="glass-card block p-4 text-sm">
            Top-K
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="mt-1 w-full rounded-xl border bg-transparent px-3 py-2"
              style={{ borderColor: "var(--border)" }}
            />
          </label>
        </div>
        <div className="space-y-4">
          <ImageUploader
            previewUrl={preview}
            onFile={async (file) => {
              setPreview(fileToObjectUrl(file));
              await searchSimilar(file, k);
            }}
          />
          {error ? <p className="text-sm text-rose-400">{error}</p> : null}
          {isLoading ? <LoadingSpinner label="Searching FAISS index" /> : <ResultsGallery results={filtered} selected={selected} onCompare={toggleCompare} />}
        </div>
      </div>
      {selected.length > 0 ? (
        <section className="glass-card p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-semibold">Comparison</h2>
            <button className="btn-ghost text-xs" onClick={clearCompare}>
              Clear
            </button>
          </div>
          <div className="grid gap-4 md:grid-cols-3">
            {selected.map((car) => (
              <div key={`${car.rank}-${car.path}`}>
                <p className="font-medium">{car.brand.replace(/_/g, " ")}</p>
                <p className="subtle text-sm">Similarity {(car.similarity * 100).toFixed(1)}%</p>
                <p className="text-sm">{car.price ? formatUSD(car.price) : "—"}</p>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
