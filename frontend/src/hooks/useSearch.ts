import { useMutation, useQuery } from "@tanstack/react-query";
import { searchApi } from "@/api/endpoints/search";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { setLoading, setResults, setFilters, toggleCompare, clearCompare } from "@/store/slices/searchSlice";
import { estimatePriceFromBrand } from "@/utils/helpers";

export function useSearch() {
  const dispatch = useAppDispatch();
  const search = useAppSelector((s) => s.search);

  const brandsQuery = useQuery({ queryKey: ["search-brands"], queryFn: searchApi.brands, staleTime: 60_000 });
  const statsQuery = useQuery({ queryKey: ["search-stats"], queryFn: searchApi.stats, staleTime: 60_000 });

  const mutation = useMutation({
    mutationFn: ({ file, k }: { file: File; k: number }) => searchApi.similar(file, k),
    onMutate: () => dispatch(setLoading(true)),
    onSuccess: (data, vars) => {
      const results = data.results.map((r) => ({
        ...r,
        price: r.price ?? estimatePriceFromBrand(r.brand, r.similarity),
        model: r.model ?? r.brand.replace(/_/g, " "),
      }));
      dispatch(setResults({ results, queryImage: vars.file.name }));
    },
    onSettled: () => dispatch(setLoading(false)),
  });

  return {
    ...search,
    brands: brandsQuery.data?.brands ?? [],
    stats: statsQuery.data,
    searchSimilar: (file: File, k = 8) => mutation.mutateAsync({ file, k }),
    setFilters: (next: Parameters<typeof setFilters>[0]) => dispatch(setFilters(next)),
    toggleCompare: (car: Parameters<typeof toggleCompare>[0]) => dispatch(toggleCompare(car)),
    clearCompare: () => dispatch(clearCompare()),
    error: mutation.error instanceof Error ? mutation.error.message : null,
  };
}
