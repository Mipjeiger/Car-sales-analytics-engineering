import { useCallback, useState } from "react";
import { getErrorMessage } from "@/api/client";

export function useApi<TArgs extends unknown[], TResult>(fn: (...args: TArgs) => Promise<TResult>) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<TResult | null>(null);

  const run = useCallback(
    async (...args: TArgs) => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await fn(...args);
        setData(result);
        return result;
      } catch (err) {
        const message = getErrorMessage(err);
        setError(message);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [fn],
  );

  return { run, isLoading, error, data, setError };
}
