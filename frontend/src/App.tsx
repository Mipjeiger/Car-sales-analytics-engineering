import type { ReactNode } from "react";
import { BrowserRouter } from "react-router-dom";
import { Provider } from "react-redux";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { store } from "@/store/store";
import { AppRoutes } from '@/AppRoutes';
import { ErrorBoundary } from "@/components/common/ErrorBoundary";
import { useTheme } from "@/hooks/useTheme";

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
});

function ThemeBoot({ children }: { children: ReactNode }) {
  useTheme();
  return <>{children}</>;
}

export default function App() {
  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ErrorBoundary>
          <ThemeBoot>
            <BrowserRouter>
              <AppRoutes />
            </BrowserRouter>
          </ThemeBoot>
        </ErrorBoundary>
      </QueryClientProvider>
    </Provider>
  );
}
