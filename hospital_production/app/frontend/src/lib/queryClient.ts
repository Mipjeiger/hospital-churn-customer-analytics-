import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,        // 30 s
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

