import React from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { installTheme } from "./lib/theme";
import App from "./App";
import "./styles/index.css";
installTheme();
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: 0, staleTime: 60000 } } })}
    >
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
