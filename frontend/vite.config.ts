import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    watch: {
      usePolling: true,
      interval: 1000,
    },
    hmr: {
      overlay: true,
    },
    proxy: {
      "/api": {
        target: "http://fastapi:8000",  // Use service name, NOT localhost
        changeOrigin: true,
        secure: false,
      },
      "/api-proxy": {
        target: "http://fastapi:8000",  // Use service name
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api-proxy/, ""),
      },
    },
  },
  build: {
    sourcemap: false,
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          query: ["@tanstack/react-query", "axios", "@reduxjs/toolkit"],
        },
      },
    },
  },
});