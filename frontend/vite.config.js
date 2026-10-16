import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/recharts")) return "recharts";
          if (id.includes("node_modules/victory-vendor")) return "chart-vendor";
          if (id.includes("node_modules/d3-")) return "d3";
          if (id.includes("node_modules/lucide-react")) return "icons";
        },
      },
    },
  },
});
