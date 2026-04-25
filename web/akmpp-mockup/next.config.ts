import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle for Docker (server.js + minimal node_modules).
  // See: docs/01-app/01-getting-started/17-deploying.md
  output: "standalone",
};

export default nextConfig;
