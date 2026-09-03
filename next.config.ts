import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Sites runs the complete Vinext worker with D1. Vercel receives a static
  // export and proxies the collaboration endpoint back to that durable
  // backend (see vercel.json), so the two deployments stay feature-equivalent.
  output: process.env.VERCEL === "1" ? "export" : undefined,
};

export default nextConfig;
