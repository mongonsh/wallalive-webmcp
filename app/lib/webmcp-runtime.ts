export type RegisterableWebMCPTool = {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (input: Record<string, unknown>, options?: { signal?: AbortSignal }) => Promise<unknown> | unknown;
};

export type RegisteredWebMCPTool = {
  name: string;
  title?: string | null;
  description: string;
  inputSchema?: string;
};

export type WebMCPModelContext = {
  registerTool: (tool: RegisterableWebMCPTool, options?: { signal?: AbortSignal }) => Promise<void>;
  getTools?: () => Promise<RegisteredWebMCPTool[]>;
  executeTool?: (tool: RegisteredWebMCPTool, input?: Record<string, unknown>, options?: { signal?: AbortSignal }) => Promise<string>;
};

export type WebMCPRuntimeCheck = {
  status: "registered" | "verified";
  registeredCount: number;
  verifiedTool: string | null;
};

function parseToolResult(result: string) {
  try {
    return JSON.parse(result) as Record<string, unknown>;
  } catch {
    throw new Error("The WebMCP probe returned a non-JSON result.");
  }
}

/**
 * Registers every WallAlive tool through the browser's native WebMCP API.
 * Where the current implementation exposes the spec's in-page evaluation
 * methods, this also discovers the tools and executes one read-only probe.
 */
export async function registerAndVerifyWebMCP(
  context: WebMCPModelContext,
  tools: RegisterableWebMCPTool[],
  signal: AbortSignal,
): Promise<WebMCPRuntimeCheck> {
  await Promise.all(tools.map((tool) => context.registerTool(tool, { signal })));
  if (signal.aborted) throw signal.reason ?? new DOMException("WebMCP check cancelled", "AbortError");

  if (!context.getTools || !context.executeTool) {
    return { status: "registered", registeredCount: tools.length, verifiedTool: null };
  }

  const visibleTools = await context.getTools();
  const expectedNames = new Set(tools.map((tool) => tool.name));
  const visibleNames = new Set(visibleTools.map((tool) => tool.name));
  const missing = [...expectedNames].filter((name) => !visibleNames.has(name));
  if (missing.length) throw new Error(`WebMCP discovery missed: ${missing.join(", ")}`);

  const probe = visibleTools.find((tool) => tool.name === "inspect_creative_scene");
  if (!probe) throw new Error("The read-only WebMCP probe is unavailable.");
  const result = parseToolResult(await context.executeTool(probe, {}, { signal }));
  const verification = typeof result.verification === "object" && result.verification !== null
    ? result.verification as Record<string, unknown>
    : null;
  if (result.ok !== true || verification?.cameraDataIncluded !== false) {
    throw new Error("The read-only WebMCP probe failed its privacy assertion.");
  }

  return { status: "verified", registeredCount: expectedNames.size, verifiedTool: probe.name };
}
