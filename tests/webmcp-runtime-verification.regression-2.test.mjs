import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { registerAndVerifyWebMCP } from "../app/lib/webmcp-runtime.ts";

test("runtime verification discovers every registered WebMCP tool and executes a privacy-safe probe", async () => {
  const [runtime, page] = await Promise.all([
    readFile(new URL("../app/lib/webmcp-runtime.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(runtime, /await Promise\.all\(tools\.map\(\(tool\) => context\.registerTool\(tool, \{ signal \}\)\)\)/);
  assert.match(runtime, /const visibleTools = await context\.getTools\(\)/);
  assert.match(runtime, /context\.executeTool\(probe, \{\}, \{ signal \}\)/);
  assert.match(runtime, /verification\?\.cameraDataIncluded !== false/);
  assert.match(page, /registerAndVerifyWebMCP\(context, tools, controller\.signal\)/);
  assert.match(page, /WEBMCP ✓/);
  assert.match(page, /WEBMCP OFF/);
  assert.doesNotMatch(page, /DEMO READY|DEMO MODE/);
});

test("runtime actually registers, discovers, and executes a privacy-safe probe", async () => {
  const registered = [];
  let probeExecutions = 0;
  const context = {
    async registerTool(tool) { registered.push(tool); },
    async getTools() { return registered.map(({ name, title, description }) => ({ name, title, description })); },
    async executeTool(tool, input) {
      assert.equal(tool.name, "inspect_creative_scene");
      assert.deepEqual(input, {});
      probeExecutions += 1;
      return JSON.stringify({ ok: true, verification: { cameraDataIncluded: false } });
    },
  };
  const schema = { type: "object", properties: {}, additionalProperties: false };
  const annotations = { readOnlyHint: true, untrustedContentHint: true };
  const tools = [
    { name: "inspect_creative_scene", title: "Inspect", description: "Read safe state", inputSchema: schema, annotations, execute() {} },
    { name: "inspect_reconstruction_readiness", title: "Inspect readiness", description: "Read safe readiness", inputSchema: schema, annotations, execute() {} },
  ];
  const result = await registerAndVerifyWebMCP(context, tools, new AbortController().signal);
  assert.deepEqual(registered.map(({ name }) => name), ["inspect_creative_scene", "inspect_reconstruction_readiness"]);
  assert.equal(probeExecutions, 1);
  assert.deepEqual(result, { status: "verified", registeredCount: 2, verifiedTool: "inspect_creative_scene" });
});
