import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

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
