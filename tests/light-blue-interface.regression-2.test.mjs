import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("child-facing shell uses one light blue workspace with concise actions", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  const v45 = css.slice(css.indexOf("/* v45"));
  assert.match(v45, /--ink: #17324a/);
  assert.match(v45, /--cream: #f3f9ff/);
  assert.match(v45, /--paper: #ffffff/);
  assert.match(v45, /\.steps-panel \{ display: none !important; \}/);
  assert.match(v45, /\.alive-layout[\s\S]+display: block !important/);
  assert.match(page, /<h1>Draw it\.<br \/><em>Bring it to life\.<\/em><\/h1>/);
  assert.match(page, /Draw, upload, or scan\. Then play together in 3D\./);
  assert.match(page, />UPLOAD<\/button>/);
  assert.match(page, />DRAW <span>✦<\/span><\/button>/);
  assert.match(page, />TOGETHER <span>∞<\/span><\/button>/);
  assert.doesNotMatch(page, /className="judge-prompt-banner"/);
});
