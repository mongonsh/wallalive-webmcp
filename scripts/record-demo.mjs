import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const outputDirectory = resolve(process.argv[2] ?? "outputs/demo-frames");
const targetUrl = process.argv[3] ?? "https://wallalive-webmcp.mungunshagai-tb.chatgpt.site";
const smokeMode = process.argv.includes("--smoke");
const mobileMode = process.argv.includes("--mobile");
const chromePath = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profileDirectory = mkdtempSync(join(tmpdir(), "wallalive-demo-chrome-"));
const port = 9300 + Math.floor(Math.random() * 500);
const fps = 10;
const width = mobileMode ? 390 : 1440;
const height = mobileMode ? 844 : 900;

mkdirSync(outputDirectory, { recursive: true });

const chrome = spawn(chromePath, [
  "--headless=new",
  `--remote-debugging-port=${port}`,
  "--remote-allow-origins=*",
  `--user-data-dir=${profileDirectory}`,
  `--window-size=${width},${height}`,
  "--force-device-scale-factor=1",
  "--hide-scrollbars",
  "--enable-unsafe-swiftshader",
  "--use-angle=swiftshader-webgl",
  "--no-first-run",
  "--no-default-browser-check",
  targetUrl,
], { stdio: ["ignore", "ignore", "pipe"] });

const wait = (milliseconds) => new Promise((resolveWait) => setTimeout(resolveWait, milliseconds));

async function findPage() {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    try {
      const pages = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json());
      const page = pages.find((item) => item.type === "page" && item.url.startsWith("http"));
      if (page?.webSocketDebuggerUrl) return page;
    } catch {
      // Chrome may still be starting; retry until the DevTools endpoint is ready.
    }
    await wait(150);
  }
  throw new Error("Chrome DevTools did not expose the WallAlive page.");
}

let socket;
let commandId = 0;
const pending = new Map();

function send(method, params = {}) {
  const id = ++commandId;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolveCommand, rejectCommand) => {
    pending.set(id, { resolve: resolveCommand, reject: rejectCommand });
  });
}

async function clickButton(label) {
  const expression = `(() => {
    const button = [...document.querySelectorAll('button')].find((item) => item.textContent.trim().includes(${JSON.stringify(label)}));
    if (!button) return false;
    button.click();
    return true;
  })()`;
  const response = await send("Runtime.evaluate", { expression, returnByValue: true });
  if (!response?.result?.value) throw new Error(`Could not find demo button: ${label}`);
  console.log(`Clicked ${label}`);
}

async function waitForHydratedButton(label) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    const expression = `(() => {
      const button = [...document.querySelectorAll('button')].find((item) => item.textContent.trim().includes(${JSON.stringify(label)}));
      return Boolean(button && Object.keys(button).some((key) => key.startsWith('__reactProps')));
    })()`;
    const response = await send("Runtime.evaluate", { expression, returnByValue: true });
    if (response?.result?.value) return;
    await wait(250);
  }
  throw new Error(`WallAlive did not hydrate the ${label} button.`);
}

try {
  const page = await findPage();
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener("open", resolveOpen, { once: true });
    socket.addEventListener("error", rejectOpen, { once: true });
  });

  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const task = pending.get(message.id);
      if (!task) return;
      pending.delete(message.id);
      if (message.error) task.reject(new Error(message.error.message));
      else task.resolve(message.result);
      return;
    }
    if (message.method === "Runtime.exceptionThrown") {
      console.error("Browser exception:", message.params.exceptionDetails?.text, message.params.exceptionDetails?.exception?.description ?? "");
    }
  });

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  await send("Runtime.evaluate", { expression: "window.scrollTo(0, 0)" });
  await waitForHydratedButton("PLAY JUDGE DEMO");
  let frameNumber = 0;
  let recording = true;
  const captureLoop = (async () => {
    while (recording) {
      const started = Date.now();
      const screenshot = await send("Page.captureScreenshot", {
        format: "jpeg",
        quality: 88,
        fromSurface: true,
        captureBeyondViewport: false,
        optimizeForSpeed: true,
      });
      writeFileSync(join(outputDirectory, `frame-${String(frameNumber).padStart(5, "0")}.jpg`), Buffer.from(screenshot.data, "base64"));
      frameNumber += 1;
      await wait(Math.max(0, 1000 / fps - (Date.now() - started)));
    }
  })();

  await wait(4000);
  await clickButton("PLAY JUDGE DEMO");
  if (mobileMode) await send("Runtime.evaluate", { expression: "document.querySelector('.camera-frame')?.scrollIntoView({ block: 'center' })" });
  await wait(800);
  const demoState = await send("Runtime.evaluate", { expression: "document.querySelector('.notice')?.textContent", returnByValue: true });
  console.log(`Demo state: ${demoState?.result?.value ?? "unknown"}`);
  await wait(6200);
  if (!smokeMode) {
    await clickButton("tools");
    await wait(6000);
    await clickButton("privacy");
    await wait(6000);
    await clickButton("history");
    await wait(6000);
    await clickButton("agent");
    await wait(3500);
    await clickButton("Dance");
    await wait(4000);
    await clickButton("Wave");
    await wait(4000);
  }

  recording = false;
  await captureLoop;
  writeFileSync(join(outputDirectory, "manifest.json"), JSON.stringify({ fps, width, height, frameCount: frameNumber, source: targetUrl, smokeMode, mobileMode }, null, 2));
  console.log(`Recorded ${frameNumber} frames to ${outputDirectory}`);
} finally {
  socket?.close();
  chrome.kill("SIGTERM");
  try {
    rmSync(profileDirectory, { recursive: true, force: true, maxRetries: 5, retryDelay: 100 });
  } catch {
    // Chrome can briefly retain profile files during shutdown; the OS temp cleaner will recover them.
  }
}
