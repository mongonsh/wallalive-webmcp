"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CaptureTarget } from "../lib/drawing";
import { makeTransparentArtworkPixels } from "../lib/target-cutout";
import type { SharedDrawingOperation, SharedDrawingPoint, SharedDrawingTool, SharedParticipant, SharedRoomSession } from "../lib/collaboration";

type Tool = SharedDrawingTool;

type DrawingWallProps = {
  open: boolean;
  onClose: () => void;
  onMake3D: (drawing: { dataUrl: string; target: CaptureTarget }) => void;
  sharedSession?: SharedRoomSession | null;
  sharedParticipants?: SharedParticipant[];
  sharedOperations?: SharedDrawingOperation[];
  onSharedOperation?: (operation: SharedDrawingOperation) => void;
};

const tools: Array<{ id: Tool; label: string; glyph: string }> = [
  { id: "pencil", label: "Pencil", glyph: "✎" },
  { id: "brush", label: "Brush", glyph: "●" },
  { id: "marker", label: "Marker", glyph: "▰" },
  { id: "spray", label: "Spray", glyph: "⁙" },
  { id: "eraser", label: "Eraser", glyph: "◇" },
  { id: "fill", label: "Fill", glyph: "◩" },
  { id: "line", label: "Line", glyph: "╱" },
  { id: "rectangle", label: "Box", glyph: "□" },
  { id: "circle", label: "Circle", glyph: "○" },
  { id: "triangle", label: "Triangle", glyph: "△" },
  { id: "star", label: "Star", glyph: "☆" },
];

const palette = [
  "#18312e", "#ffffff", "#ff674d", "#ff9e4f", "#ffd84a", "#c8f15a", "#4fbd76", "#5fc7df",
  "#3978d4", "#7d67c7", "#d85fba", "#f2a5b9", "#915a3c", "#b9aa92", "#73777d", "#2d2638",
];

type Point = { x: number; y: number; pressure: number };
type Gesture = { pointerId: number; start: Point; last: Point; base: ImageData | null; moved: boolean; id: string; points: Point[]; seed: number };

const normalizePoint = (point: Point, canvas: HTMLCanvasElement): SharedDrawingPoint => ({
  x: point.x / canvas.width,
  y: point.y / canvas.height,
  pressure: point.pressure,
});

const denormalizePoint = (point: SharedDrawingPoint, canvas: HTMLCanvasElement): Point => ({
  x: point.x * canvas.width,
  y: point.y * canvas.height,
  pressure: point.pressure,
});

function pseudoRandom(seed: number, index: number) {
  const value = Math.sin(seed * 0.0001 + index * 12.9898) * 43758.5453;
  return value - Math.floor(value);
}

function canvasPoint(canvas: HTMLCanvasElement, event: PointerEvent | React.PointerEvent<HTMLCanvasElement>): Point {
  const bounds = canvas.getBoundingClientRect();
  const pressure = event.pointerType === "mouse" ? 0.5 : Math.max(0.18, event.pressure || 0.5);
  return {
    x: (event.clientX - bounds.left) * canvas.width / Math.max(1, bounds.width),
    y: (event.clientY - bounds.top) * canvas.height / Math.max(1, bounds.height),
    pressure,
  };
}

function hexRgb(color: string) {
  const value = color.replace("#", "");
  const normalized = value.length === 3 ? value.split("").map((digit) => digit + digit).join("") : value;
  return [0, 2, 4].map((offset) => Number.parseInt(normalized.slice(offset, offset + 2), 16));
}

function floodFill(canvas: HTMLCanvasElement, start: Point, color: string) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;
  const { width, height } = canvas;
  const image = context.getImageData(0, 0, width, height);
  const pixels = image.data;
  const startX = Math.max(0, Math.min(width - 1, Math.round(start.x)));
  const startY = Math.max(0, Math.min(height - 1, Math.round(start.y)));
  const startIndex = (startY * width + startX) * 4;
  const target = [pixels[startIndex], pixels[startIndex + 1], pixels[startIndex + 2], pixels[startIndex + 3]];
  const replacement = [...hexRgb(color), 255];
  if (target.every((value, index) => Math.abs(value - replacement[index]) < 4)) return;

  const tolerance = 30;
  const matches = (index: number) => Math.abs(pixels[index] - target[0]) <= tolerance
    && Math.abs(pixels[index + 1] - target[1]) <= tolerance
    && Math.abs(pixels[index + 2] - target[2]) <= tolerance
    && Math.abs(pixels[index + 3] - target[3]) <= tolerance;
  const queue = new Int32Array(width * height);
  const visited = new Uint8Array(width * height);
  let read = 0;
  let write = 0;
  queue[write++] = startY * width + startX;
  visited[startY * width + startX] = 1;
  const enqueue = (position: number) => {
    if (!visited[position]) {
      visited[position] = 1;
      queue[write++] = position;
    }
  };
  while (read < write) {
    const position = queue[read++];
    const x = position % width;
    const y = Math.floor(position / width);
    const index = position * 4;
    if (!matches(index)) continue;
    pixels[index] = replacement[0];
    pixels[index + 1] = replacement[1];
    pixels[index + 2] = replacement[2];
    pixels[index + 3] = replacement[3];
    if (x > 0) enqueue(position - 1);
    if (x + 1 < width) enqueue(position + 1);
    if (y > 0) enqueue(position - width);
    if (y + 1 < height) enqueue(position + width);
  }
  context.putImageData(image, 0, 0);
}

function strokeStyle(context: CanvasRenderingContext2D, tool: Tool, color: string, size: number, pressure: number) {
  context.globalCompositeOperation = "source-over";
  context.globalAlpha = tool === "marker" ? 0.24 : tool === "pencil" ? 0.78 : 1;
  context.strokeStyle = tool === "eraser" ? "#ffffff" : color;
  context.fillStyle = tool === "eraser" ? "#ffffff" : color;
  context.lineCap = "round";
  context.lineJoin = "round";
  const scale = tool === "pencil" ? 0.42 : tool === "marker" ? 1.7 : tool === "eraser" ? 1.8 : 1;
  context.lineWidth = Math.max(1.5, size * scale * (0.7 + pressure * 0.6));
}

function drawShape(context: CanvasRenderingContext2D, tool: Tool, start: Point, end: Point, color: string, size: number) {
  strokeStyle(context, tool, color, size, end.pressure);
  const width = end.x - start.x;
  const height = end.y - start.y;
  const centerX = start.x + width / 2;
  const centerY = start.y + height / 2;
  context.beginPath();
  if (tool === "line") {
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
  } else if (tool === "rectangle") {
    context.rect(start.x, start.y, width, height);
  } else if (tool === "circle") {
    context.ellipse(centerX, centerY, Math.abs(width / 2), Math.abs(height / 2), 0, 0, Math.PI * 2);
  } else if (tool === "triangle") {
    context.moveTo(centerX, start.y);
    context.lineTo(end.x, end.y);
    context.lineTo(start.x, end.y);
    context.closePath();
  } else if (tool === "star") {
    const radius = Math.max(8, Math.hypot(width, height) / 2);
    const inner = radius * 0.43;
    for (let index = 0; index < 10; index += 1) {
      const angle = -Math.PI / 2 + index * Math.PI / 5;
      const pointRadius = index % 2 === 0 ? radius : inner;
      const x = start.x + Math.cos(angle) * pointRadius;
      const y = start.y + Math.sin(angle) * pointRadius;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    }
    context.closePath();
  }
  context.stroke();
  context.globalAlpha = 1;
}

function drawSpray(context: CanvasRenderingContext2D, point: Point, color: string, size: number, seed: number, offset = 0) {
  strokeStyle(context, "spray", color, size, point.pressure);
  const radius = size * 1.5;
  const density = Math.max(12, Math.round(size * 1.2));
  for (let index = 0; index < density; index += 1) {
    const angle = pseudoRandom(seed, offset + index * 2) * Math.PI * 2;
    const distance = Math.sqrt(pseudoRandom(seed, offset + index * 2 + 1)) * radius;
    context.fillRect(point.x + Math.cos(angle) * distance, point.y + Math.sin(angle) * distance, Math.max(1.2, size * 0.09), Math.max(1.2, size * 0.09));
  }
  context.globalAlpha = 1;
}

function renderSharedOperation(canvas: HTMLCanvasElement, operation: SharedDrawingOperation) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return;
  if (operation.kind === "clear") {
    context.globalAlpha = 1;
    context.globalCompositeOperation = "source-over";
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const points = operation.points.map((point) => denormalizePoint(point, canvas));
  if (operation.kind === "fill") {
    floodFill(canvas, points[0], operation.color);
    return;
  }
  if (["line", "rectangle", "circle", "triangle", "star"].includes(operation.tool)) {
    drawShape(context, operation.tool, points[0], points.at(-1) ?? points[0], operation.color, operation.size);
    return;
  }
  if (operation.tool === "spray") {
    points.forEach((point, index) => drawSpray(context, point, operation.color, operation.size, operation.seed, index * 200));
    return;
  }
  const list = points.length === 1 ? [points[0], { ...points[0], x: points[0].x + 0.2, y: points[0].y + 0.2 }] : points;
  for (let index = 1; index < list.length; index += 1) {
    strokeStyle(context, operation.tool, operation.color, operation.size, list[index].pressure);
    context.beginPath();
    context.moveTo(list[index - 1].x, list[index - 1].y);
    context.lineTo(list[index].x, list[index].y);
    context.stroke();
  }
  context.globalAlpha = 1;
}

export function DrawingWall({ open, onClose, onMake3D, sharedSession = null, sharedParticipants = [], sharedOperations = [], onSharedOperation }: DrawingWallProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const historyRef = useRef<string[]>([]);
  const historyIndexRef = useRef(-1);
  const appliedSharedOperationsRef = useRef(new Set<string>());
  const initializedWallRef = useRef<string | null>(null);
  const [tool, setTool] = useState<Tool>("brush");
  const [color, setColor] = useState("#18312e");
  const [size, setSize] = useState(18);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [historyLength, setHistoryLength] = useState(0);
  const [status, setStatus] = useState("Start with one big character. Leave space around it.");

  const saveSnapshot = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const next = historyRef.current.slice(0, historyIndexRef.current + 1);
    next.push(canvas.toDataURL("image/png"));
    if (next.length > 18) next.shift();
    historyRef.current = next;
    historyIndexRef.current = next.length - 1;
    setHistoryIndex(historyIndexRef.current);
    setHistoryLength(next.length);
  }, []);

  const restore = useCallback((index: number) => {
    const canvas = canvasRef.current;
    const source = historyRef.current[index];
    const context = canvas?.getContext("2d");
    if (!canvas || !context || !source) return;
    const image = new Image();
    image.onload = () => {
      context.globalAlpha = 1;
      context.globalCompositeOperation = "source-over";
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    };
    image.src = source;
    historyIndexRef.current = index;
    setHistoryIndex(index);
  }, []);

  const sharedRoomId = sharedSession?.roomId ?? "";

  useEffect(() => {
    if (!open) return;
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const wallScope = sharedRoomId || "solo";
    if (initializedWallRef.current === wallScope) {
      setStatus(sharedRoomId ? `Room ${sharedRoomId} is live. Keep drawing together.` : "Your drawing is still here.");
      return;
    }
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    initializedWallRef.current = wallScope;
    appliedSharedOperationsRef.current = new Set();
    historyRef.current = [canvas.toDataURL("image/png")];
    historyIndexRef.current = 0;
    setHistoryIndex(0);
    setHistoryLength(1);
    setStatus(sharedRoomId ? `Room ${sharedRoomId} is live. Draw together.` : "Start with one big character. Leave space around it.");
  }, [open, sharedRoomId]);

  useEffect(() => {
    if (!open || !sharedSession) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    let applied = 0;
    sharedOperations.forEach((operation) => {
      if (appliedSharedOperationsRef.current.has(operation.id)) return;
      renderSharedOperation(canvas, operation);
      appliedSharedOperationsRef.current.add(operation.id);
      applied += 1;
    });
    if (applied) {
      saveSnapshot();
    }
  }, [open, saveSnapshot, sharedOperations, sharedSession]);

  const drawSegment = useCallback((from: Point, to: Point, selectedTool = tool, seed = 1, offset = 0) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    if (selectedTool === "spray") {
      drawSpray(context, to, color, size, seed, offset);
      return;
    }
    strokeStyle(context, selectedTool, color, size, to.pressure);
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.globalAlpha = 1;
  }, [color, size, tool]);

  const pointerDown = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const canvas = event.currentTarget;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return;
    const point = canvasPoint(canvas, event);
    canvas.setPointerCapture(event.pointerId);
    if (tool === "fill") {
      floodFill(canvas, point, color);
      if (sharedSession && onSharedOperation) {
        const operation: SharedDrawingOperation = {
          id: crypto.randomUUID(), participantId: sharedSession.participantId, author: sharedSession.username,
          kind: "fill", tool, color, size, points: [normalizePoint(point, canvas)], seed: 0, createdAt: new Date().toISOString(),
        };
        appliedSharedOperationsRef.current.add(operation.id);
        onSharedOperation(operation);
      }
      saveSnapshot();
      setStatus("Area filled. Add details or make it 3D.");
      return;
    }
    const shape = ["line", "rectangle", "circle", "triangle", "star"].includes(tool);
    const id = crypto.randomUUID();
    const seed = Math.floor(Math.random() * 2147483646) + 1;
    gestureRef.current = { pointerId: event.pointerId, start: point, last: point, base: shape ? context.getImageData(0, 0, canvas.width, canvas.height) : null, moved: false, id, points: [point], seed };
    if (!shape) {
      drawSegment(point, { ...point, x: point.x + 0.2, y: point.y + 0.2 }, tool, seed, 0);
    }
  }, [color, drawSegment, onSharedOperation, saveSnapshot, sharedSession, size, tool]);

  const pointerMove = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const canvas = event.currentTarget;
    const gesture = gestureRef.current;
    const context = canvas.getContext("2d");
    if (!gesture || gesture.pointerId !== event.pointerId || !context) return;
    const shape = ["line", "rectangle", "circle", "triangle", "star"].includes(tool);
    const native = event.nativeEvent;
    const samples = native.getCoalescedEvents?.() ?? [native];
    if (shape) {
      const point = canvasPoint(canvas, samples[samples.length - 1]);
      if (gesture.base) context.putImageData(gesture.base, 0, 0);
      drawShape(context, tool, gesture.start, point, color, size);
      gesture.last = point;
      gesture.points = [gesture.start, point];
    } else {
      for (const sample of samples) {
        const point = canvasPoint(canvas, sample);
        drawSegment(gesture.last, point, tool, gesture.seed, gesture.points.length * 200);
        gesture.last = point;
        if (gesture.points.length < 256) gesture.points.push(point);
      }
    }
    gesture.moved = true;
  }, [color, drawSegment, size, tool]);

  const pointerUp = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = gestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId) return;
    gestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (sharedSession && onSharedOperation) {
      const operation: SharedDrawingOperation = {
        id: gesture.id,
        participantId: sharedSession.participantId,
        author: sharedSession.username,
        kind: "gesture",
        tool,
        color,
        size,
        points: gesture.points.map((point) => normalizePoint(point, event.currentTarget)),
        seed: gesture.seed,
        createdAt: new Date().toISOString(),
      };
      appliedSharedOperationsRef.current.add(operation.id);
      onSharedOperation(operation);
    }
    saveSnapshot();
    setStatus("Saved. Undo anytime—or make the wall alive.");
  }, [color, onSharedOperation, saveSnapshot, sharedSession, size, tool]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.globalAlpha = 1;
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (sharedSession && onSharedOperation) {
      const operation: SharedDrawingOperation = {
        id: crypto.randomUUID(), participantId: sharedSession.participantId, author: sharedSession.username,
        kind: "clear", tool, color, size, points: [], seed: 0, createdAt: new Date().toISOString(),
      };
      appliedSharedOperationsRef.current.add(operation.id);
      onSharedOperation(operation);
    }
    saveSnapshot();
    setStatus("Clean wall. Make something new.");
  }, [color, onSharedOperation, saveSnapshot, sharedSession, size, tool]);

  const make3D = useCallback(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d", { willReadFrequently: true });
    if (!canvas || !context) return;
    const image = context.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = makeTransparentArtworkPixels(image.data, canvas.width, canvas.height);
    let left = canvas.width;
    let top = canvas.height;
    let right = -1;
    let bottom = -1;
    for (let y = 0; y < canvas.height; y += 3) {
      for (let x = 0; x < canvas.width; x += 3) {
        const index = (y * canvas.width + x) * 4;
        if (pixels[index + 3] > 20) {
          left = Math.min(left, x);
          top = Math.min(top, y);
          right = Math.max(right, x);
          bottom = Math.max(bottom, y);
        }
      }
    }
    if (right < 0) {
      setStatus("Draw something first—then I can find its shape.");
      return;
    }
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = canvas.width;
    exportCanvas.height = canvas.height;
    const exportContext = exportCanvas.getContext("2d");
    if (!exportContext) {
      setStatus("This browser could not prepare the transparent artwork.");
      return;
    }
    const transparentImage = exportContext.createImageData(canvas.width, canvas.height);
    transparentImage.data.set(pixels);
    exportContext.putImageData(transparentImage, 0, 0);
    onMake3D({
      dataUrl: exportCanvas.toDataURL("image/png"),
      target: { x: ((left + right) / 2) / canvas.width, y: ((top + bottom) / 2) / canvas.height },
    });
  }, [onMake3D]);

  if (!open) return null;

  return (
    <div className="drawing-wall-backdrop" role="dialog" aria-modal="true" aria-labelledby="drawing-wall-title">
      <section className="drawing-wall">
        <header>
          <div><span>{sharedSession ? `LIVE ROOM · ${sharedSession.roomId}` : "WALL STUDIO"}</span><h2 id="drawing-wall-title">{sharedSession ? "Draw together. Wake the whole cast." : "Draw a world. Wake a friend."}</h2></div>
          <div className="wall-history">
            {sharedSession ? <div className="wall-collaborators" aria-label="Creators in this room">{sharedParticipants.slice(0, 5).map((participant) => <i key={participant.id} style={{ background: participant.accent }} title={`@${participant.username}`}>{participant.username.slice(0, 1).toUpperCase()}</i>)}</div> : null}
            <button disabled={Boolean(sharedSession) || historyIndex <= 0} onClick={() => restore(historyIndex - 1)} aria-label={sharedSession ? "Undo is disabled in a live shared room" : "Undo"}>↶</button>
            <button disabled={Boolean(sharedSession) || historyIndex < 0 || historyIndex >= historyLength - 1} onClick={() => restore(historyIndex + 1)} aria-label={sharedSession ? "Redo is disabled in a live shared room" : "Redo"}>↷</button>
            <button onClick={clearCanvas}>CLEAR</button>
            <button onClick={onClose} aria-label="Close drawing wall">×</button>
          </div>
        </header>

        <div className="wall-guide" aria-label="Three steps to make a playable character">
          <span><b>1</b> Draw one big character</span>
          <span><b>2</b> Add a face and limbs</span>
          <span><b>3</b> Make it 3D</span>
        </div>

        <div className="wall-workspace">
          <aside className="wall-tools" aria-label="Drawing tools">
            {tools.map((item) => <button key={item.id} className={tool === item.id ? "active" : ""} onClick={() => setTool(item.id)} title={item.label}><i>{item.glyph}</i><span>{item.label}</span></button>)}
          </aside>
          <div className="wall-canvas-wrap">
            <canvas
              ref={canvasRef}
              width="1400"
              height="850"
              onPointerDown={pointerDown}
              onPointerMove={pointerMove}
              onPointerUp={pointerUp}
              onPointerCancel={pointerUp}
              aria-label="Large drawing wall"
            />
            <div className="wall-canvas-status"><i />{status}</div>
          </div>
        </div>

        <footer>
          <div className="wall-palette" aria-label="Paint colors">
            {palette.map((swatch) => <button key={swatch} className={color === swatch ? "active" : ""} style={{ background: swatch }} onClick={() => setColor(swatch)} aria-label={`Use color ${swatch}`} />)}
            <label title="Custom color"><input type="color" value={color} onChange={(event) => setColor(event.target.value)} /><span>＋</span></label>
          </div>
          <label className="wall-size">SIZE<input type="range" min="3" max="72" value={size} onChange={(event) => setSize(Number(event.target.value))} /><i style={{ width: Math.max(5, size / 2), height: Math.max(5, size / 2), background: color }} /></label>
          <button className="wall-make-3d" onClick={make3D}><span>MAKE IT 3D</span><i>↗</i></button>
        </footer>
      </section>
    </div>
  );
}
