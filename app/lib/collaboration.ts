export const SHARED_DRAWING_TOOLS = [
  "pencil", "brush", "marker", "spray", "eraser", "fill",
  "line", "rectangle", "circle", "triangle", "star",
] as const;

export type SharedDrawingTool = (typeof SHARED_DRAWING_TOOLS)[number];
export type SharedDrawingPoint = { x: number; y: number; pressure: number };

export type SharedDrawingOperation = {
  id: string;
  participantId: string;
  author: string;
  kind: "gesture" | "fill" | "clear";
  tool: SharedDrawingTool;
  color: string;
  size: number;
  points: SharedDrawingPoint[];
  seed: number;
  createdAt: string;
  sequence?: number;
};

export type SharedParticipant = {
  id: string;
  username: string;
  accent: string;
  lastSeenAt: string;
};

export type SharedRoomSnapshot = {
  roomId: string;
  title: string;
  ownerUsername: string;
  participants: SharedParticipant[];
  operations: SharedDrawingOperation[];
  latestSequence: number;
};

export type SharedRoomSession = {
  roomId: string;
  participantId: string;
  sessionToken: string;
  username: string;
};

export const normalizeUsername = (value: unknown) => typeof value === "string"
  ? value.trim().replace(/[^a-zA-Z0-9_.-]/g, "").slice(0, 24)
  : "";

export function validateSharedOperation(value: unknown): SharedDrawingOperation | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const input = value as Record<string, unknown>;
  const kind = input.kind;
  const tool = input.tool;
  if (!(["gesture", "fill", "clear"] as const).includes(kind as "gesture" | "fill" | "clear")) return null;
  if (!SHARED_DRAWING_TOOLS.includes(tool as SharedDrawingTool)) return null;
  const author = normalizeUsername(input.author);
  const participantId = typeof input.participantId === "string" ? input.participantId.slice(0, 64) : "";
  const id = typeof input.id === "string" ? input.id.slice(0, 80) : "";
  const color = typeof input.color === "string" && /^#[0-9a-f]{6}$/i.test(input.color) ? input.color : "#18312e";
  const rawPoints = Array.isArray(input.points) ? input.points.slice(0, 256) : [];
  const points = rawPoints.flatMap((point) => {
    if (!point || typeof point !== "object" || Array.isArray(point)) return [];
    const candidate = point as Record<string, unknown>;
    const x = Number(candidate.x);
    const y = Number(candidate.y);
    const pressure = Number(candidate.pressure);
    if (![x, y, pressure].every(Number.isFinite)) return [];
    return [{
      x: Math.min(1, Math.max(0, x)),
      y: Math.min(1, Math.max(0, y)),
      pressure: Math.min(1, Math.max(0.05, pressure)),
    }];
  });
  if (!id || !author || !participantId || (kind !== "clear" && points.length < 1)) return null;
  return {
    id,
    participantId,
    author,
    kind: kind as SharedDrawingOperation["kind"],
    tool: tool as SharedDrawingTool,
    color,
    size: Math.min(72, Math.max(3, Number(input.size) || 18)),
    points,
    seed: Math.max(0, Math.floor(Number(input.seed) || 0)) % 2147483647,
    createdAt: typeof input.createdAt === "string" ? input.createdAt.slice(0, 40) : new Date().toISOString(),
  };
}

export function sharedRoomInviteUrl(origin: string, roomId: string, username: string) {
  const url = new URL(origin);
  url.searchParams.set("room", roomId);
  if (username) url.searchParams.set("invite", normalizeUsername(username));
  return url.toString();
}
