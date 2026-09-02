import { env } from "cloudflare:workers";
import { normalizeUsername, validateSharedOperation, type SharedDrawingOperation, type SharedParticipant, type SharedRoomSnapshot } from "../../lib/collaboration";

export const runtime = "edge";

type D1Result<T> = { results?: T[] };
type D1Statement = {
  bind: (...values: unknown[]) => D1Statement;
  first: <T>() => Promise<T | null>;
  all: <T>() => Promise<D1Result<T>>;
  run: () => Promise<{ meta?: { last_row_id?: number } }>;
};
type D1DatabaseLike = { prepare: (sql: string) => D1Statement; batch: (statements: D1Statement[]) => Promise<unknown> };

const database = () => (env as unknown as { DB: D1DatabaseLike }).DB;
const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { "cache-control": "no-store" } });
const cleanRoomId = (value: unknown) => typeof value === "string" ? value.trim().toUpperCase().replace(/[^A-Z0-9]/g, "").slice(0, 16) : "";
const cleanId = (value: unknown) => typeof value === "string" ? value.slice(0, 80) : "";

async function ensureSchema(db: D1DatabaseLike) {
  await db.batch([
    db.prepare("CREATE TABLE IF NOT EXISTS shared_rooms (id TEXT PRIMARY KEY, title TEXT NOT NULL, owner_username TEXT NOT NULL, created_at TEXT NOT NULL)"),
    db.prepare("CREATE TABLE IF NOT EXISTS shared_participants (id TEXT PRIMARY KEY, room_id TEXT NOT NULL, username TEXT NOT NULL, token_hash TEXT NOT NULL, accent TEXT NOT NULL, last_seen_at TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(room_id, username))"),
    db.prepare("CREATE TABLE IF NOT EXISTS shared_invites (id TEXT PRIMARY KEY, room_id TEXT NOT NULL, username TEXT NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL, accepted_at TEXT)"),
    db.prepare("CREATE TABLE IF NOT EXISTS shared_drawing_ops (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, room_id TEXT NOT NULL, participant_id TEXT NOT NULL, author TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)"),
    db.prepare("CREATE INDEX IF NOT EXISTS idx_shared_ops_room_sequence ON shared_drawing_ops(room_id, sequence)"),
    db.prepare("CREATE INDEX IF NOT EXISTS idx_shared_participants_room ON shared_participants(room_id, last_seen_at)"),
  ]);
}

async function hashToken(token: string) {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(token));
  return Array.from(new Uint8Array(digest)).map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function randomCode(length = 10) {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const values = crypto.getRandomValues(new Uint8Array(length));
  return Array.from(values, (value) => alphabet[value % alphabet.length]).join("");
}

function randomToken() {
  const values = crypto.getRandomValues(new Uint8Array(24));
  return Array.from(values, (value) => value.toString(16).padStart(2, "0")).join("");
}

function participantAccent(username: string) {
  const colors = ["#ff674d", "#3978d4", "#4fbd76", "#7d67c7", "#d85fba", "#ff9e4f", "#5fc7df"];
  const score = [...username].reduce((sum, letter) => sum + letter.charCodeAt(0), 0);
  return colors[score % colors.length];
}

async function authorize(db: D1DatabaseLike, roomId: string, participantId: string, token: string) {
  if (!roomId || !participantId || token.length < 20) return null;
  const row = await db.prepare("SELECT id, username, token_hash FROM shared_participants WHERE id = ? AND room_id = ?")
    .bind(participantId, roomId).first<{ id: string; username: string; token_hash: string }>();
  if (!row || row.token_hash !== await hashToken(token)) return null;
  await db.prepare("UPDATE shared_participants SET last_seen_at = ? WHERE id = ?").bind(new Date().toISOString(), participantId).run();
  return row;
}

async function snapshot(db: D1DatabaseLike, roomId: string, since = 0): Promise<SharedRoomSnapshot | null> {
  const room = await db.prepare("SELECT id, title, owner_username FROM shared_rooms WHERE id = ?").bind(roomId)
    .first<{ id: string; title: string; owner_username: string }>();
  if (!room) return null;
  const participantRows = await db.prepare("SELECT id, username, accent, last_seen_at FROM shared_participants WHERE room_id = ? ORDER BY created_at ASC")
    .bind(roomId).all<{ id: string; username: string; accent: string; last_seen_at: string }>();
  const operationRows = await db.prepare("SELECT sequence, payload FROM shared_drawing_ops WHERE room_id = ? AND sequence > ? ORDER BY sequence ASC LIMIT 400")
    .bind(roomId, Math.max(0, since)).all<{ sequence: number; payload: string }>();
  const operations = (operationRows.results ?? []).flatMap((row) => {
    try {
      const operation = validateSharedOperation(JSON.parse(row.payload));
      return operation ? [{ ...operation, sequence: row.sequence }] : [];
    } catch { return []; }
  });
  const latestSequence = operations.at(-1)?.sequence ?? since;
  return {
    roomId: room.id,
    title: room.title,
    ownerUsername: room.owner_username,
    participants: (participantRows.results ?? []).map((row): SharedParticipant => ({ id: row.id, username: row.username, accent: row.accent, lastSeenAt: row.last_seen_at })),
    operations,
    latestSequence,
  };
}

export async function GET(request: Request) {
  const db = database();
  await ensureSchema(db);
  const url = new URL(request.url);
  const roomId = cleanRoomId(url.searchParams.get("roomId"));
  const since = Math.max(0, Number(url.searchParams.get("since")) || 0);
  const room = await snapshot(db, roomId, since);
  return room ? json(room) : json({ error: "That shared room does not exist." }, 404);
}

export async function POST(request: Request) {
  if (Number(request.headers.get("content-length") || 0) > 24_000) return json({ error: "That shared drawing update is too large." }, 413);
  const body = await request.json().catch(() => null) as Record<string, unknown> | null;
  if (!body) return json({ error: "Invalid room request." }, 400);
  const db = database();
  await ensureSchema(db);
  const action = body.action;

  if (action === "create") {
    const username = normalizeUsername(body.username);
    if (username.length < 2) return json({ error: "Use at least two letters for your creator name." }, 400);
    let roomId = randomCode();
    while (await db.prepare("SELECT id FROM shared_rooms WHERE id = ?").bind(roomId).first()) roomId = randomCode();
    const participantId = crypto.randomUUID();
    const sessionToken = randomToken();
    const now = new Date().toISOString();
    await db.batch([
      db.prepare("INSERT INTO shared_rooms (id, title, owner_username, created_at) VALUES (?, ?, ?, ?)").bind(roomId, `${username}'s drawing room`, username, now),
      db.prepare("INSERT INTO shared_participants (id, room_id, username, token_hash, accent, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
        .bind(participantId, roomId, username, await hashToken(sessionToken), participantAccent(username), now, now),
    ]);
    return json({ session: { roomId, participantId, sessionToken, username }, snapshot: await snapshot(db, roomId, 0) }, 201);
  }

  if (action === "join") {
    const roomId = cleanRoomId(body.roomId);
    const username = normalizeUsername(body.username);
    if (username.length < 2) return json({ error: "Use at least two letters for your creator name." }, 400);
    if (!await db.prepare("SELECT id FROM shared_rooms WHERE id = ?").bind(roomId).first()) return json({ error: "That shared room does not exist." }, 404);
    const existing = await db.prepare("SELECT id FROM shared_participants WHERE room_id = ? AND username = ?").bind(roomId, username).first<{ id: string }>();
    if (existing) return json({ error: "That creator name is already in this room. Use your original tab or choose another name." }, 409);
    const participantId = crypto.randomUUID();
    const sessionToken = randomToken();
    const now = new Date().toISOString();
    await db.prepare("INSERT INTO shared_participants (id, room_id, username, token_hash, accent, last_seen_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
      .bind(participantId, roomId, username, await hashToken(sessionToken), participantAccent(username), now, now).run();
    await db.prepare("UPDATE shared_invites SET accepted_at = ? WHERE room_id = ? AND username = ? AND accepted_at IS NULL").bind(now, roomId, username).run();
    return json({ session: { roomId, participantId, sessionToken, username }, snapshot: await snapshot(db, roomId, 0) });
  }

  const roomId = cleanRoomId(body.roomId);
  const participantId = cleanId(body.participantId);
  const sessionToken = typeof body.sessionToken === "string" ? body.sessionToken.slice(0, 96) : "";
  const participant = await authorize(db, roomId, participantId, sessionToken);
  if (!participant) return json({ error: "This room session has expired. Join again." }, 401);

  if (action === "invite") {
    const username = normalizeUsername(body.username);
    if (username.length < 2) return json({ error: "Enter your friend's username." }, 400);
    await db.prepare("INSERT INTO shared_invites (id, room_id, username, created_by, created_at, accepted_at) VALUES (?, ?, ?, ?, ?, NULL)")
      .bind(crypto.randomUUID(), roomId, username, participant.username, new Date().toISOString()).run();
    return json({ ok: true, username });
  }

  if (action === "append-operation") {
    const operation = validateSharedOperation(body.operation);
    if (!operation || operation.participantId !== participantId || operation.author !== participant.username) return json({ error: "Invalid shared drawing operation." }, 400);
    const count = await db.prepare("SELECT COUNT(*) AS value FROM shared_drawing_ops WHERE room_id = ?").bind(roomId).first<{ value: number }>();
    if (Number(count?.value ?? 0) >= 5000) return json({ error: "This room reached its drawing limit. Make it 3D, then start a new room." }, 409);
    const result = await db.prepare("INSERT OR IGNORE INTO shared_drawing_ops (id, room_id, participant_id, author, payload, created_at) VALUES (?, ?, ?, ?, ?, ?)")
      .bind(operation.id, roomId, participantId, participant.username, JSON.stringify(operation), operation.createdAt).run();
    const existing = await db.prepare("SELECT sequence FROM shared_drawing_ops WHERE id = ?").bind(operation.id).first<{ sequence: number }>();
    const sequence = Number(result.meta?.last_row_id ?? 0) || Number(existing?.sequence ?? 0);
    return json({ operation: { ...operation, sequence } satisfies SharedDrawingOperation });
  }

  return json({ error: "Unknown room action." }, 400);
}
