CREATE TABLE IF NOT EXISTS shared_rooms (id TEXT PRIMARY KEY, title TEXT NOT NULL, owner_username TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS shared_participants (id TEXT PRIMARY KEY, room_id TEXT NOT NULL, username TEXT NOT NULL, token_hash TEXT NOT NULL, accent TEXT NOT NULL, last_seen_at TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(room_id, username));
CREATE TABLE IF NOT EXISTS shared_invites (id TEXT PRIMARY KEY, room_id TEXT NOT NULL, username TEXT NOT NULL, created_by TEXT NOT NULL, created_at TEXT NOT NULL, accepted_at TEXT);
CREATE TABLE IF NOT EXISTS shared_drawing_ops (sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, room_id TEXT NOT NULL, participant_id TEXT NOT NULL, author TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_shared_ops_room_sequence ON shared_drawing_ops(room_id, sequence);
CREATE INDEX IF NOT EXISTS idx_shared_participants_room ON shared_participants(room_id, last_seen_at);
