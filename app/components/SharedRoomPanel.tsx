"use client";

import { useState } from "react";
import type { SharedParticipant, SharedRoomSession } from "../lib/collaboration";

type SharedRoomPanelProps = {
  open: boolean;
  session: SharedRoomSession | null;
  participants: SharedParticipant[];
  status: "solo" | "connecting" | "live" | "error";
  message: string;
  invitedRoom?: string;
  invitedUsername?: string;
  onClose: () => void;
  onCreate: (username: string) => Promise<unknown>;
  onJoin: (roomId: string, username: string) => Promise<unknown>;
  onInvite: (username: string) => Promise<{ friend: string; inviteUrl: string }>;
  onLeave: () => void;
  onOpenWall: () => void;
};

export function SharedRoomPanel({ open, session, participants, status, message, invitedRoom = "", invitedUsername = "", onClose, onCreate, onJoin, onInvite, onLeave, onOpenWall }: SharedRoomPanelProps) {
  const [username, setUsername] = useState(invitedUsername);
  const [roomId, setRoomId] = useState(invitedRoom);
  const [friend, setFriend] = useState("");
  const [inviteUrl, setInviteUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [localMessage, setLocalMessage] = useState("");

  const act = async (operation: () => Promise<unknown>) => {
    setBusy(true);
    setLocalMessage("");
    try { await operation(); } catch (error) { setLocalMessage(error instanceof Error ? error.message : "That did not work."); }
    finally { setBusy(false); }
  };

  if (!open) return null;
  return (
    <div className="shared-room-backdrop" role="dialog" aria-modal="true" aria-labelledby="shared-room-title">
      <section className="shared-room-panel">
        <header><div><span>CREATE TOGETHER</span><h2 id="shared-room-title">One wall. Many imaginations.</h2></div><button onClick={onClose} aria-label="Close shared room">×</button></header>
        {!session ? <div className="shared-room-entry">
          <div className="shared-room-illustration" aria-hidden="true"><i /><i /><i /><strong>✦</strong></div>
          <div className="shared-room-fields">
            <label><span>Your creator name</span><input value={username} onChange={(event) => setUsername(event.target.value)} maxLength={24} placeholder="mika" autoComplete="nickname" /></label>
            <button className="room-primary" disabled={busy} onClick={() => act(() => onCreate(username))}>START A ROOM <i>→</i></button>
            <div className="room-or"><span />OR JOIN<span /></div>
            <label><span>Room code</span><input value={roomId} onChange={(event) => setRoomId(event.target.value.toUpperCase())} maxLength={16} placeholder="MOON7PIP" /></label>
            <button disabled={busy} onClick={() => act(() => onJoin(roomId, username))}>JOIN FRIEND <i>↗</i></button>
          </div>
        </div> : <>
          <div className="shared-room-live">
            <div><i className={status} /><span>ROOM {session.roomId}</span><button onClick={() => navigator.clipboard?.writeText(session.roomId)}>COPY CODE</button></div>
            <h3>{participants.length} creator{participants.length === 1 ? "" : "s"} on the wall</h3>
            <div className="room-people">{participants.map((person) => <div key={person.id}><i style={{ background: person.accent }}>{person.username.slice(0, 1).toUpperCase()}</i><span>@{person.username}</span></div>)}</div>
          </div>
          <div className="room-invite">
            <label><span>Invite by username</span><div><input value={friend} onChange={(event) => setFriend(event.target.value)} maxLength={24} placeholder="friend_name" /><button disabled={busy} onClick={() => act(async () => { const result = await onInvite(friend); setInviteUrl(result.inviteUrl); })}>MAKE INVITE</button></div></label>
            {inviteUrl ? <div className="invite-ready"><span>Invite ready for @{friend}</span><button onClick={() => navigator.clipboard?.writeText(inviteUrl)}>COPY LINK</button></div> : null}
          </div>
          <button className="room-primary room-open-wall" onClick={onOpenWall}>OPEN OUR DRAWING WALL <i>✦</i></button>
          <button className="room-leave" onClick={onLeave}>Leave room</button>
        </>}
        <footer><i className={status} /><span>{localMessage || message}</span><small>Vector strokes sync. Camera frames and artwork pixels do not.</small></footer>
      </section>
    </div>
  );
}
