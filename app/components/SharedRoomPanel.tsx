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
  const joiningInvitation = Boolean(invitedRoom && (
    !session
    || session.roomId !== invitedRoom
    || (invitedUsername && session.username !== invitedUsername)
  ));

  const act = async (operation: () => Promise<unknown>) => {
    setBusy(true);
    setLocalMessage("");
    try { await operation(); } catch (error) { setLocalMessage(error instanceof Error ? error.message : "That did not work."); }
    finally { setBusy(false); }
  };

  const shareInvite = async () => {
    if (!inviteUrl) return;
    try {
      if (navigator.share) {
        await navigator.share({ title: "Draw with me in WallAlive", text: `Join my WallAlive room as @${friend}.`, url: inviteUrl });
        setLocalMessage("Invite shared.");
        return;
      }
      await navigator.clipboard.writeText(inviteUrl);
      setLocalMessage("Invite link copied.");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setLocalMessage("Copy the link below and send it to your friend.");
    }
  };

  if (!open) return null;
  return (
    <div className="shared-room-backdrop" role="dialog" aria-modal="true" aria-labelledby="shared-room-title">
      <section className="shared-room-panel">
        <header><div><span>{joiningInvitation ? "YOU'RE INVITED" : "CREATE TOGETHER"}</span><h2 id="shared-room-title">{joiningInvitation ? `Join room ${invitedRoom}.` : "One wall. Many imaginations."}</h2></div><button onClick={onClose} aria-label="Close shared room">×</button></header>
        {!session || joiningInvitation ? <div className="shared-room-entry">
          <div className="shared-room-illustration" aria-hidden="true"><i /><i /><i /><strong>✦</strong></div>
          <div className="shared-room-fields">
            {joiningInvitation ? <>
              <p className="invite-intro">Your friend has the wall ready. Pick your name, join, then open the wall.</p>
              <label><span>Your creator name</span><input value={username} onChange={(event) => setUsername(event.target.value)} maxLength={24} placeholder="mika" autoComplete="nickname" /></label>
              <label><span>Room code</span><input value={invitedRoom} readOnly /></label>
              <button className="room-primary" disabled={busy} onClick={() => act(() => onJoin(invitedRoom, username))}>JOIN THIS ROOM <i>↗</i></button>
              <div className="room-steps"><span>1 · Join</span><span>2 · Open wall</span><span>3 · Draw together</span></div>
            </> : <>
              <label><span>Your creator name</span><input value={username} onChange={(event) => setUsername(event.target.value)} maxLength={24} placeholder="mika" autoComplete="nickname" /></label>
              <button className="room-primary" disabled={busy} onClick={() => act(() => onCreate(username))}>START A ROOM <i>→</i></button>
              <div className="room-or"><span />OR JOIN<span /></div>
              <label><span>Room code</span><input value={roomId} onChange={(event) => setRoomId(event.target.value.toUpperCase())} maxLength={16} placeholder="MOON7PIP" /></label>
              <button disabled={busy} onClick={() => act(() => onJoin(roomId, username))}>JOIN FRIEND <i>↗</i></button>
            </>}
          </div>
        </div> : <>
          <div className="shared-room-live">
            <div><i className={status} /><span>ROOM {session.roomId}</span><button onClick={() => navigator.clipboard?.writeText(session.roomId)}>COPY CODE</button></div>
            <h3>{participants.length} creator{participants.length === 1 ? "" : "s"} on the wall</h3>
            <div className="room-people">{participants.map((person) => <div key={person.id}><i style={{ background: person.accent }}>{person.username.slice(0, 1).toUpperCase()}</i><span>@{person.username}</span></div>)}</div>
          </div>
          <div className="room-invite">
            <label><span>Invite by username</span><div><input value={friend} onChange={(event) => setFriend(event.target.value)} maxLength={24} placeholder="friend_name" /><button disabled={busy} onClick={() => act(async () => { const result = await onInvite(friend); setInviteUrl(result.inviteUrl); })}>MAKE INVITE</button></div></label>
            {inviteUrl ? <>
              <div className="invite-ready"><input aria-label="Invitation link" readOnly value={inviteUrl} onFocus={(event) => event.currentTarget.select()} /><button onClick={() => void shareInvite()}>SHARE INVITE</button></div>
              <div className="room-steps"><span>1 · Share link</span><span>2 · Friend taps join</span><span>3 · Open the wall</span></div>
            </> : null}
          </div>
          <button className="room-primary room-open-wall" onClick={onOpenWall}>OPEN OUR DRAWING WALL <i>✦</i></button>
          <button className="room-leave" onClick={onLeave}>Leave room</button>
        </>}
        <footer><i className={status} /><span>{localMessage || message}</span><small>Vector strokes sync. Camera frames and artwork pixels do not.</small></footer>
      </section>
    </div>
  );
}
