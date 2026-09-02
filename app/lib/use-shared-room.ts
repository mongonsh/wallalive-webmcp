"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  normalizeUsername,
  sharedRoomInviteUrl,
  type SharedDrawingOperation,
  type SharedParticipant,
  type SharedRoomSession,
  type SharedRoomSnapshot,
} from "./collaboration";

const STORAGE_KEY = "wallalive-shared-room-v1";
type RoomStatus = "solo" | "connecting" | "live" | "error";

async function roomRequest<T>(body: Record<string, unknown>): Promise<T> {
  const response = await fetch("/api/rooms", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error || "The shared room could not be updated.");
  return result;
}

export function useSharedRoom() {
  const [session, setSession] = useState<SharedRoomSession | null>(null);
  const [participants, setParticipants] = useState<SharedParticipant[]>([]);
  const [operations, setOperations] = useState<SharedDrawingOperation[]>([]);
  const [status, setStatus] = useState<RoomStatus>("solo");
  const [message, setMessage] = useState("Create a room and invite a friend by username.");
  const latestSequenceRef = useRef(0);
  const sessionRef = useRef<SharedRoomSession | null>(null);

  const applySnapshot = useCallback((snapshot: SharedRoomSnapshot, replace = false) => {
    setParticipants(snapshot.participants);
    latestSequenceRef.current = Math.max(latestSequenceRef.current, snapshot.latestSequence);
    setOperations((current) => {
      const base = replace ? [] : current;
      const known = new Set(base.map((operation) => operation.id));
      return [...base, ...snapshot.operations.filter((operation) => !known.has(operation.id))]
        .sort((left, right) => (left.sequence ?? 0) - (right.sequence ?? 0))
        .slice(-1200);
    });
  }, []);

  const persistSession = useCallback((next: SharedRoomSession | null) => {
    sessionRef.current = next;
    setSession(next);
    if (next) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    else localStorage.removeItem(STORAGE_KEY);
  }, []);

  const refresh = useCallback(async (replace = false) => {
    const current = sessionRef.current;
    if (!current) return;
    const params = new URLSearchParams({ roomId: current.roomId, since: replace ? "0" : String(latestSequenceRef.current) });
    const response = await fetch(`/api/rooms?${params}`);
    const snapshot = await response.json() as SharedRoomSnapshot & { error?: string };
    if (!response.ok) throw new Error(snapshot.error || "Shared room sync failed.");
    applySnapshot(snapshot, replace);
    setStatus("live");
  }, [applySnapshot]);

  const createRoom = useCallback(async (usernameValue: string) => {
    const username = normalizeUsername(usernameValue);
    if (username.length < 2) throw new Error("Use at least two letters for your creator name.");
    setStatus("connecting");
    try {
      const result = await roomRequest<{ session: SharedRoomSession; snapshot: SharedRoomSnapshot }>({ action: "create", username });
      latestSequenceRef.current = 0;
      persistSession(result.session);
      applySnapshot(result.snapshot, true);
      setStatus("live");
      setMessage(`${username}'s room is live.`);
      return result.session;
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "The room could not be created.");
      throw error;
    }
  }, [applySnapshot, persistSession]);

  const joinRoom = useCallback(async (roomIdValue: string, usernameValue: string) => {
    const roomId = roomIdValue.trim().toUpperCase().slice(0, 16);
    const username = normalizeUsername(usernameValue);
    if (!roomId || username.length < 2) throw new Error("Add the room code and your creator name.");
    setStatus("connecting");
    try {
      const result = await roomRequest<{ session: SharedRoomSession; snapshot: SharedRoomSnapshot }>({ action: "join", roomId, username });
      latestSequenceRef.current = 0;
      persistSession(result.session);
      applySnapshot(result.snapshot, true);
      setStatus("live");
      setMessage(`${username} joined room ${roomId}.`);
      return result.session;
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "The room could not be joined.");
      throw error;
    }
  }, [applySnapshot, persistSession]);

  const leaveRoom = useCallback(() => {
    persistSession(null);
    latestSequenceRef.current = 0;
    setParticipants([]);
    setOperations([]);
    setStatus("solo");
    setMessage("Left the shared room. Your local drawing stays in this tab.");
  }, [persistSession]);

  const appendOperation = useCallback(async (operation: SharedDrawingOperation) => {
    const current = sessionRef.current;
    if (!current) return;
    setOperations((existing) => existing.some((item) => item.id === operation.id) ? existing : [...existing, operation]);
    try {
      const result = await roomRequest<{ operation: SharedDrawingOperation }>({
        action: "append-operation", roomId: current.roomId, participantId: current.participantId,
        sessionToken: current.sessionToken, operation,
      });
      if (result.operation.sequence) latestSequenceRef.current = Math.max(latestSequenceRef.current, result.operation.sequence);
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "A drawing stroke could not be shared.");
    }
  }, []);

  const prepareInvite = useCallback(async (friendValue: string) => {
    const current = sessionRef.current;
    const friend = normalizeUsername(friendValue);
    if (!current) throw new Error("Create or join a room first.");
    if (friend.length < 2) throw new Error("Enter your friend's username.");
    await roomRequest({ action: "invite", roomId: current.roomId, participantId: current.participantId, sessionToken: current.sessionToken, username: friend });
    const inviteUrl = sharedRoomInviteUrl(window.location.origin + window.location.pathname, current.roomId, friend);
    setMessage(`Invite ready for @${friend}. You choose how to share it.`);
    return { friend, inviteUrl };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      let stored: SharedRoomSession | null = null;
      try { stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null") as SharedRoomSession | null; } catch { stored = null; }
      if (stored?.roomId && stored.sessionToken) {
        persistSession(stored);
        setStatus("connecting");
        refresh(true).catch(() => leaveRoom());
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [leaveRoom, persistSession, refresh]);

  useEffect(() => {
    if (!session) return;
    const timer = window.setInterval(() => refresh(false).catch((error) => {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Room sync paused.");
    }), 900);
    return () => window.clearInterval(timer);
  }, [refresh, session]);

  return { session, participants, operations, status, message, createRoom, joinRoom, leaveRoom, appendOperation, prepareInvite, refresh };
}
