/** Production D1 schema mirrored by app/api/rooms/route.ts for first-run safety. */
export const sharedRoomSchema = {
  tables: ["shared_rooms", "shared_participants", "shared_invites", "shared_drawing_ops"],
  version: 1,
} as const;
