export type ReflectionRevision = "new-ending" | "new-feeling" | "add-friend";

export type LearningReflection = {
  retell: string;
  nextChange: ReflectionRevision;
  savedAt: string;
};

export type LearningProgressInput = {
  story: {
    title: string;
    learningGoal: string;
    plannedBeats: number;
    completedBeats: number;
    status: "awaiting-human-approval" | "playing" | "complete" | "dismissed";
  } | null;
  reflection: LearningReflection | null;
  humanTurns: number;
  agentTurns: number;
  participantCount: number;
  sharedVectorOperations: number;
  worldInteractions: Record<string, number>;
  worldTotals: Record<string, number>;
};

export function buildLearningProgress(input: LearningProgressInput) {
  const completedWorlds = Object.entries(input.worldTotals)
    .filter(([world, total]) => total > 0 && (input.worldInteractions[world] ?? 0) >= total)
    .map(([world]) => world);
  const completedInteractions = Object.values(input.worldInteractions).reduce((sum, value) => sum + value, 0);
  const plannedBeats = input.story?.plannedBeats ?? 0;
  const completedBeats = Math.min(plannedBeats, Math.max(0, input.story?.completedBeats ?? 0));
  const phase = input.reflection
    ? "reflected"
    : completedBeats > 0 || completedWorlds.length > 0
      ? "performed-needs-reflection"
      : input.story
        ? "planned"
        : "not-started";

  return {
    phase,
    story: input.story ? {
      title: input.story.title,
      learningGoal: input.story.learningGoal,
      status: input.story.status,
      plannedBeats,
      completedBeats,
    } : null,
    observedEvidence: {
      completedWorlds,
      completedWorldInteractions: completedInteractions,
      humanTurns: input.humanTurns,
      agentTurns: input.agentTurns,
      participantCount: input.participantCount,
      sharedVectorOperations: input.sharedVectorOperations,
      childRetellRecorded: Boolean(input.reflection?.retell),
      revisionChoiceRecorded: Boolean(input.reflection?.nextChange),
    },
    reflection: input.reflection,
    suggestedNextScaffold: phase === "not-started"
      ? "Help the learner plan a short beginning, middle, and ending."
      : phase === "planned"
        ? "Wait for visible human approval, then perform the staged sequence."
        : phase === "performed-needs-reflection"
          ? "Invite the learner to retell what happened and choose one revision."
          : `Use the learner's “${input.reflection?.nextChange.replaceAll("-", " ")}” choice to stage one revised version.`,
    interpretationBoundary: "Observational learning evidence only—not a score, diagnosis, or measured learning gain.",
    cameraDataIncluded: false,
    artworkPixelsIncluded: false,
  };
}
