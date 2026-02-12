export const queryKeys = {
  chatSessions: ['chatSessions'] as const,
  chatSession: (sessionId: string | null) => ['chatSession', sessionId] as const,
  models: ['models'] as const,
  health: ['health'] as const,
};
