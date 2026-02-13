export const queryKeys = {
  chatSessions: (userId: string) => ['chatSessions', userId] as const,
  chatSession: (userId: string, sessionId: string | null) => ['chatSession', userId, sessionId] as const,
  models: ['models'] as const,
  health: ['health'] as const,
};
