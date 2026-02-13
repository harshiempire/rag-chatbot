import { useQuery } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useChatSessionQuery = (
  userId: string,
  sessionId: string | null,
  shouldFetch: boolean = true,
) =>
  useQuery({
    queryKey: queryKeys.chatSession(userId, sessionId),
    queryFn: () => (sessionId ? chatHistoryStore.getSession(userId, sessionId) : Promise.resolve(null)),
    enabled: Boolean(userId && sessionId && shouldFetch),
  });
