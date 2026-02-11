import { useQuery } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useChatSessionQuery = (sessionId: string | null) =>
  useQuery({
    queryKey: queryKeys.chatSession(sessionId),
    queryFn: () => (sessionId ? chatHistoryStore.getSession(sessionId) : Promise.resolve(null)),
    enabled: Boolean(sessionId),
  });
