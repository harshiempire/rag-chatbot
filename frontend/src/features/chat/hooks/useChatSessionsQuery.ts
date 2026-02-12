import { useQuery } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useChatSessionsQuery = (userId: string) =>
  useQuery({
    queryKey: queryKeys.chatSessions(userId),
    queryFn: () => chatHistoryStore.listSessions(userId),
    enabled: Boolean(userId),
  });
