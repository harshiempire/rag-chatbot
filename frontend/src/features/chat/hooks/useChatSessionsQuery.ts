import { useQuery } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useChatSessionsQuery = () =>
  useQuery({
    queryKey: queryKeys.chatSessions,
    queryFn: () => chatHistoryStore.listSessions(),
  });
