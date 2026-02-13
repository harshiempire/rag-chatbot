import { useMutation, useQueryClient } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';
import type { ChatSession } from '../../../shared/types/chat';

export const useSaveSessionMutation = (userId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (session: ChatSession) => chatHistoryStore.saveSession(userId, session),
    onSuccess: (session) => {
      queryClient.setQueryData(queryKeys.chatSession(userId, session.id), session);
      void queryClient.invalidateQueries({ queryKey: queryKeys.chatSessions(userId) });
    },
  });
};
