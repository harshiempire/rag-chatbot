import { useMutation, useQueryClient } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';
import type { ChatSession } from '../../../shared/types/chat';

export const useSaveSessionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (session: ChatSession) => chatHistoryStore.saveSession(session),
    onSuccess: (session) => {
      queryClient.setQueryData(queryKeys.chatSession(session.id), session);
      void queryClient.invalidateQueries({ queryKey: queryKeys.chatSessions });
    },
  });
};
