import { useMutation, useQueryClient } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useDeleteSessionMutation = (userId: string) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      await chatHistoryStore.deleteSession(userId, sessionId);
      return sessionId;
    },
    onSuccess: (sessionId) => {
      queryClient.removeQueries({ queryKey: queryKeys.chatSession(userId, sessionId), exact: true });
      void queryClient.invalidateQueries({ queryKey: queryKeys.chatSessions(userId) });
    },
  });
};
