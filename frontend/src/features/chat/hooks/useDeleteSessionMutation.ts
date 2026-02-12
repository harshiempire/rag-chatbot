import { useMutation, useQueryClient } from '@tanstack/react-query';
import { chatHistoryStore } from '../../../shared/storage/chatHistoryStore';
import { queryKeys } from '../../../shared/api/queryKeys';

export const useDeleteSessionMutation = () => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (sessionId: string) => {
      await chatHistoryStore.deleteSession(sessionId);
      return sessionId;
    },
    onSuccess: (sessionId) => {
      queryClient.removeQueries({ queryKey: queryKeys.chatSession(sessionId), exact: true });
      void queryClient.invalidateQueries({ queryKey: queryKeys.chatSessions });
    },
  });
};
