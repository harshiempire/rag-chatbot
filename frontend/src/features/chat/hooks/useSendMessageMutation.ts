import { useMutation } from '@tanstack/react-query';
import type { ChatEvent, ChatSession, RAGStreamRequest } from '../../../shared/types/chat';
import { chatTransport } from '../../../shared/api/chatTransport';
import { useSaveSessionMutation } from './useSaveSessionMutation';

interface SendMessageParams {
  request: RAGStreamRequest;
  onEvent: (event: ChatEvent) => void;
  sessionSnapshot: () => ChatSession | null;
  signal?: AbortSignal;
}

export const useSendMessageMutation = () => {
  const saveSessionMutation = useSaveSessionMutation();

  return useMutation({
    mutationFn: async ({ request, onEvent, signal }: SendMessageParams) => {
      for await (const event of chatTransport.sendMessage(request, signal)) {
        onEvent(event);
      }
    },
    onSuccess: async (_, variables) => {
      const snapshot = variables.sessionSnapshot();
      if (snapshot) {
        await saveSessionMutation.mutateAsync(snapshot);
      }
    },
  });
};
