import { useMutation } from '@tanstack/react-query';
import type { ChatEvent, RAGStreamRequest } from '../../../shared/types/chat';
import { chatTransport } from '../../../shared/api/chatTransport';

interface SendMessageParams {
  request: RAGStreamRequest;
  onEvent: (event: ChatEvent) => void;
  signal?: AbortSignal;
}

export const useSendMessageMutation = () => {
  return useMutation({
    mutationFn: async ({ request, onEvent, signal }: SendMessageParams) => {
      for await (const event of chatTransport.sendMessage(request, signal)) {
        onEvent(event);
      }
    },
  });
};
