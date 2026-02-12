import { API_BASE_URL } from './config';
import { parseSSEJson } from './sse';
import { parseChatEvent } from './chatEventSchema';
import { authorizedFetch } from './httpClient';
import type { ChatEvent, RAGStreamRequest } from '../types/chat';

export interface ChatTransport {
  sendMessage(payload: RAGStreamRequest, signal?: AbortSignal): AsyncGenerator<ChatEvent>;
}

export class SseChatTransport implements ChatTransport {
  async *sendMessage(payload: RAGStreamRequest, signal?: AbortSignal): AsyncGenerator<ChatEvent> {
    const response = await authorizedFetch(`${API_BASE_URL}/rag/query/stream/events`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    });

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const body = await response.json();
        detail = body?.detail ?? detail;
      } catch {
        // Keep fallback detail.
      }

      throw new Error(detail);
    }

    for await (const event of parseSSEJson(response, parseChatEvent, signal)) {
      yield event;
    }
  }
}

export const chatTransport: ChatTransport = new SseChatTransport();
