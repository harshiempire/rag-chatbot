import type { ChatSession, ChatSessionSummary } from '../types/chat';
import { API_BASE_URL } from '../api/config';
import { authorizedFetch } from '../api/httpClient';

export class ChatHistoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ChatHistoryError';
  }
}

export interface ChatHistoryStore {
  listSessions(userId: string): Promise<ChatSessionSummary[]>;
  getSession(userId: string, sessionId: string): Promise<ChatSession | null>;
  saveSession(userId: string, session: ChatSession): Promise<ChatSession>;
  deleteSession(userId: string, sessionId: string): Promise<void>;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) {
        detail = body.detail;
      }
    } catch {
      // Keep fallback detail.
    }
    throw new ChatHistoryError(detail);
  }
  return (await response.json()) as T;
}

export class ApiHistoryStore implements ChatHistoryStore {
  async listSessions(_userId: string): Promise<ChatSessionSummary[]> {
    const response = await authorizedFetch(`${API_BASE_URL}/chat/sessions/summary`, {
      method: 'GET',
    });
    return parseResponse<ChatSessionSummary[]>(response);
  }

  async getSession(_userId: string, sessionId: string): Promise<ChatSession | null> {
    const response = await authorizedFetch(`${API_BASE_URL}/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'GET',
    });
    if (response.status === 404) {
      return null;
    }
    return parseResponse<ChatSession>(response);
  }

  async saveSession(_userId: string, session: ChatSession): Promise<ChatSession> {
    const response = await authorizedFetch(`${API_BASE_URL}/chat/sessions/${encodeURIComponent(session.id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(session),
    });
    return parseResponse<ChatSession>(response);
  }

  async deleteSession(_userId: string, sessionId: string): Promise<void> {
    const response = await authorizedFetch(`${API_BASE_URL}/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE',
    });
    await parseResponse<{ message: string }>(response);
  }
}

export const chatHistoryStore: ChatHistoryStore = new ApiHistoryStore();
