import type { ChatSession } from '../types/chat';

const STORAGE_KEY = 'rag_chat_v1';

interface ChatHistoryEnvelope {
  sessions: Record<string, ChatSession>;
}

export class ChatHistoryError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ChatHistoryError';
  }
}

export interface ChatHistoryStore {
  listSessions(): Promise<ChatSession[]>;
  getSession(sessionId: string): Promise<ChatSession | null>;
  saveSession(session: ChatSession): Promise<ChatSession>;
  deleteSession(sessionId: string): Promise<void>;
}

export class LocalStorageHistoryStore implements ChatHistoryStore {
  private readEnvelope(): ChatHistoryEnvelope {
    const raw = this.readStorage(STORAGE_KEY);
    if (!raw) {
      return { sessions: {} };
    }

    try {
      const parsed = JSON.parse(raw) as ChatHistoryEnvelope;
      return parsed?.sessions ? parsed : { sessions: {} };
    } catch {
      // Recover from malformed local data without crashing the UI.
      return { sessions: {} };
    }
  }

  private writeEnvelope(envelope: ChatHistoryEnvelope): void {
    this.writeStorage(STORAGE_KEY, JSON.stringify(envelope));
  }

  private readStorage(key: string): string | null {
    if (typeof localStorage === 'undefined') {
      throw new ChatHistoryError('Local storage is unavailable in this environment.');
    }

    try {
      return localStorage.getItem(key);
    } catch (error) {
      throw new ChatHistoryError(
        `Unable to read local chat history: ${error instanceof Error ? error.message : 'unknown error'}`
      );
    }
  }

  private writeStorage(key: string, value: string): void {
    if (typeof localStorage === 'undefined') {
      throw new ChatHistoryError('Local storage is unavailable in this environment.');
    }

    try {
      localStorage.setItem(key, value);
    } catch (error) {
      throw new ChatHistoryError(
        `Unable to save chat history locally: ${error instanceof Error ? error.message : 'unknown error'}`
      );
    }
  }

  async listSessions(): Promise<ChatSession[]> {
    const envelope = this.readEnvelope();
    return Object.values(envelope.sessions).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async getSession(sessionId: string): Promise<ChatSession | null> {
    const envelope = this.readEnvelope();
    return envelope.sessions[sessionId] ?? null;
  }

  async saveSession(session: ChatSession): Promise<ChatSession> {
    const envelope = this.readEnvelope();
    envelope.sessions[session.id] = session;
    this.writeEnvelope(envelope);
    return session;
  }

  async deleteSession(sessionId: string): Promise<void> {
    const envelope = this.readEnvelope();
    delete envelope.sessions[sessionId];
    this.writeEnvelope(envelope);
  }
}

export const chatHistoryStore: ChatHistoryStore = new LocalStorageHistoryStore();

// Future plug-and-play replacement:
// export const chatHistoryStore: ChatHistoryStore = new ApiHistoryStore(...)
