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
  listSessions(userId: string): Promise<ChatSession[]>;
  getSession(userId: string, sessionId: string): Promise<ChatSession | null>;
  saveSession(userId: string, session: ChatSession): Promise<ChatSession>;
  deleteSession(userId: string, sessionId: string): Promise<void>;
}

export class LocalStorageHistoryStore implements ChatHistoryStore {
  private toScopedKey(userId: string): string {
    const normalized = userId.trim().toLowerCase();
    return `${STORAGE_KEY}:${normalized}`;
  }

  private readEnvelope(userId: string): ChatHistoryEnvelope {
    const raw = this.readStorage(this.toScopedKey(userId));
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

  private writeEnvelope(userId: string, envelope: ChatHistoryEnvelope): void {
    this.writeStorage(this.toScopedKey(userId), JSON.stringify(envelope));
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

  async listSessions(userId: string): Promise<ChatSession[]> {
    const envelope = this.readEnvelope(userId);
    return Object.values(envelope.sessions).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  async getSession(userId: string, sessionId: string): Promise<ChatSession | null> {
    const envelope = this.readEnvelope(userId);
    return envelope.sessions[sessionId] ?? null;
  }

  async saveSession(userId: string, session: ChatSession): Promise<ChatSession> {
    const envelope = this.readEnvelope(userId);
    envelope.sessions[session.id] = session;
    this.writeEnvelope(userId, envelope);
    return session;
  }

  async deleteSession(userId: string, sessionId: string): Promise<void> {
    const envelope = this.readEnvelope(userId);
    delete envelope.sessions[sessionId];
    this.writeEnvelope(userId, envelope);
  }
}

export const chatHistoryStore: ChatHistoryStore = new LocalStorageHistoryStore();

// Future plug-and-play replacement:
// export const chatHistoryStore: ChatHistoryStore = new ApiHistoryStore(...)
