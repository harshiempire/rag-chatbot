
import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import type {
  ChatEvent,
  ChatMessage,
  ChatSession,
  ChatSessionSummary,
  ChatUsage,
  LLMProvider,
  PipelineStatusItem,
  RAGStreamRequest,
} from '../../../shared/types/chat';
import { createId } from '../../../shared/utils/id';
import { useChatSessionQuery } from '../hooks/useChatSessionQuery';
import { useChatSessionsQuery } from '../hooks/useChatSessionsQuery';
import { useSaveSessionMutation } from '../hooks/useSaveSessionMutation';
import { useSendMessageMutation } from '../hooks/useSendMessageMutation';
import { useDeleteSessionMutation } from '../hooks/useDeleteSessionMutation';
import { RAG_DEFAULT_SOURCE_ID, RAG_DEFAULT_TOP_K } from '../../../shared/api/config';
import { useAuth } from '../../auth/context/AuthContext';

import { Sidebar } from './layout/Sidebar';
import { ChatArea } from './chat/ChatArea';
import { ChatInput } from './chat/ChatInput';
import { Menu } from 'lucide-react';
import { Button } from './ui/Button';

// --- Reducer & State Definitions ---

interface ChatState {
  session: ChatSession | null;
  streamingMessageId: string | null;
}

type ChatAction =
  | { type: 'set_session'; session: ChatSession }
  | { type: 'reset' }
  | { type: 'start_stream'; session: ChatSession; assistantMessageId: string }
  | { type: 'append_token'; messageId: string; token: string }
  | { type: 'add_status'; messageId: string; status: PipelineStatusItem }
  | { type: 'add_source'; messageId: string; source: NonNullable<ChatMessage['sources']>[number] }
  | { type: 'set_error'; messageId: string; error: string }
  | { type: 'set_usage'; messageId: string; usage: ChatUsage }
  | { type: 'apply_final'; messageId: string; answer: string }
  | { type: 'complete'; messageId: string };

const initialState: ChatState = {
  session: null,
  streamingMessageId: null,
};

function updateAssistantMessage(
  session: ChatSession,
  messageId: string,
  transform: (message: ChatMessage) => ChatMessage
): ChatSession {
  return {
    ...session,
    updatedAt: Date.now(),
    messages: session.messages.map((message) =>
      message.id === messageId && message.role === 'assistant' ? transform(message) : message
    ),
  };
}

function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'set_session':
      return { ...state, session: action.session, streamingMessageId: null };
    case 'reset':
      return initialState;
    case 'start_stream':
      return {
        session: action.session,
        streamingMessageId: action.assistantMessageId,
      };
    case 'append_token': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          content: `${message.content}${action.token}`,
        })),
      };
    }
    case 'add_status': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => {
          const history = message.statusHistory ?? [];
          const lastStatus = history[history.length - 1];
          // If the new status is for the same stage as the last one, update it instead of appending
          // specific check: if last was 'start' and new is 'done' for same stage?
          // actually generally if stages match, we likely want to update the entry with the latest state/label.
          if (lastStatus && lastStatus.stage === action.status.stage) {
            const newHistory = [...history];
            newHistory[history.length - 1] = action.status;
            return { ...message, statusHistory: newHistory };
          }
          return { ...message, statusHistory: [...history, action.status] };
        }),
      };
    }
    case 'add_source': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          sources: [...(message.sources ?? []), action.source],
        })),
      };
    }
    case 'set_error': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          error: action.error,
        })),
      };
    }
    case 'set_usage': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          usage: action.usage,
        })),
      };
    }
    case 'apply_final': {
      if (!state.session) return state;
      return {
        ...state,
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          content: action.answer.trim() ? action.answer : message.content,
        })),
      };
    }
    case 'complete': {
      if (!state.session) return state;
      return {
        streamingMessageId: null,
        session: updateAssistantMessage(state.session, action.messageId, (message) => {
          // Force the last status to be done if it exists, to ensure loader is removed
          let newHistory = message.statusHistory;
          if (newHistory && newHistory.length > 0) {
            const lastIdx = newHistory.length - 1;
            const lastStatus = newHistory[lastIdx];
            if (lastStatus.state !== 'done') {
              newHistory = [...newHistory];
              newHistory[lastIdx] = {
                ...lastStatus,
                state: 'done',
                // Update label to indicate completion if we are forcing it closed
                label: lastStatus.label === 'Generating answer' ? 'Answer generated' : lastStatus.label
              };
            }
          }
          return {
            ...message,
            done: true,
            statusHistory: newHistory
          };
        }),
      };
    }
    default:
      return state;
  }
}

function createSession(provider: LLMProvider): ChatSession {
  const now = Date.now();
  return {
    id: createId(),
    title: 'New Chat',
    llmProvider: provider,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function toSessionSummary(session: ChatSession): ChatSessionSummary {
  return {
    id: session.id,
    title: session.title,
    llmProvider: session.llmProvider,
    createdAt: session.createdAt,
    updatedAt: session.updatedAt,
  };
}

function formatError(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

const PROVIDER_PREFERENCE_KEY = 'rag_provider_pref_v1';

function isLlmProvider(value: string): value is LLMProvider {
  return value === 'local' || value === 'openai' || value === 'anthropic' || value === 'google' || value === 'openrouter';
}

function readProviderPreference(userId: string): LLMProvider | null {
  if (!userId || typeof localStorage === 'undefined') {
    return null;
  }
  try {
    const raw = localStorage.getItem(`${PROVIDER_PREFERENCE_KEY}:${userId}`);
    return raw && isLlmProvider(raw) ? raw : null;
  } catch {
    return null;
  }
}

function writeProviderPreference(userId: string, provider: LLMProvider): void {
  if (!userId || typeof localStorage === 'undefined') {
    return;
  }
  try {
    localStorage.setItem(`${PROVIDER_PREFERENCE_KEY}:${userId}`, provider);
  } catch {
    // Ignore storage failures.
  }
}

const FOLLOWUP_REFERENCE_PATTERN =
  /\b(same|that|this|these|those|it|they|them|under the same|same section|same rule)\b/i;
const EXPLICIT_LEGAL_REFERENCE_PATTERN =
  /(§\s*\d+[\w.-]*|\bpart\s+\d+[a-z]?\b|\btitle\s+\d+\b|\b\d+\s*cfr\b)/i;

function asTrimmedString(value: unknown): string | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }
  if (typeof value !== 'string') {
    return null;
  }
  const text = value.trim();
  return text.length > 0 ? text : null;
}

function parseRefFromText(text: string, key: string): string | null {
  const value = text.trim();
  if (!value) {
    return null;
  }
  if (key === 'section') {
    const match = value.match(/§\s*([\d.]+[a-z]?)/i);
    return match?.[1]?.toLowerCase() ?? null;
  }
  if (key === 'part') {
    const sectionMatch = value.match(/§\s*([\d]+)\./i);
    if (sectionMatch?.[1]) return sectionMatch[1].toLowerCase();
    const partMatch = value.match(/\bpart\s+(\d+[a-z]?)\b/i);
    return partMatch?.[1]?.toLowerCase() ?? null;
  }
  if (key === 'chapter') {
    const match = value.match(/\bchapter\s+([ivxlcdm]+|\d+)\b/i);
    return match?.[1]?.toUpperCase() ?? null;
  }
  if (key === 'title') {
    const match = value.match(/\btitle\s+(\d+)\b/i);
    return match?.[1] ?? null;
  }
  return null;
}

function getSourceReferenceValue(
  source: NonNullable<ChatMessage['sources']>[number],
  key: string
): string | null {
  const metadata = source.metadata as Record<string, unknown> | undefined;
  if (!metadata || typeof metadata !== 'object') {
    return null;
  }
  const chunkMetadata = metadata.chunk_metadata as Record<string, unknown> | undefined;
  const docMetadata = metadata.doc_metadata as Record<string, unknown> | undefined;
  const directValue =
    asTrimmedString(chunkMetadata?.[key]) ??
    asTrimmedString(docMetadata?.[key]) ??
    asTrimmedString(metadata[key]);
  if (directValue) {
    return directValue;
  }

  const fromTitle = parseRefFromText(source.title ?? '', key);
  if (fromTitle) {
    return fromTitle;
  }
  return parseRefFromText(source.snippet ?? '', key);
}

function topKeyByCount(counter: Map<string, number>): string | null {
  let winner: string | null = null;
  let maxCount = 0;
  for (const [key, count] of counter.entries()) {
    if (count > maxCount) {
      winner = key;
      maxCount = count;
    }
  }
  return winner;
}

function inferFollowupMetadataFilters(
  question: string,
  messages: ChatMessage[]
): Record<string, string | string[]> | undefined {
  if (EXPLICIT_LEGAL_REFERENCE_PATTERN.test(question)) {
    return undefined;
  }
  if (!FOLLOWUP_REFERENCE_PATTERN.test(question)) {
    return undefined;
  }

  const latestAssistantWithSources = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && (message.sources?.length ?? 0) > 0);
  if (!latestAssistantWithSources?.sources?.length) {
    return undefined;
  }

  const partCounts = new Map<string, number>();
  const sectionCounts = new Map<string, number>();
  const chapterCounts = new Map<string, number>();
  const titleCounts = new Map<string, number>();

  for (const source of latestAssistantWithSources.sources) {
    const part = getSourceReferenceValue(source, 'part');
    const section = getSourceReferenceValue(source, 'section');
    const chapter = getSourceReferenceValue(source, 'chapter');
    const title = getSourceReferenceValue(source, 'title');
    if (part) partCounts.set(part, (partCounts.get(part) ?? 0) + 1);
    if (section) sectionCounts.set(section, (sectionCounts.get(section) ?? 0) + 1);
    if (chapter) chapterCounts.set(chapter, (chapterCounts.get(chapter) ?? 0) + 1);
    if (title) titleCounts.set(title, (titleCounts.get(title) ?? 0) + 1);
  }

  const inferred: Record<string, string | string[]> = {};
  const section = topKeyByCount(sectionCounts);
  const part = topKeyByCount(partCounts);
  const chapter = topKeyByCount(chapterCounts);
  const title = topKeyByCount(titleCounts);

  if (section) inferred.section = section;
  if (part) inferred.part = part;
  if (chapter) inferred.chapter = chapter;
  if (title) inferred.title = title;

  return Object.keys(inferred).length > 0 ? inferred : undefined;
}

export function ChatPage() {
  const { user, logout } = useAuth();
  const userId = user?.id ?? '';
  const sessionsQuery = useChatSessionsQuery(userId);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [provider, setProvider] = useState<LLMProvider>('local');
  const [_composerError, setComposerError] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isNewChatMode, setIsNewChatMode] = useState(false);

  const persistedSessionIds = useMemo(
    () => new Set((sessionsQuery.data ?? []).map((session) => session.id)),
    [sessionsQuery.data],
  );
  const shouldFetchSelectedSession = Boolean(
    selectedSessionId && persistedSessionIds.has(selectedSessionId),
  );
  const sessionQuery = useChatSessionQuery(
    userId,
    selectedSessionId,
    shouldFetchSelectedSession,
  );
  const saveSessionMutation = useSaveSessionMutation(userId);
  const sendMessageMutation = useSendMessageMutation();
  const deleteSessionMutation = useDeleteSessionMutation(userId);

  const [state, dispatch] = useReducer(chatReducer, initialState);
  const stateRef = useRef(state);
  const abortControllerRef = useRef<AbortController | null>(null);
  const cancelRequestedRef = useRef(false);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    const preferred = readProviderPreference(userId);
    if (preferred) {
      setProvider(preferred);
    }
  }, [userId]);

  useEffect(() => {
    if (isNewChatMode) {
      return;
    }
    if (!selectedSessionId && sessionsQuery.data?.length) {
      setSelectedSessionId(sessionsQuery.data[0].id);
    }
  }, [selectedSessionId, sessionsQuery.data, isNewChatMode]);

  useEffect(() => {
    if (!sessionQuery.data || state.streamingMessageId) {
      return;
    }

    const currentSession = stateRef.current.session;
    if (
      currentSession &&
      currentSession.id === sessionQuery.data.id &&
      currentSession.updatedAt >= sessionQuery.data.updatedAt
    ) {
      return;
    }

    dispatch({ type: 'set_session', session: sessionQuery.data });
    setProvider(sessionQuery.data.llmProvider);
    setIsNewChatMode(false);
  }, [sessionQuery.data, state.streamingMessageId]);

  const sessions = useMemo(() => {
    const fromQuery = sessionsQuery.data ?? [];
    const local = state.session;
    if (!local) {
      return fromQuery;
    }

    const localSummary = toSessionSummary(local);
    const merged = fromQuery.some((item) => item.id === localSummary.id)
      ? fromQuery.map((item) => {
          if (item.id !== localSummary.id) {
            return item;
          }
          return item.updatedAt >= localSummary.updatedAt ? item : localSummary;
        })
      : [localSummary, ...fromQuery];

    return merged.sort((a, b) => b.updatedAt - a.updatedAt);
  }, [sessionsQuery.data, state.session]);

  const isStreaming = Boolean(state.streamingMessageId) || sendMessageMutation.isPending;

  const persistSession = async (session: ChatSession) => {
    try {
      await saveSessionMutation.mutateAsync(session);
    } catch (error) {
      console.error("Failed to save session", error);
      setComposerError("Failed to save session");
    }
  };

  const onSend = async (text: string) => {
    if (isStreaming) {
      return;
    }

    setComposerError(null);

    const startingSession = stateRef.current.session ?? createSession(provider);

    const userMessage: ChatMessage = {
      id: createId(),
      role: 'user',
      content: text,
      createdAt: Date.now(),
      done: true,
    };

    const assistantMessageId = createId();
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      createdAt: Date.now(),
      done: false,
      statusHistory: [],
      sources: [],
    };

    const nextSession: ChatSession = {
      ...startingSession,
      llmProvider: provider,
      title:
        startingSession.messages.length === 0
          ? text.slice(0, 60).trim() || 'New Chat'
          : startingSession.title,
      updatedAt: Date.now(),
      messages: [...startingSession.messages, userMessage, assistantMessage],
    };

    setSelectedSessionId(nextSession.id);
    setIsNewChatMode(false);
    dispatch({ type: 'start_stream', session: nextSession, assistantMessageId });
    let streamState: ChatState = {
      session: nextSession,
      streamingMessageId: assistantMessageId,
    };
    const dispatchStreamAction = (action: ChatAction) => {
      streamState = chatReducer(streamState, action);
      dispatch(action);
    };

    const request: RAGStreamRequest = {
      question: text,
      llm_provider: provider,
      classification_filter: ['public'],
      source_id: RAG_DEFAULT_SOURCE_ID,
      top_k: RAG_DEFAULT_TOP_K,
      temperature: 0.7,
      min_similarity: 0.2,
      session_id: nextSession.id,
      chat_history: startingSession.messages
        .filter((message) => message.role === 'user' || message.role === 'assistant')
        .map((message) => ({ role: message.role, content: message.content }))
        .slice(-12),
      conversation_context: startingSession.messages
        .filter((message) => message.role === 'user')
        .map((message) => message.content.trim())
        .filter(Boolean)
        .slice(-2),
      metadata_filters: inferFollowupMetadataFilters(text, startingSession.messages),
      retrieval_mode: 'hybrid',
    };

    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    cancelRequestedRef.current = false;
    let receivedDoneEvent = false;

    try {
      await sendMessageMutation.mutateAsync({
        request,
        signal: abortController.signal,
        onEvent: (event: ChatEvent) => {
          if (event.type === 'status') {
            dispatchStreamAction({
              type: 'add_status',
              messageId: assistantMessageId,
              status: {
                ...event.data,
                at: Date.now(),
              },
            });
            return;
          }

          if (event.type === 'token') {
            dispatchStreamAction({ type: 'append_token', messageId: assistantMessageId, token: event.data.text });
            return;
          }

          if (event.type === 'source') {
            dispatchStreamAction({ type: 'add_source', messageId: assistantMessageId, source: event.data });
            return;
          }

          if (event.type === 'final') {
            dispatchStreamAction({ type: 'apply_final', messageId: assistantMessageId, answer: event.data.answer });
            dispatchStreamAction({
              type: 'set_usage',
              messageId: assistantMessageId,
              usage: {
                timingsMs: event.data.timings_ms,
                retrievedCount: event.data.retrieved_count,
                promptContextCount: event.data.prompt_context_count,
              },
            });
            return;
          }

          if (event.type === 'error') {
            dispatchStreamAction({ type: 'set_error', messageId: assistantMessageId, error: event.data.message });
            return;
          }

          if (event.type === 'done') {
            receivedDoneEvent = true;
            dispatchStreamAction({ type: 'complete', messageId: assistantMessageId });
          }
        },
      });

      if (!receivedDoneEvent && !cancelRequestedRef.current) {
        dispatchStreamAction({
          type: 'set_error',
          messageId: assistantMessageId,
          error: 'The stream ended unexpectedly before completion.',
        });
        dispatchStreamAction({ type: 'complete', messageId: assistantMessageId });
      }
    } catch (error) {
      if (!cancelRequestedRef.current && !isAbortError(error)) {
        const message = formatError(error, 'Unable to get a response from the RAG service.');
        dispatchStreamAction({ type: 'set_error', messageId: assistantMessageId, error: message });
        dispatchStreamAction({ type: 'complete', messageId: assistantMessageId });
        setComposerError(message);
      }
    } finally {
      const wasCancelled = cancelRequestedRef.current;
      if (wasCancelled) {
        streamState = chatReducer(streamState, {
          type: 'set_error',
          messageId: assistantMessageId,
          error: 'Generation canceled by user.',
        });
        streamState = chatReducer(streamState, { type: 'complete', messageId: assistantMessageId });
      }

      abortControllerRef.current = null;
      cancelRequestedRef.current = false;

      if (streamState.session) {
        await persistSession(streamState.session);
      }
    }
  };

  const onCancel = () => {
    if (!state.streamingMessageId) {
      return;
    }

    cancelRequestedRef.current = true;
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    dispatch({
      type: 'set_error',
      messageId: state.streamingMessageId,
      error: 'Generation canceled by user.',
    });
    dispatch({ type: 'complete', messageId: state.streamingMessageId });
  };

  const onNewChat = () => {
    const preferred = readProviderPreference(userId);
    if (preferred) {
      setProvider(preferred);
    }
    setSelectedSessionId(null);
    setIsNewChatMode(true);
    dispatch({ type: 'reset' });
    setComposerError(null);
  };

  const onDeleteSession = async (sessionId: string) => {
    if (deleteSessionMutation.isPending) {
      return;
    }

    try {
      await deleteSessionMutation.mutateAsync(sessionId);
      const remaining = sessions.filter((session) => session.id !== sessionId);
      const nextSession = remaining[0] ?? null;

      if (selectedSessionId === sessionId) {
        setSelectedSessionId(nextSession?.id ?? null);
        if (!nextSession) {
          dispatch({ type: 'reset' });
          setIsNewChatMode(true);
        } else {
          setIsNewChatMode(false);
        }
      }
    } catch (error) {
      console.error("Failed to delete session", error);
      setComposerError(formatError(error, 'Unable to delete this chat.'));
    }
  };

  if (!user) {
    return null;
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <div className={`fixed inset-y-0 left-0 z-50 transform ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:relative md:translate-x-0 transition duration-200 ease-in-out`}>
        <Sidebar
          sessions={sessions}
          selectedSessionId={selectedSessionId}
          userEmail={user.email}
          onLogout={() => {
            void logout();
          }}
          onSelectSession={(id) => {
            setSelectedSessionId(id);
            setIsNewChatMode(false);
            setIsSidebarOpen(false);
          }}
          onDeleteSession={onDeleteSession}
          onNewChat={() => {
            onNewChat();
            setIsSidebarOpen(false);
          }}
          isLoading={sessionsQuery.isLoading}
        />
      </div>

      <div className="flex-1 flex flex-col min-w-0 relative">
        <header className="flex items-center justify-between p-4 border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setIsSidebarOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </Button>
            <div className="flex flex-col">
              <span className="font-semibold text-lg text-slate-100">{state.session?.title || "New Chat"}</span>
              <div className="flex items-center gap-2 mt-1">
                <select
                  value={provider}
                  onChange={(e) => {
                    const nextProvider = e.target.value as LLMProvider;
                    setProvider(nextProvider);
                    writeProviderPreference(userId, nextProvider);
                  }}
                  className="bg-slate-900/50 border border-slate-800 text-xs text-slate-400 rounded-md px-2 py-1 outline-none focus:border-slate-600 focus:text-slate-200 transition-colors cursor-pointer"
                >
                  <option value="local">Local Model</option>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google Gemini</option>
                  <option value="openrouter">OpenRouter</option>
                </select>
              </div>
            </div>
          </div>
        </header>

        <ChatArea
          messages={state.session?.messages ?? []}
          isStreaming={isStreaming}
          streamingMessageId={state.streamingMessageId}
        />

        <div className="p-4 bg-gradient-to-t from-slate-950 to-transparent">
          <ChatInput
            onSend={onSend}
            onStop={onCancel}
            isLoading={isStreaming}
          />
        </div>
      </div>
    </div>
  );
}
