import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import clsx from 'clsx';
import type {
  ChatEvent,
  ChatMessage,
  ChatSession,
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
import { MessageComposer } from './MessageComposer';
import { MessageList } from './MessageList';
import { RAG_DEFAULT_SOURCE_ID, RAG_DEFAULT_TOP_K } from '../../../shared/api/config';

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
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          statusHistory: [...(message.statusHistory ?? []), action.status],
        })),
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
        session: updateAssistantMessage(state.session, action.messageId, (message) => ({
          ...message,
          done: true,
        })),
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

function formatError(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

function formatSessionTime(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();

  if (isToday) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }

  return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function getSessionPreview(session: ChatSession): string {
  const latestMessage = [...session.messages].reverse().find((message) => message.content.trim());
  if (!latestMessage) {
    return 'No messages yet';
  }

  return latestMessage.content.slice(0, 86).trim();
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function getPipelineStatusLabel(state: ChatState): string {
  if (!state.streamingMessageId || !state.session) {
    return 'Ready';
  }

  const activeMessage = state.session.messages.find((message) => message.id === state.streamingMessageId);
  if (!activeMessage) {
    return 'Generating response...';
  }

  const latestStatus = activeMessage.statusHistory?.at(-1);
  if (!latestStatus) {
    return 'Retrieving context...';
  }

  if (latestStatus.state === 'done') {
    return `${latestStatus.label} complete`;
  }

  return latestStatus.label;
}

export function ChatPage() {
  const sessionsQuery = useChatSessionsQuery();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [provider, setProvider] = useState<LLMProvider>('local');
  const [composerError, setComposerError] = useState<string | null>(null);

  const sessionQuery = useChatSessionQuery(selectedSessionId);
  const saveSessionMutation = useSaveSessionMutation();
  const sendMessageMutation = useSendMessageMutation();
  const deleteSessionMutation = useDeleteSessionMutation();

  const [state, dispatch] = useReducer(chatReducer, initialState);
  const stateRef = useRef(state);
  const abortControllerRef = useRef<AbortController | null>(null);
  const cancelRequestedRef = useRef(false);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    if (!selectedSessionId && sessionsQuery.data?.length) {
      setSelectedSessionId(sessionsQuery.data[0].id);
    }
  }, [selectedSessionId, sessionsQuery.data]);

  useEffect(() => {
    if (!sessionQuery.data || state.streamingMessageId) {
      return;
    }

    dispatch({ type: 'set_session', session: sessionQuery.data });
    setProvider(sessionQuery.data.llmProvider);
  }, [sessionQuery.data, state.streamingMessageId]);

  const sessions = useMemo(() => {
    const fromQuery = sessionsQuery.data ?? [];
    const local = state.session;
    if (!local) {
      return fromQuery;
    }

    if (fromQuery.some((item) => item.id === local.id)) {
      return fromQuery;
    }

    return [local, ...fromQuery].sort((a, b) => b.updatedAt - a.updatedAt);
  }, [sessionsQuery.data, state.session]);

  const pipelineStatusLabel = useMemo(() => getPipelineStatusLabel(state), [state]);
  const selectedSession = state.session ?? sessions.find((session) => session.id === selectedSessionId) ?? null;
  const isStreaming = Boolean(state.streamingMessageId) || sendMessageMutation.isPending;

  const persistSession = async (session: ChatSession) => {
    try {
      await saveSessionMutation.mutateAsync(session);
    } catch (error) {
      setComposerError(formatError(error, 'Unable to save this chat locally.'));
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
    dispatch({ type: 'start_stream', session: nextSession, assistantMessageId });

    const request: RAGStreamRequest = {
      question: text,
      llm_provider: provider,
      classification_filter: ['public'],
      source_id: RAG_DEFAULT_SOURCE_ID,
      top_k: RAG_DEFAULT_TOP_K,
      temperature: 0.7,
      min_similarity: 0.2,
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
            dispatch({
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
            dispatch({ type: 'append_token', messageId: assistantMessageId, token: event.data.text });
            return;
          }

          if (event.type === 'source') {
            dispatch({ type: 'add_source', messageId: assistantMessageId, source: event.data });
            return;
          }

          if (event.type === 'final') {
            dispatch({ type: 'apply_final', messageId: assistantMessageId, answer: event.data.answer });
            dispatch({
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
            dispatch({ type: 'set_error', messageId: assistantMessageId, error: event.data.message });
            return;
          }

          if (event.type === 'done') {
            receivedDoneEvent = true;
            dispatch({ type: 'complete', messageId: assistantMessageId });
          }
        },
      });

      if (!receivedDoneEvent && !cancelRequestedRef.current) {
        dispatch({
          type: 'set_error',
          messageId: assistantMessageId,
          error: 'The stream ended unexpectedly before completion.',
        });
        dispatch({ type: 'complete', messageId: assistantMessageId });
      }
    } catch (error) {
      if (!cancelRequestedRef.current && !isAbortError(error)) {
        const message = formatError(error, 'Unable to get a response from the RAG service.');
        dispatch({ type: 'set_error', messageId: assistantMessageId, error: message });
        dispatch({ type: 'complete', messageId: assistantMessageId });
        setComposerError(message);
      }
    } finally {
      abortControllerRef.current = null;
      cancelRequestedRef.current = false;

      if (stateRef.current.session) {
        await persistSession(stateRef.current.session);
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
    setSelectedSessionId(null);
    dispatch({ type: 'reset' });
    setProvider('local');
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
          setProvider('local');
        }
      }
    } catch (error) {
      setComposerError(formatError(error, 'Unable to delete this chat.'));
    }
  };

  const loadingSession = Boolean(selectedSessionId) && sessionQuery.isLoading && !state.streamingMessageId;
  const sessionsLoading = sessionsQuery.isLoading && !sessionsQuery.data;

  return (
    <div className="chat-shell">
      <aside className="chat-sidebar">
        <div className="chat-sidebar__brand">
          <p className="chat-sidebar__eyebrow">Assistant</p>
          <h1>RAG Studio</h1>
          <p className="chat-sidebar__subtitle">Grounded answers with citations and live pipeline traces.</p>
        </div>

        <button type="button" className="new-chat-btn" onClick={onNewChat}>
          + New chat
        </button>

        {sessionsLoading ? (
          <div className="sidebar-feedback">Loading conversations...</div>
        ) : null}

        {sessionsQuery.isError ? (
          <div className="sidebar-feedback sidebar-feedback--error">
            <p>{formatError(sessionsQuery.error, 'Unable to load saved sessions.')}</p>
            <button type="button" onClick={() => void sessionsQuery.refetch()}>
              Retry
            </button>
          </div>
        ) : null}

        {!sessionsLoading && !sessionsQuery.isError && sessions.length === 0 ? (
          <div className="sidebar-feedback">No saved chats yet. Start your first conversation.</div>
        ) : null}

        <div className="session-list" role="list" aria-label="Saved chat sessions">
          {sessions.map((session) => (
            <article
              key={session.id}
              className={clsx('session-item', selectedSessionId === session.id && 'session-item--active')}
              role="listitem"
            >
              <button
                type="button"
                className="session-item__button"
                onClick={() => {
                  setSelectedSessionId(session.id);
                  setComposerError(null);
                }}
              >
                <span className="session-item__title">{session.title}</span>
                <span className="session-item__preview">{getSessionPreview(session)}</span>
                <span className="session-item__time">{formatSessionTime(session.updatedAt)}</span>
              </button>
              <button
                type="button"
                className="session-item__delete"
                onClick={() => void onDeleteSession(session.id)}
                disabled={deleteSessionMutation.isPending}
                aria-label={`Delete ${session.title}`}
              >
                Delete
              </button>
            </article>
          ))}
        </div>
      </aside>

      <main className="chat-workspace">
        <header className="chat-workspace__header">
          <div>
            <h2>{selectedSession?.title ?? 'New conversation'}</h2>
            <p>{(selectedSession?.messages.length ?? 0) / 2} turns in this chat</p>
          </div>
          <div className="chat-workspace__status">
            <span className={clsx('pipeline-pill', isStreaming ? 'pipeline-pill--live' : 'pipeline-pill--idle')}>
              {pipelineStatusLabel}
            </span>
          </div>
        </header>

        <section className="chat-workspace__body" aria-live="polite">
          {loadingSession ? <div className="workspace-feedback">Loading conversation...</div> : null}

          {sessionQuery.isError && !state.session ? (
            <div className="workspace-feedback workspace-feedback--error">
              <p>{formatError(sessionQuery.error, 'Unable to load the selected conversation.')}</p>
              <button type="button" onClick={() => void sessionQuery.refetch()}>
                Retry
              </button>
            </div>
          ) : null}

          {!loadingSession && !sessionQuery.isError ? (
            <MessageList messages={state.session?.messages ?? []} streamingMessageId={state.streamingMessageId} />
          ) : null}
        </section>

        <footer className="chat-workspace__composer">
          {composerError ? (
            <div className="composer-error" role="alert">
              <span>{composerError}</span>
              <button type="button" onClick={() => setComposerError(null)}>
                Dismiss
              </button>
            </div>
          ) : null}

          <MessageComposer
            provider={provider}
            onProviderChange={setProvider}
            onSend={onSend}
            onCancel={onCancel}
            isStreaming={isStreaming}
          />
        </footer>
      </main>
    </div>
  );
}
