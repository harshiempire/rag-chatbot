import { useEffect, useMemo, useReducer, useRef, useState } from 'react';
import clsx from 'clsx';
import type {
  ChatEvent,
  ChatMessage,
  ChatSession,
  LLMProvider,
  PipelineStatusItem,
  RAGStreamRequest,
} from '../../../shared/types/chat';
import { createId } from '../../../shared/utils/id';
import { useChatSessionQuery } from '../hooks/useChatSessionQuery';
import { useChatSessionsQuery } from '../hooks/useChatSessionsQuery';
import { useSaveSessionMutation } from '../hooks/useSaveSessionMutation';
import { useSendMessageMutation } from '../hooks/useSendMessageMutation';
import { MessageComposer } from './MessageComposer';
import { MessageList } from './MessageList';

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

export function ChatPage() {
  const sessionsQuery = useChatSessionsQuery();
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [provider, setProvider] = useState<LLMProvider>('local');

  const sessionQuery = useChatSessionQuery(selectedSessionId);
  const saveSessionMutation = useSaveSessionMutation();
  const sendMessageMutation = useSendMessageMutation();

  const [state, dispatch] = useReducer(chatReducer, initialState);
  const stateRef = useRef(state);
  const abortControllerRef = useRef<AbortController | null>(null);

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

  const onSend = async (text: string) => {
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
      top_k: 5,
      temperature: 0.7,
      min_similarity: 0.2,
    };

    abortControllerRef.current = new AbortController();

    try {
      await sendMessageMutation.mutateAsync({
        request,
        signal: abortControllerRef.current.signal,
        sessionSnapshot: () => stateRef.current.session,
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
            return;
          }

          if (event.type === 'error') {
            dispatch({ type: 'set_error', messageId: assistantMessageId, error: event.data.message });
            return;
          }

          if (event.type === 'done') {
            dispatch({ type: 'complete', messageId: assistantMessageId });
          }
        },
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown streaming error';
      dispatch({ type: 'set_error', messageId: assistantMessageId, error: message });
      dispatch({ type: 'complete', messageId: assistantMessageId });

      if (stateRef.current.session) {
        await saveSessionMutation.mutateAsync(stateRef.current.session);
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  const onCancel = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    if (state.streamingMessageId) {
      dispatch({
        type: 'set_error',
        messageId: state.streamingMessageId,
        error: 'Canceled by user',
      });
      dispatch({ type: 'complete', messageId: state.streamingMessageId });
    }
  };

  const onNewChat = () => {
    setSelectedSessionId(null);
    dispatch({ type: 'reset' });
    setProvider('local');
  };

  const loading = sessionsQuery.isLoading || (Boolean(selectedSessionId) && sessionQuery.isLoading);

  return (
    <div className="chat-page">
      <aside className="chat-sidebar">
        <button type="button" className="new-chat-btn" onClick={onNewChat}>
          New Chat
        </button>

        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              className={clsx('session-item', selectedSessionId === session.id && 'active')}
              onClick={() => setSelectedSessionId(session.id)}
            >
              <span className="session-title">{session.title}</span>
              <span className="session-time">{new Date(session.updatedAt).toLocaleString()}</span>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <h1>RAG Chat</h1>
          <p>Structured streaming with pipeline status and sources</p>
        </header>

        {loading ? <div className="loading">Loading...</div> : <MessageList messages={state.session?.messages ?? []} />}

        <MessageComposer
          provider={provider}
          onProviderChange={setProvider}
          onSend={onSend}
          onCancel={onCancel}
          isStreaming={Boolean(state.streamingMessageId) || sendMessageMutation.isPending}
        />
      </main>
    </div>
  );
}
