export type LLMProvider = 'openai' | 'anthropic' | 'google' | 'openrouter' | 'local';
export type DataClassification = 'public' | 'internal' | 'confidential' | 'restricted';

export type PipelineStage = 'routing' | 'embedding' | 'retrieval' | 'prompt_build' | 'generation' | 'agent';
export type PipelineState = 'start' | 'done';

export interface PipelineStatusItem {
  stage: PipelineStage;
  state: PipelineState;
  label: string;
  meta?: Record<string, unknown>;
  at: number;
}

export interface ChatSource {
  id: string;
  title: string;
  snippet: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface ChatUsage {
  timingsMs?: Record<string, number>;
  retrievedCount?: number;
  promptContextCount?: number;
}

export interface StatusEvent {
  type: 'status';
  data: {
    stage: PipelineStage;
    state: PipelineState;
    label: string;
    meta?: Record<string, unknown>;
  };
}

export interface TokenEvent {
  type: 'token';
  data: {
    text: string;
  };
}

export interface SourceEvent {
  type: 'source';
  data: ChatSource;
}

export interface FinalEvent {
  type: 'final';
  data: {
    answer: string;
    timings_ms: Record<string, number>;
    retrieved_count: number;
    prompt_context_count: number;
    /** false when RAG found no documents and fell back to an ungrounded LLM response (Decision 1) */
    is_grounded?: boolean;
    ticket_link?: string | null;
  };
}

export interface ErrorEvent {
  type: 'error';
  data: {
    code: string;
    message: string;
  };
}

export interface DoneEvent {
  type: 'done';
  data: Record<string, never>;
}

// Agent-only events (emitted by /rag/agent/stream/events, ignored in normal flow)
export interface ToolCallEvent {
  type: 'tool_call';
  data: {
    tool: string;
    input: string;
  };
}

export interface ToolResultEvent {
  type: 'tool_result';
  data: {
    output: string;
    count: number;
  };
}

export type ChatEvent = StatusEvent | TokenEvent | SourceEvent | FinalEvent | ErrorEvent | DoneEvent | ToolCallEvent | ToolResultEvent;

export interface RAGStreamRequest {
  question: string;
  llm_provider: LLMProvider;
  classification_filter: DataClassification[];
  source_id?: string;
  top_k: number;
  temperature: number;
  min_similarity: number;
  session_id?: string;
  chat_history?: ChatHistoryTurn[];
  conversation_context?: string[];
  metadata_filters?: Record<string, string | string[]>;
  retrieval_mode?: 'dense' | 'hybrid';
}

export interface ChatHistoryTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  createdAt: number;
  done?: boolean;
  error?: string;
  sources?: ChatSource[];
  statusHistory?: PipelineStatusItem[];
  usage?: ChatUsage;
  /** false when this response is ungrounded (RAG found no documents) — Decision 1 */
  isGrounded?: boolean;
  ticketLink?: string | null;
  /** the original user question that triggered the ungrounded response */
  originalQuestion?: string;
  /** populated after the user successfully submits a training ticket for this message */
  submittedTicket?: { id: number; url: string };
}

export interface ChatSession {
  id: string;
  title: string;
  llmProvider: LLMProvider;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}

export interface ChatSessionSummary {
  id: string;
  title: string;
  llmProvider: LLMProvider;
  createdAt: number;
  updatedAt: number;
}
