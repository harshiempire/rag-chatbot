export type LLMProvider = 'openai' | 'anthropic' | 'google' | 'openrouter' | 'local';
export type DataClassification = 'public' | 'internal' | 'confidential' | 'restricted';

export type PipelineStage = 'embedding' | 'retrieval' | 'prompt_build' | 'generation';
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

export type ChatEvent = StatusEvent | TokenEvent | SourceEvent | FinalEvent | ErrorEvent | DoneEvent;

export interface RAGStreamRequest {
  question: string;
  llm_provider: LLMProvider;
  classification_filter: DataClassification[];
  top_k: number;
  temperature: number;
  min_similarity: number;
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
  usage?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  llmProvider: LLMProvider;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
}
