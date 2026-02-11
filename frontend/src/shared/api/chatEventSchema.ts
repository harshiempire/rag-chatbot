import { z } from 'zod';
import type { ChatEvent } from '../types/chat';

const statusEventSchema = z.object({
  type: z.literal('status'),
  data: z.object({
    stage: z.enum(['embedding', 'retrieval', 'prompt_build', 'generation']),
    state: z.enum(['start', 'done']),
    label: z.string(),
    meta: z.record(z.unknown()).optional(),
  }),
});

const tokenEventSchema = z.object({
  type: z.literal('token'),
  data: z.object({
    text: z.string(),
  }),
});

const sourceEventSchema = z.object({
  type: z.literal('source'),
  data: z.object({
    id: z.string(),
    title: z.string(),
    snippet: z.string(),
    score: z.number(),
    metadata: z.record(z.unknown()).optional(),
  }),
});

const finalEventSchema = z.object({
  type: z.literal('final'),
  data: z.object({
    answer: z.string(),
    timings_ms: z.record(z.number()),
    retrieved_count: z.number(),
    prompt_context_count: z.number(),
  }),
});

const errorEventSchema = z.object({
  type: z.literal('error'),
  data: z.object({
    code: z.string(),
    message: z.string(),
  }),
});

const doneEventSchema = z.object({
  type: z.literal('done'),
  data: z.record(z.never()).or(z.object({})),
});

export const chatEventSchema = z.discriminatedUnion('type', [
  statusEventSchema,
  tokenEventSchema,
  sourceEventSchema,
  finalEventSchema,
  errorEventSchema,
  doneEventSchema,
]);

export const parseChatEvent = (value: unknown): ChatEvent | null => {
  const parsed = chatEventSchema.safeParse(value);
  return parsed.success ? (parsed.data as ChatEvent) : null;
};
