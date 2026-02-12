import { createParser, type EventSourceMessage } from 'eventsource-parser';

export async function* parseSSEJson<T>(
  response: Response,
  map: (value: unknown) => T | null,
  signal?: AbortSignal
): AsyncGenerator<T> {
  if (!response.body) {
    throw new Error('Missing response body for SSE stream');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const queue: T[] = [];

  const parser = createParser({
    onEvent(event: EventSourceMessage) {
      if (!event.data || event.data === '[DONE]') {
        return;
      }

      try {
        const parsed = JSON.parse(event.data);
        const mapped = map(parsed);
        if (mapped) {
          queue.push(mapped);
        }
      } catch {
        // Ignore malformed chunks to keep the stream resilient.
      }
    },
  });

  while (true) {
    if (signal?.aborted) {
      await reader.cancel();
      break;
    }

    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    parser.feed(decoder.decode(value, { stream: true }));

    while (queue.length > 0) {
      const item = queue.shift();
      if (item) {
        yield item;
      }
    }
  }

  parser.feed(decoder.decode());
  while (queue.length > 0) {
    const item = queue.shift();
    if (item) {
      yield item;
    }
  }
}
