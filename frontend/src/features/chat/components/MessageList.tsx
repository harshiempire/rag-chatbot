import type { ChatMessage } from '../../../shared/types/chat';
import { MarkdownRenderer } from './MarkdownRenderer';
import { StatusTimeline } from './StatusTimeline';
import { SourcesPanel } from './SourcesPanel';

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  if (!messages.length) {
    return <div className="empty-state">Start by asking a question about eCFR Chapter 12.</div>;
  }

  return (
    <div className="message-list">
      {messages.map((message) => (
        <div key={message.id} className={`message message-${message.role}`}>
          <div className="message-role">{message.role === 'user' ? 'You' : 'Assistant'}</div>
          <div className="message-content">
            <MarkdownRenderer content={message.content || (message.done ? '_' : 'Thinking...')} />
          </div>

          {message.role === 'assistant' && message.statusHistory ? (
            <StatusTimeline items={message.statusHistory} />
          ) : null}

          {message.role === 'assistant' && message.sources ? <SourcesPanel sources={message.sources} /> : null}

          {message.error ? <div className="message-error">{message.error}</div> : null}
        </div>
      ))}
    </div>
  );
}
