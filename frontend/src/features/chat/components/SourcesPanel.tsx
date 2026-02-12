import type { ChatSource } from '../../../shared/types/chat';

interface SourcesPanelProps {
  sources: ChatSource[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  if (!sources.length) {
    return null;
  }

  return (
    <div className="sources-panel">
      <div className="sources-title">Sources</div>
      <ul>
        {sources.map((source) => (
          <li key={source.id} className="source-item">
            <div className="source-header">
              <strong>{source.title}</strong>
              <span>{source.score.toFixed(3)}</span>
            </div>
            <div className="source-snippet">{source.snippet}</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
