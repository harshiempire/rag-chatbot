import type { PipelineStatusItem } from '../../../shared/types/chat';

interface StatusTimelineProps {
  items: PipelineStatusItem[];
}

export function StatusTimeline({ items }: StatusTimelineProps) {
  if (!items.length) {
    return null;
  }

  return (
    <div className="status-timeline">
      {items.map((item, index) => (
        <div key={`${item.stage}-${item.state}-${index}`} className="status-item">
          <span className={`status-dot ${item.state === 'done' ? 'done' : 'start'}`} />
          <div>
            <div className="status-label">{item.label}</div>
            <div className="status-meta">
              {item.stage} - {item.state}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
