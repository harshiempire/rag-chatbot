import { useState } from 'react';
import type { LLMProvider } from '../../../shared/types/chat';

interface MessageComposerProps {
  onSend: (text: string) => Promise<void>;
  onCancel: () => void;
  isStreaming: boolean;
  provider: LLMProvider;
  onProviderChange: (provider: LLMProvider) => void;
}

const providers: LLMProvider[] = ['local', 'openrouter'];

export function MessageComposer({
  onSend,
  onCancel,
  isStreaming,
  provider,
  onProviderChange,
}: MessageComposerProps) {
  const [value, setValue] = useState('');

  const submit = async () => {
    const text = value.trim();
    if (!text || isStreaming) {
      return;
    }

    setValue('');
    await onSend(text);
  };

  return (
    <div className="composer">
      <div className="composer-top">
        <label htmlFor="provider">Provider</label>
        <select
          id="provider"
          value={provider}
          onChange={(event) => onProviderChange(event.target.value as LLMProvider)}
          disabled={isStreaming}
        >
          {providers.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </div>

      <textarea
        placeholder="Ask a question..."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={isStreaming}
        rows={4}
      />

      <div className="composer-actions">
        <button type="button" onClick={submit} disabled={isStreaming || !value.trim()}>
          Send
        </button>
        <button type="button" onClick={onCancel} disabled={!isStreaming}>
          Cancel
        </button>
      </div>
    </div>
  );
}
