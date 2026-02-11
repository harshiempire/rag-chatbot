import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';
import rehypeHighlight from 'rehype-highlight';

type MarkdownRendererProps = {
  content: string;
};

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeSanitize, rehypeHighlight]}
      components={{
        code({ children, className, ...props }) {
          const isInline = !className?.includes('language-');
          const rawText = String(children).replace(/\n$/, '');

          if (isInline) {
            return (
              <code className="inline-code" {...props}>
                {children}
              </code>
            );
          }

          return <CodeBlock className={className} text={rawText} />;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

type CodeBlockProps = {
  className?: string;
  text: string;
};

function CodeBlock({ className, text }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const onCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1000);
  };

  return (
    <div className="code-block-wrapper">
      <button type="button" onClick={onCopy} className="copy-btn">
        {copied ? 'Copied' : 'Copy'}
      </button>
      <pre>
        <code className={className}>{text}</code>
      </pre>
    </div>
  );
}
