
import { useRef, useEffect } from "react";
import { MessageBubble } from "./MessageBubble";
import { WelcomeScreen } from "./WelcomeScreen";
import type { ChatMessage } from "../../../../shared/types/chat";

interface ChatAreaProps {
    messages: ChatMessage[];
    isStreaming: boolean;
    streamingMessageId: string | null;
}

export function ChatArea({ messages, isStreaming, streamingMessageId }: ChatAreaProps) {
    const bottomRef = useRef<HTMLDivElement>(null);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (messages.length > 0) {
            bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages.length, streamingMessageId]);

    if (messages.length === 0) {
        return <WelcomeScreen />;
    }

    return (
        <div className="flex-1 overflow-y-auto px-4 py-8 scroll-smooth" ref={scrollRef}>
            <div className="max-w-3xl mx-auto space-y-8">
                {messages.map((message) => (
                    <MessageBubble
                        key={message.id}
                        message={message}
                        isStreaming={isStreaming && message.id === streamingMessageId}
                    />
                ))}
                <div ref={bottomRef} className="h-4" />
            </div>
        </div>
    );
}
