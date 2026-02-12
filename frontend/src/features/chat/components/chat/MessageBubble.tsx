
import { useState } from "react";
import { Bot, ChevronDown, ChevronRight, FileText, AlertCircle } from "lucide-react";

import { MarkdownRenderer } from "../MarkdownRenderer";
import { StatusHistory } from "./StatusHistory";
import type { ChatMessage, PipelineStatusItem } from "../../../../shared/types/chat";

interface MessageBubbleProps {
    message: ChatMessage;
    isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
    const isUser = message.role === "user";
    const [showSources, setShowSources] = useState(false);

    if (isUser) {
        return (
            <div className="flex w-full justify-end mb-6">
                <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm max-w-[85%] md:max-w-[75%] shadow-md">
                    <div className="whitespace-pre-wrap">{message.content}</div>
                </div>
            </div>
        );
    }

    const hasSources = message.sources && message.sources.length > 0;
    const hasStatus = message.statusHistory && message.statusHistory.length > 0;
    // Determine if we should show the thinking state
    const isThinking = isStreaming && !message.content && hasStatus;
    const isDoneThinking = message.content.length > 0 && hasStatus;

    return (
        <div className="flex w-full gap-4 mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className="flex-shrink-0 mt-1">
                <div className="w-8 h-8 rounded-full bg-emerald-600/20 flex items-center justify-center border border-emerald-500/20">
                    <Bot className="w-5 h-5 text-emerald-400" />
                </div>
            </div>

            <div className="flex-1 min-w-0 space-y-2">
                {/* Thinking / Process State */}
                {(isThinking || isDoneThinking) && (
                    <div className="mb-2">
                        <StatusHistory
                            statusHistory={message.statusHistory || []}
                            finished={!!message.done || !!message.error}
                        />
                    </div>
                )}

                {/* Sources Accordion */}
                {hasSources && (
                    <div className="mb-2">
                        <button
                            onClick={() => setShowSources(!showSources)}
                            className="flex items-center gap-2 text-xs font-medium text-slate-500 hover:text-slate-300 transition-colors"
                        >
                            <FileText className="w-3 h-3" />
                            <span>{message.sources?.length} Sources found</span>
                            {showSources ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                        </button>

                        {showSources && (
                            <div className="mt-2 grid gap-2 grid-cols-1 sm:grid-cols-2">
                                {message.sources?.map((source) => (
                                    <div key={source.id} className="bg-slate-800/50 border border-slate-700 p-3 rounded-lg text-xs hover:bg-slate-800 transition-colors cursor-pointer group">
                                        <div className="font-semibold text-slate-300 mb-1 truncate group-hover:text-blue-400 transition-colors">{source.title}</div>
                                        <div className="text-slate-500 line-clamp-2">{source.snippet}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Content */}
                <div className="prose prose-invert prose-sm max-w-none leading-relaxed text-slate-200">
                    {message.content ? (
                        <MarkdownRenderer content={message.content} />
                    ) : isThinking ? (
                        <span className="text-slate-500 animate-pulse">Generating response...</span>
                    ) : null}
                </div>

                {/* Error State */}
                {message.error && (
                    <div className="bg-red-900/20 border border-red-500/50 text-red-200 px-4 py-3 rounded-lg text-sm flex items-start gap-3 mt-2">
                        <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-500" />
                        <div>
                            <p className="font-medium">Error generation response</p>
                            <p className="text-red-300/80 text-xs mt-1">{message.error}</p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}


