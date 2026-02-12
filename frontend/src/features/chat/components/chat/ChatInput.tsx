
import { useRef } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "../ui/Button";
import { Textarea } from "../ui/Textarea";

interface ChatInputProps {
    onSend: (message: string) => void;
    onStop: () => void;
    isLoading: boolean;
    disabled?: boolean;
}

export function ChatInput({ onSend, onStop, isLoading, disabled }: ChatInputProps) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    const handleSubmit = () => {
        const value = textareaRef.current?.value.trim();
        if (!value || isLoading) return;

        onSend(value);
        if (textareaRef.current) {
            textareaRef.current.value = "";
            textareaRef.current.style.height = "auto";
        }
    };

    const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
        const target = e.currentTarget;
        target.style.height = "auto";
        target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
    };

    return (
        <div className="relative max-w-3xl mx-auto w-full p-4">
            <div className="relative flex items-end gap-2 bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-2 shadow-2xl ring-1 ring-white/5 focus-within:ring-blue-500/50 focus-within:border-blue-500/50 transition-all">
                <Textarea
                    ref={textareaRef}
                    onKeyDown={handleKeyDown}
                    onInput={handleInput}
                    placeholder="Ask anything..."
                    className="min-h-[50px] max-h-[200px] bg-transparent border-0 focus-visible:ring-0 resize-none py-3 text-base text-slate-100 placeholder:text-slate-500"
                    disabled={disabled}
                    rows={1}
                />

                <div className="pb-1 pr-1">
                    {isLoading ? (
                        <Button
                            size="icon"
                            variant="default"
                            className="h-9 w-9 bg-slate-100 text-slate-900 hover:bg-slate-300 rounded-xl"
                            onClick={onStop}
                        >
                            <Square className="h-4 w-4 fill-current" />
                        </Button>
                    ) : (
                        <Button
                            size="icon"
                            variant="default"
                            className="h-9 w-9 bg-blue-600 hover:bg-blue-500 text-white rounded-xl"
                            onClick={handleSubmit}
                            disabled={disabled}
                        >
                            <Send className="h-4 w-4" />
                        </Button>
                    )}
                </div>
            </div>
            <div className="text-center mt-2">
                <p className="text-xs text-slate-500">
                    AI can make mistakes. Please review critical information.
                </p>
            </div>
        </div>
    );
}
