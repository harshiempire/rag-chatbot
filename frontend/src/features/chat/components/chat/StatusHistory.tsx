
import { useState, useEffect } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { StatusItem } from "./StatusItem";
import type { PipelineStatusItem } from "../../../../shared/types/chat";

interface StatusHistoryProps {
    statusHistory: PipelineStatusItem[];
    finished?: boolean;
}

export function StatusHistory({ statusHistory, finished = false }: StatusHistoryProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [latestStatus, setLatestStatus] = useState<PipelineStatusItem | null>(null);

    useEffect(() => {
        if (statusHistory.length > 0) {
            setLatestStatus(statusHistory[statusHistory.length - 1]);
        }
    }, [statusHistory]);

    if (!latestStatus && statusHistory.length === 0) return null;

    return (
        <div className="flex flex-col w-full text-sm font-medium text-slate-400 mb-2">
            {/* Latest Status / Toggle Header */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center gap-2 py-2 px-3 hover:bg-slate-900/50 rounded-lg transition-colors text-left w-full group"
            >
                <div className="flex-shrink-0 mt-0.5 text-slate-500 group-hover:text-slate-300">
                    {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                </div>

                <div className="flex-1">
                    {latestStatus && (
                        <StatusItem status={latestStatus} done={finished} />
                    )}
                </div>
            </button>

            {/* Expanded History */}
            {isOpen && (
                <div className="pl-4 ml-3 border-l-2 border-slate-800 space-y-3 py-2 mt-1 animate-in slide-in-from-top-2 duration-200">
                    {statusHistory.slice(0, -1).map((status, idx) => (
                        <div key={idx} className="relative">
                            {/* Connector dot */}
                            <div className="absolute -left-[21px] top-2 h-2 w-2 rounded-full bg-slate-700 ring-4 ring-slate-950" />
                            <StatusItem
                                key={idx}
                                status={status}
                                done={idx < statusHistory.length - 1 || finished}
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
