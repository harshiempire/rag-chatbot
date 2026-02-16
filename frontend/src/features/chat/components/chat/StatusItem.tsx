import { CheckCircle2, Compass, Loader2, Search, Database, FileText, Sparkles } from "lucide-react";
import { cn } from "../../../../shared/utils/cn";
import type { PipelineStatusItem } from "../../../../shared/types/chat";

interface StatusItemProps {
    status: PipelineStatusItem;
    done?: boolean;
}

export function StatusItem({ status, done = false }: StatusItemProps) {
    const isComplete = status.state === "done" || done;

    // Icon selection based on stage
    const getIcon = () => {
        if (!isComplete) return <Loader2 className="h-4 w-4 animate-spin text-blue-400" />;

        switch (status.stage) {
            case "routing": return <Compass className="h-4 w-4 text-green-400" />;
            case "embedding": return <FileText className="h-4 w-4 text-green-400" />;
            case "retrieval": return <Search className="h-4 w-4 text-green-400" />;
            case "prompt_build": return <Database className="h-4 w-4 text-green-400" />;
            case "generation": return <Sparkles className="h-4 w-4 text-green-400" />;
            default: return <CheckCircle2 className="h-4 w-4 text-green-400" />;
        }
    };

    // Label formatting with metadata
    const getLabel = () => {
        // If we have a specific count in metadata, use it
        if (status.stage === "retrieval" && status.state === "done" && status.meta?.retrieved_count !== undefined) {
            const count = status.meta.retrieved_count;
            const partDistribution = status.meta.part_distribution as Record<string, number> | undefined;
            if (partDistribution && typeof partDistribution === "object") {
                const parts = Object.entries(partDistribution)
                    .filter(([, partCount]) => typeof partCount === "number")
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 3)
                    .map(([part, partCount]) => `${part}:${partCount}`)
                    .join(", ");
                if (parts) {
                    return `Retrieved ${count} source${count === 1 ? '' : 's'} (${parts})`;
                }
            }
            return `Retrieved ${count} source${count === 1 ? '' : 's'}`;
        }

        // Duration if available
        const duration = status.meta?.duration_ms ? ` (${Math.round(status.meta.duration_ms as number)}ms)` : "";

        let label = status.label;
        if (isComplete && label === "Generating answer") {
            label = "Answer generated";
        }

        return (
            <span>
                {label}
                {duration && <span className="text-slate-500 text-xs ml-2">{duration}</span>}
            </span>
        );
    };

    return (
        <div className="flex items-center gap-3 text-sm text-slate-300 animate-in fade-in duration-300">
            <div className="flex-shrink-0 flex items-center justify-center w-5 h-5">
                {getIcon()}
            </div>
            <div className={cn("flex-1 truncate", !isComplete && "animate-pulse")}>
                {getLabel()}
            </div>
        </div>
    );
}
