
import { LogOut, MessageSquare, Plus, Trash2 } from "lucide-react";
import { format, isToday, isYesterday } from "date-fns";
import { cn } from "../../../../shared/utils/cn";
import { Button } from "../ui/Button";
import type { ChatSession } from "../../../../shared/types/chat";

interface SidebarProps {
    sessions: ChatSession[];
    selectedSessionId: string | null;
    userEmail: string;
    onLogout: () => void;
    onSelectSession: (id: string) => void;
    onDeleteSession: (id: string) => void;
    onNewChat: () => void;
    isLoading?: boolean;
}

export function Sidebar({
    sessions,
    selectedSessionId,
    userEmail,
    onLogout,
    onSelectSession,
    onDeleteSession,
    onNewChat,
    isLoading,
}: SidebarProps) {
    // Group sessions by date
    const groupedSessions = sessions.reduce((groups, session) => {
        const date = new Date(session.updatedAt);
        let key = "Older";

        if (isToday(date)) {
            key = "Today";
        } else if (isYesterday(date)) {
            key = "Yesterday";
        } else {
            key = format(date, "MMMM yyyy");
        }

        if (!groups[key]) {
            groups[key] = [];
        }
        groups[key].push(session);
        return groups;
    }, {} as Record<string, ChatSession[]>);

    const groups = ["Today", "Yesterday", ...Object.keys(groupedSessions).filter(k => k !== "Today" && k !== "Yesterday")];

    const userLabel = userEmail.split("@")[0] || "User";
    const avatar = userEmail.slice(0, 1).toUpperCase() || "U";

    return (
        <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 w-[260px] flex-shrink-0">
            <div className="p-4">
                <Button
                    onClick={onNewChat}
                    className="w-full justify-start gap-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
                    variant="ghost"
                >
                    <Plus className="h-4 w-4" />
                    New chat
                </Button>
            </div>

            <div className="flex-1 overflow-y-auto px-2 pb-4">
                {isLoading ? (
                    <div className="text-slate-500 text-sm p-4 text-center">Loading...</div>
                ) : sessions.length === 0 ? (
                    <div className="text-slate-500 text-sm p-4 text-center">No chats yet</div>
                ) : (
                    <div className="space-y-6">
                        {groups.map((group) => {
                            const groupSessions = groupedSessions[group];
                            if (!groupSessions?.length) return null;

                            return (
                                <div key={group}>
                                    <h3 className="px-2 text-xs font-semibold text-slate-500 mb-2">{group}</h3>
                                    <div className="space-y-1">
                                        {groupSessions.map((session) => (
                                            <div
                                                key={session.id}
                                                className="group relative"
                                            >
                                                <button
                                                    onClick={() => onSelectSession(session.id)}
                                                    className={cn(
                                                        "w-full text-left px-2 py-2 rounded-md text-sm transition-colors flex items-center gap-2",
                                                        selectedSessionId === session.id
                                                            ? "bg-slate-800 text-slate-200"
                                                            : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                                                    )}
                                                >
                                                    <MessageSquare className="h-4 w-4 flex-shrink-0 opacity-70" />
                                                    <span className="truncate flex-1">{session.title || "New Chat"}</span>
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        onDeleteSession(session.id);
                                                    }}
                                                    className={cn(
                                                        "absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-md text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity",
                                                        selectedSessionId === session.id && "opacity-100"
                                                    )}
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-slate-800">
                <div className="flex items-center gap-2 px-2 py-3 mt-auto">
                    <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-xs">
                        {avatar}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-200 truncate">{userLabel}</p>
                        <p className="text-xs text-slate-500 truncate">{userEmail}</p>
                    </div>
                    <button
                        type="button"
                        onClick={onLogout}
                        className="rounded-md p-1.5 text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors"
                        title="Log out"
                    >
                        <LogOut className="h-4 w-4" />
                    </button>
                </div>
            </div>
        </div>
    );
}
