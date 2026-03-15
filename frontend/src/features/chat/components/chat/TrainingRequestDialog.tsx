import { useState } from "react";
import { AlertTriangle, X, Ticket, ExternalLink, Loader2 } from "lucide-react";
import { API_BASE_URL } from "../../../../shared/api/config";
import { authorizedFetch } from "../../../../shared/api/httpClient";

interface TrainingRequestDialogProps {
  question: string;
  onClose: () => void;
  onTicketCreated?: (ticket: { id: number; url: string }) => void;
}

type Priority = "low" | "normal" | "high";

interface TicketResult {
  ticket_id: number;
  ticket_url: string;
  title: string;
}

export function TrainingRequestDialog({ question, onClose, onTicketCreated }: TrainingRequestDialogProps) {
  const [notes, setNotes] = useState("");
  const [priority, setPriority] = useState<Priority>("normal");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState<TicketResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setIsSubmitting(true);
    setError(null);
    try {
      const resp = await authorizedFetch(`${API_BASE_URL}/rag/ticket`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, notes: notes.trim() || undefined, priority }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${resp.status}`);
      }
      const data: TicketResult = await resp.json();
      setResult(data);
      onTicketCreated?.({ id: data.ticket_id, url: data.ticket_url });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="relative w-full max-w-lg mx-4 bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
          <div className="flex items-center gap-2 text-amber-400">
            <Ticket className="w-5 h-5" />
            <span className="font-semibold text-white">Request Training</span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-6 py-5 space-y-4">
          {!result ? (
            <>
              {/* Question preview */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Unanswered question
                </label>
                <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 line-clamp-3">
                  {question}
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">
                  Additional context <span className="text-slate-500">(optional)</span>
                </label>
                <textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  rows={3}
                  placeholder="Why is this important? Any related regulation references..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-amber-500/60 resize-none"
                />
              </div>

              {/* Priority */}
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-2">Priority</label>
                <div className="flex gap-2">
                  {(["low", "normal", "high"] as Priority[]).map((p) => (
                    <button
                      key={p}
                      onClick={() => setPriority(p)}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-medium capitalize border transition-colors ${
                        priority === p
                          ? p === "high"
                            ? "bg-red-500/20 border-red-500/60 text-red-300"
                            : p === "normal"
                            ? "bg-amber-500/20 border-amber-500/60 text-amber-300"
                            : "bg-slate-600/40 border-slate-500 text-slate-300"
                          : "bg-slate-800 border-slate-700 text-slate-500 hover:border-slate-500"
                      }`}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>

              {/* Error */}
              {error && (
                <div className="bg-red-900/20 border border-red-500/40 text-red-300 px-3 py-2 rounded-lg text-xs flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-3 pt-1">
                <button
                  onClick={onClose}
                  className="flex-1 py-2 rounded-lg text-sm text-slate-400 border border-slate-700 hover:border-slate-500 hover:text-slate-200 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting}
                  className="flex-1 py-2 rounded-lg text-sm font-medium bg-amber-500 hover:bg-amber-400 text-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Submitting…
                    </>
                  ) : (
                    "Submit ticket"
                  )}
                </button>
              </div>
            </>
          ) : (
            /* Success state */
            <div className="text-center py-4 space-y-4">
              <div className="w-12 h-12 rounded-full bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center mx-auto">
                <Ticket className="w-6 h-6 text-emerald-400" />
              </div>
              <div>
                <p className="font-medium text-white">Ticket #{result.ticket_id} created</p>
                <p className="text-xs text-slate-400 mt-1">
                  The team has been notified to add training data for this topic.
                </p>
              </div>
              <a
                href={result.ticket_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 text-xs text-emerald-400 underline hover:text-emerald-300 transition-colors"
              >
                View ticket in Zammad <ExternalLink className="w-3 h-3" />
              </a>
              <button
                onClick={onClose}
                className="block w-full py-2 rounded-lg text-sm text-slate-400 border border-slate-700 hover:border-slate-500 hover:text-slate-200 transition-colors"
              >
                Close
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
