
import { MessageSquare, Zap, Database } from "lucide-react";

export function WelcomeScreen() {
    return (
        <div className="flex flex-col items-center justify-center h-full max-w-2xl mx-auto px-4 text-center space-y-8 animate-in fade-in duration-500">
            <div className="space-y-4">
                <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl ring-1 ring-slate-700/50">
                    <Database className="w-8 h-8 text-blue-400" />
                </div>
                <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
                    RAG Studio
                </h1>
                <p className="text-slate-400 text-lg max-w-md mx-auto">
                    Your AI assistant grounded in your data. Ask questions and get accurate, cited answers.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full text-left">
                <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-xl hover:bg-slate-800 transition-colors">
                    <div className="w-8 h-8 bg-blue-500/10 rounded-lg flex items-center justify-center mb-3">
                        <Database className="w-4 h-4 text-blue-400" />
                    </div>
                    <h3 className="font-semibold text-slate-200 mb-1">Grounded</h3>
                    <p className="text-sm text-slate-500">Answers are based on your provided documents and data sources.</p>
                </div>
                <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-xl hover:bg-slate-800 transition-colors">
                    <div className="w-8 h-8 bg-purple-500/10 rounded-lg flex items-center justify-center mb-3">
                        <Zap className="w-4 h-4 text-purple-400" />
                    </div>
                    <h3 className="font-semibold text-slate-200 mb-1">Fast</h3>
                    <p className="text-sm text-slate-500">Optimized for speed with streaming responses and efficient retrieval.</p>
                </div>
                <div className="p-4 bg-slate-800/50 border border-slate-700/50 rounded-xl hover:bg-slate-800 transition-colors">
                    <div className="w-8 h-8 bg-green-500/10 rounded-lg flex items-center justify-center mb-3">
                        <MessageSquare className="w-4 h-4 text-green-400" />
                    </div>
                    <h3 className="font-semibold text-slate-200 mb-1">Interactive</h3>
                    <p className="text-sm text-slate-500">Ask follow-up questions and explore the context of your data.</p>
                </div>
            </div>
        </div>
    );
}
