import { useState, type FormEvent } from 'react';
import { Mail, Lock, MessageSquare } from 'lucide-react';
import { Button } from '../../chat/components/ui/Button';
import { Input } from '../../chat/components/ui/Input';

type AuthMode = 'login' | 'signup';

interface AuthFormPageProps {
  mode: AuthMode;
  onSubmit: (email: string, password: string) => Promise<void>;
  onSwitchMode: () => void;
  message?: string | null;
}

export function AuthFormPage({ mode, onSubmit, onSwitchMode, message }: AuthFormPageProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const isSignup = mode === 'signup';

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      await onSubmit(email.trim(), password);
    } catch (submitError) {
      const detail = submitError instanceof Error ? submitError.message : 'Authentication failed.';
      setError(detail);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(circle_at_15%_15%,rgba(14,165,233,0.2),transparent_45%),radial-gradient(circle_at_85%_20%,rgba(56,189,248,0.16),transparent_40%),radial-gradient(circle_at_50%_90%,rgba(148,163,184,0.1),transparent_45%)]" />
      <div className="relative w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900/80 shadow-2xl shadow-black/40 backdrop-blur-xl">
        <div className="px-8 pt-8 pb-6 border-b border-slate-800">
          <div className="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/70 px-3 py-1 text-xs text-slate-300">
            <MessageSquare className="h-3.5 w-3.5" />
            RAG Chat
          </div>
          <h1 className="mt-4 text-2xl font-semibold text-slate-100">
            {isSignup ? 'Create your account' : 'Welcome back'}
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            {isSignup
              ? 'Use your email and password to create an account.'
              : 'Sign in with your email to continue to your chats.'}
          </p>
        </div>

        <form className="px-8 py-7 space-y-4" onSubmit={handleSubmit}>
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">Email</span>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <Input
                type="email"
                autoComplete="email"
                className="pl-9 bg-slate-950 border-slate-700"
                placeholder="you@example.com"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>
          </label>

          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-slate-400">Password</span>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <Input
                type="password"
                autoComplete={isSignup ? 'new-password' : 'current-password'}
                className="pl-9 bg-slate-950 border-slate-700"
                placeholder={isSignup ? 'At least 8 characters' : 'Your password'}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                minLength={isSignup ? 8 : 1}
                required
              />
            </div>
          </label>

          {message ? (
            <p className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
              {message}
            </p>
          ) : null}
          {error ? (
            <p className="rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-300">
              {error}
            </p>
          ) : null}

          <Button type="submit" className="w-full bg-cyan-400 text-slate-950 hover:bg-cyan-300" disabled={isSubmitting}>
            {isSubmitting ? 'Please wait...' : isSignup ? 'Create account' : 'Login'}
          </Button>
        </form>

        <div className="px-8 pb-7 text-sm text-slate-400">
          {isSignup ? 'Already have an account?' : "Don't have an account?"}{' '}
          <button type="button" onClick={onSwitchMode} className="font-medium text-cyan-300 hover:text-cyan-200">
            {isSignup ? 'Login' : 'Sign up'}
          </button>
        </div>
      </div>
    </div>
  );
}
