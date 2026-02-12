import { useCallback, useEffect, useState } from 'react';
import { AuthFormPage } from '../features/auth/components/AuthFormPage';
import { useAuth } from '../features/auth/context/AuthContext';
import { ChatPage } from '../features/chat/components/ChatPage';

export function App() {
  const { isInitializing, isAuthenticated, login, signup } = useAuth();
  const [pathname, setPathname] = useState(() => window.location.pathname);
  const [signupMessage, setSignupMessage] = useState<string | null>(null);

  const navigate = useCallback((nextPath: string, replace = false) => {
    if (replace) {
      window.history.replaceState({}, '', nextPath);
    } else {
      window.history.pushState({}, '', nextPath);
    }
    setPathname(nextPath);
  }, []);

  useEffect(() => {
    const onPopState = () => {
      setPathname(window.location.pathname);
    };
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  useEffect(() => {
    if (isInitializing) {
      return;
    }
    const onAuthRoute = pathname === '/login' || pathname === '/signup';
    if (isAuthenticated && onAuthRoute) {
      navigate('/', true);
      return;
    }
    if (!isAuthenticated && !onAuthRoute) {
      navigate('/login', true);
    }
  }, [isAuthenticated, isInitializing, navigate, pathname]);

  if (isInitializing) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-300 grid place-items-center">
        <span className="text-sm tracking-wide">Checking session...</span>
      </div>
    );
  }

  if (!isAuthenticated) {
    const isSignup = pathname === '/signup';
    return (
      <AuthFormPage
        mode={isSignup ? 'signup' : 'login'}
        message={!isSignup ? signupMessage : null}
        onSwitchMode={() => {
          setSignupMessage(null);
          navigate(isSignup ? '/login' : '/signup');
        }}
        onSubmit={async (email, password) => {
          if (isSignup) {
            await signup(email, password);
            setSignupMessage('Account created. Please log in with your new credentials.');
            navigate('/login', true);
            return;
          }

          await login(email, password);
          setSignupMessage(null);
          navigate('/', true);
        }}
      />
    );
  }

  return <ChatPage />;
}
