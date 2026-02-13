import { createContext, useCallback, useContext, useEffect, useMemo, useState, type PropsWithChildren } from 'react';
import * as authApi from '../../../shared/api/authApi';
import { clearAccessToken, getAccessToken, setAccessToken, setRefreshHandler } from '../../../shared/auth/session';
import type { AuthUser } from '../../../shared/types/auth';

interface AuthContextValue {
  user: AuthUser | null;
  isInitializing: boolean;
  isAuthenticated: boolean;
  signup: (email: string, password: string) => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isUnauthorizedError(error: unknown): boolean {
  if (!(error instanceof Error)) {
    return false;
  }
  return /401|unauthorized|invalid|expired/i.test(error.message);
}

export function AuthProvider({ children }: PropsWithChildren) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  const refreshSession = useCallback(async (): Promise<string | null> => {
    try {
      const response = await authApi.refresh();
      setAccessToken(response.access_token);
      setUser(response.user);
      return response.access_token;
    } catch {
      clearAccessToken();
      setUser(null);
      return null;
    }
  }, []);

  useEffect(() => {
    setRefreshHandler(refreshSession);
    return () => setRefreshHandler(null);
  }, [refreshSession]);

  useEffect(() => {
    let cancelled = false;

    const initialize = async () => {
      const existingToken = getAccessToken();
      if (!existingToken) {
        await refreshSession();
        if (!cancelled) {
          setIsInitializing(false);
        }
        return;
      }

      try {
        const currentUser = await authApi.me(existingToken);
        if (!cancelled) {
          setUser(currentUser);
        }
      } catch (error) {
        if (isUnauthorizedError(error)) {
          const refreshed = await refreshSession();
          if (!refreshed && !cancelled) {
            setUser(null);
          }
        } else {
          clearAccessToken();
          if (!cancelled) {
            setUser(null);
          }
        }
      } finally {
        if (!cancelled) {
          setIsInitializing(false);
        }
      }
    };

    void initialize();
    return () => {
      cancelled = true;
    };
  }, [refreshSession]);

  const signup = useCallback(async (email: string, password: string) => {
    await authApi.signup({ email, password });
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const response = await authApi.login({ email, password });
    setAccessToken(response.access_token);
    setUser(response.user);
  }, []);

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      clearAccessToken();
      setUser(null);
    }
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isInitializing,
      isAuthenticated: Boolean(user),
      signup,
      login,
      logout,
    }),
    [isInitializing, login, logout, signup, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider.');
  }
  return context;
}
