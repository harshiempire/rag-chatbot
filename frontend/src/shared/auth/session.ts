const ACCESS_TOKEN_STORAGE_KEY = 'rag_access_token_v1';

let inMemoryAccessToken: string | null = null;
let refreshHandler: (() => Promise<string | null>) | null = null;

function readStoredToken(): string | null {
  if (typeof localStorage === 'undefined') {
    return null;
  }
  try {
    return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredToken(token: string | null): void {
  if (typeof localStorage === 'undefined') {
    return;
  }
  try {
    if (token) {
      localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
      return;
    }
    localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  } catch {
    // Ignore localStorage write errors so auth can still proceed in memory.
  }
}

export function getAccessToken(): string | null {
  if (inMemoryAccessToken !== null) {
    return inMemoryAccessToken;
  }
  inMemoryAccessToken = readStoredToken();
  return inMemoryAccessToken;
}

export function setAccessToken(token: string): void {
  inMemoryAccessToken = token;
  writeStoredToken(token);
}

export function clearAccessToken(): void {
  inMemoryAccessToken = null;
  writeStoredToken(null);
}

export function setRefreshHandler(handler: (() => Promise<string | null>) | null): void {
  refreshHandler = handler;
}

export async function refreshAccessToken(): Promise<string | null> {
  if (!refreshHandler) {
    return null;
  }
  return refreshHandler();
}
