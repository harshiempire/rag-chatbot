import { clearAccessToken, getAccessToken, refreshAccessToken, setAccessToken } from '../auth/session';

async function request(input: RequestInfo | URL, init: RequestInit): Promise<Response> {
  return fetch(input, {
    ...init,
    credentials: init.credentials ?? 'include',
  });
}

function withAuthorization(init: RequestInit, token: string | null): RequestInit {
  if (!token) {
    return init;
  }
  const headers = new Headers(init.headers ?? {});
  headers.set('Authorization', `Bearer ${token}`);
  return {
    ...init,
    headers,
  };
}

export async function authorizedFetch(
  input: RequestInfo | URL,
  init: RequestInit = {}
): Promise<Response> {
  const currentToken = getAccessToken();
  let response = await request(input, withAuthorization(init, currentToken));
  if (response.status !== 401) {
    return response;
  }

  const refreshedToken = await refreshAccessToken();
  if (!refreshedToken) {
    clearAccessToken();
    return response;
  }

  setAccessToken(refreshedToken);
  response = await request(input, withAuthorization(init, refreshedToken));
  return response;
}
