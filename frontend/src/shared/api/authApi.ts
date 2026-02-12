import { API_BASE_URL } from './config';
import type {
  AuthMessageResponse,
  AuthTokenResponse,
  AuthUser,
  LoginRequest,
  SignupRequest,
} from '../types/auth';

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) {
        detail = body.detail;
      }
    } catch {
      // Keep fallback error message.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function signup(payload: SignupRequest): Promise<AuthMessageResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });
  return parseResponse<AuthMessageResponse>(response);
}

export async function login(payload: LoginRequest): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });
  return parseResponse<AuthTokenResponse>(response);
}

export async function refresh(): Promise<AuthTokenResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
    method: 'POST',
    credentials: 'include',
  });
  return parseResponse<AuthTokenResponse>(response);
}

export async function logout(): Promise<AuthMessageResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/logout`, {
    method: 'POST',
    credentials: 'include',
  });
  return parseResponse<AuthMessageResponse>(response);
}

export async function me(accessToken: string): Promise<AuthUser> {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: 'GET',
    credentials: 'include',
    headers: {
      Authorization: `Bearer ${accessToken}`,
    },
  });
  return parseResponse<AuthUser>(response);
}
