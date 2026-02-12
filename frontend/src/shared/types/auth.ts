export interface AuthUser {
  id: string;
  email: string;
  created_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: 'bearer';
  expires_in: number;
  user: AuthUser;
}

export interface AuthMessageResponse {
  message: string;
}

export interface SignupRequest {
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}
