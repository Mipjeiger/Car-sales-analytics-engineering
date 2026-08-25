import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from "@/store/hooks";
import { loginSuccess } from "@/store/slices/authSlice";
import { AUTH_LOGIN_URL } from "@/utils/constants";
import type { UserRole, User } from "@/types/auth";

interface LoginSuccessResponse {
  access_token: string;
  token_type: "bearer";
  role: string;
  email: string;
  expires_in: number;
}

interface LoginErrorResponse {
  detail?: string;
}

export default function Login() {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await fetch(AUTH_LOGIN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      let payload: LoginSuccessResponse | LoginErrorResponse | null = null;
      try {
        payload = await response.json();
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail =
          payload && "detail" in payload ? payload.detail : undefined;
        throw new Error(detail ?? "Invalid email or password");
      }

      if (!payload || !("access_token" in payload)) {
        throw new Error("Login succeeded but token is missing from response");
      }

      const user: User = {
        email: payload.email,
        role: payload.role.toLowerCase() as UserRole,
        id: undefined,
        name: undefined
      };

      dispatch(
        loginSuccess({
          user,
          token: payload.access_token,
        }),
      );

      navigate("/", { replace: true });
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "An unexpected error occurred";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-container">
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            autoComplete="email"
            required
          />
        </div>

        <div>
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            placeholder="Enter your password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
            required
          />
        </div>

        {error && (
          <div role="alert" style={{ color: 'red', marginTop: '10px' }}>
            {error}
          </div>
        )}

        <button type="submit" disabled={loading} style={{ marginTop: '15px' }}>
          {loading ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </div>
  );
}