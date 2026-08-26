import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAppDispatch } from "@/store/hooks";
import { loginSuccess } from "@/store/slices/authSlice";
import { AUTH_LOGIN_URL } from "@/utils/constants";
import type { UserRole, User } from "@/types/auth";

interface LoginResponse {
  access_token: string;
  token_type: "bearer";
  role: string;
  email: string;
  expires_in: number;
}

interface LoginErrorResponse {
  detail?: string | Array<{ msg?: string }>;
}

function isUserRole(value: string): value is UserRole {
  return value === "admin" || value === "analyst" || value === "viewer";
}

export default function Login() {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await fetch(AUTH_LOGIN_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          email: email.trim().toLowerCase(),
          password,
        }),
      });

      const payload: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        const errorPayload = payload as LoginErrorResponse | null;
        const detail = errorPayload?.detail;

        const message = Array.isArray(detail)
          ? detail[0]?.msg
          : detail;

        throw new Error(message || "Invalid email or password");
      }

      const data = payload as Partial<LoginResponse>;

      if (
        typeof data.access_token !== "string" ||
        typeof data.email !== "string" ||
        typeof data.role !== "string"
      ) {
        throw new Error("Invalid login response from server");
      }

      const role = data.role.toLowerCase();

      if (!isUserRole(role)) {
        throw new Error("Invalid user role returned by server");
      }

      const user: User = {
        email: data.email,
        role,
        id: undefined,
        name: undefined,
      };

      dispatch(
        loginSuccess({
          user,
          token: data.access_token,
        })
      );

      navigate("/", { replace: true });
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the authentication server"
      );
    } finally {
      setLoading(false);
    }
  }
}