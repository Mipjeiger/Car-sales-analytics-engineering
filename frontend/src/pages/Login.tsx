// frontend/src/pages/Login.tsx
import { FormEvent, useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { loginSuccess } from "@/store/slices/authSlice";
import { AUTH_LOGIN_URL } from "@/utils/constants";
import type { UserRole, User } from "@/types/auth";
import {
  Box,
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  InputAdornment,
  IconButton,
  useTheme,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import { Visibility, VisibilityOff, Email, Lock } from "@mui/icons-material";

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

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(5, 4),
  borderRadius: theme.spacing(2),
  boxShadow: "0 8px 32px rgba(0, 0, 0, 0.1)",
  width: "100%",
  maxWidth: 440,
  [theme.breakpoints.down("sm")]: {
    padding: theme.spacing(3, 2),
    margin: theme.spacing(2),
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  padding: theme.spacing(1.5),
  borderRadius: theme.spacing(1),
  textTransform: "none",
  fontSize: "1rem",
  fontWeight: 600,
  marginTop: theme.spacing(2),
  "&:hover": {
    transform: "translateY(-1px)",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
  },
  transition: "all 0.2s ease-in-out",
}));

export default function Login() {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const dispatch = useAppDispatch();
  const { user } = useAppSelector((s) => s.auth);

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      const from = (location.state as any)?.from?.pathname || "/";
      navigate(from, { replace: true });
    }
  }, [user, navigate, location]);

  const handleTogglePassword = () => {
    setShowPassword(!showPassword);
  };

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      console.log('🔐 Attempting login to:', AUTH_LOGIN_URL);
      
      const response = await fetch(AUTH_LOGIN_URL, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ email, password }),
      });

      console.log('📊 Response status:', response.status);

      let payload: LoginSuccessResponse | LoginErrorResponse | null = null;
      const responseText = await response.text();
      console.log('📄 Response body:', responseText);
      
      try {
        payload = responseText ? JSON.parse(responseText) : null;
      } catch {
        payload = null;
      }

      if (!response.ok) {
        const detail = payload && "detail" in payload 
          ? payload.detail 
          : responseText || "Invalid email or password";
        throw new Error(typeof detail === 'string' ? detail : "Authentication failed");
      }

      if (!payload || !("access_token" in payload)) {
        throw new Error("Login succeeded but token is missing from response");
      }

      const userData: User = {
        email: payload.email,
        role: payload.role.toLowerCase() as UserRole,
        id: undefined,
        name: undefined,
      };

      console.log('✅ Login successful for:', userData.email);

      dispatch(
        loginSuccess({
          user: userData,
          token: payload.access_token,
        })
      );

      // Navigate to home or the page user came from
      const from = (location.state as any)?.from?.pathname || "/";
      navigate(from, { replace: true });
      
    } catch (err) {
      const errorMessage = err instanceof Error 
        ? err.message 
        : "An unexpected error occurred";
      console.error('❌ Login error:', errorMessage);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  }

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.main} 50%, ${theme.palette.secondary.main} 100%)`,
        padding: 2,
      }}
    >
      <Container maxWidth="sm" sx={{ display: "flex", justifyContent: "center" }}>
        <StyledPaper elevation={3}>
          <Box sx={{ textAlign: "center", mb: 4 }}>
            <Box
              sx={{
                width: 64,
                height: 64,
                borderRadius: "50%",
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white",
                fontSize: 32,
                fontWeight: "bold",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
                margin: "0 auto",
                mb: 2,
              }}
            >
              🚗
            </Box>

            <Typography
              variant="h4"
              sx={{
                fontWeight: 700,
                background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                mb: 0.5,
              }}
            >
              Welcome Back
            </Typography>

            <Typography variant="body2" color="text.secondary">
              Sign in to access your Car Sales Intelligence dashboard
            </Typography>
          </Box>

          {error && (
            <Alert
              severity="error"
              sx={{ mb: 3, borderRadius: 1 }}
              onClose={() => setError("")}
            >
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              label="Email Address"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
              disabled={loading}
              sx={{ mb: 3 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Email color="action" />
                  </InputAdornment>
                ),
              }}
              placeholder="you@company.com"
            />

            <TextField
              fullWidth
              label="Password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
              disabled={loading}
              sx={{ mb: 0.5 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Lock color="action" />
                  </InputAdornment>
                ),
                endAdornment: (
                  <InputAdornment position="end">
                    <IconButton
                      aria-label="toggle password visibility"
                      onClick={handleTogglePassword}
                      edge="end"
                      disabled={loading}
                    >
                      {showPassword ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  </InputAdornment>
                ),
              }}
              placeholder="Enter your password"
            />

            <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 3 }}>
              <Typography
                variant="body2"
                color="primary"
                sx={{
                  cursor: "pointer",
                  "&:hover": { textDecoration: "underline" },
                }}
              >
                Forgot password?
              </Typography>
            </Box>

            <StyledButton
              fullWidth
              type="submit"
              variant="contained"
              disabled={loading}
              startIcon={loading && <CircularProgress size={20} color="inherit" />}
            >
              {loading ? "Signing in..." : "Sign in"}
            </StyledButton>
          </form>
        </StyledPaper>
      </Container>
    </Box>
  );
}