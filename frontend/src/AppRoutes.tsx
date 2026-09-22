// frontend/src/AppRoutes.tsx
import type { ReactElement } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Layout } from "@/components/common/Layout";
import Dashboard from "@/pages/Dashboard";
import SearchPage from "@/pages/Search";
import AnalyticsPage from "@/pages/Analytics";
import ChatPage from "@/pages/Chat";
import DamageDetectionPage from "@/pages/DamageDetection";
import ModelManagementPage from "@/pages/ModelManagement";
import SettingsPage from "@/pages/Settings";
import LoginPage from "@/pages/Login";
import { useAppSelector } from "@/store/hooks";
import type { UserRole, User } from "@/types/auth";
import {
  AUTH_TOKEN_KEY,
  AUTH_ROLE_KEY,
  AUTH_EMAIL_KEY,
} from "@/utils/constants";

function Protected({
  children,
  roles,
}: {
  children: ReactElement;
  roles?: UserRole[];
}) {
  const location = useLocation();
  const reduxUser = useAppSelector((s) => s.auth.user);

  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  const role = localStorage.getItem(AUTH_ROLE_KEY);
  const email = localStorage.getItem(AUTH_EMAIL_KEY);

  let user: User | null = null;
  
  if (reduxUser) {
    user = reduxUser;
  } else if (token && role && email) {
    user = {
      email,
      role: role.toLowerCase() as UserRole,
      id: undefined,
      name: undefined
    };
  }

  // If no user and not on login page, redirect to login
  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Check role authorization
  if (roles && roles.length > 0) {
    const userRole = user.role.toLowerCase();
    const hasRequiredRole = roles.some(r => r.toLowerCase() === userRole);
    if (!hasRequiredRole) {
      return <Navigate to="/" replace />;
    }
  }

  return children;
}

export function AppRoutes() {
  const location = useLocation();
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  const user = useAppSelector((s) => s.auth.user);

  // If user is on login page but already authenticated, redirect to dashboard
  if (location.pathname === '/login' && (token || user)) {
    return <Navigate to="/" replace />;
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        element={
          <Protected>
            <Layout />
          </Protected>
        }
      >
        <Route path="/" element={<Dashboard />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/damage" element={<DamageDetectionPage />} />
        <Route
          path="/models"
          element={
            <Protected roles={["admin"]}>
              <ModelManagementPage />
            </Protected>
          }
        />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}