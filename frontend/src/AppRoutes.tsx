import type { ReactElement } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
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
import type { UserRole } from "@/types/common.types";

function Protected({ children, roles }: { children: ReactElement; roles?: UserRole[] }) {
  const user = useAppSelector((s) => s.auth.user);
  if (!user) return <Navigate to="/login" replace />;
  if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />;
  return children;
}

export function AppRoutes() {
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
            <Protected roles={["Admin"]}>
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
