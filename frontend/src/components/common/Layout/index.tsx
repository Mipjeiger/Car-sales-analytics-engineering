import { NavLink, Outlet } from "react-router-dom";
import { NAV_ITEMS, AIRFLOW_URL, GRAFANA_URL, MLFLOW_URL } from "@/utils/constants";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { logout } from "@/store/slices/authSlice";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/utils/helpers";
import { useState } from "react";

export function Layout() {
  const user = useAppSelector((s) => s.auth.user);
  const dispatch = useAppDispatch();
  const { theme, toggle } = useTheme();
  const [open, setOpen] = useState(false);

  const items = NAV_ITEMS.filter((item) => !("adminOnly" in item && item.adminOnly) || user?.role === "Admin");

  return (
    <div className="min-h-screen md:grid md:grid-cols-[260px_1fr]">
      <aside
        className={cn(
          "z-30 border-r bg-hero-gradient text-slate-100 md:sticky md:top-0 md:h-screen",
          open ? "fixed inset-0" : "hidden md:flex md:flex-col",
        )}
        aria-label="Primary"
      >
        <div className="flex h-16 items-center justify-between px-5">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-indigo-200">Car Sales</p>
            <p className="font-semibold">Intelligence</p>
          </div>
          <button className="md:hidden" onClick={() => setOpen(false)} aria-label="Close menu">
            ✕
          </button>
        </div>
        <nav className="flex flex-1 flex-col gap-1 px-3 py-4">
          {items.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                cn(
                  "rounded-xl px-3 py-2.5 text-sm transition",
                  isActive ? "bg-white/15 text-white" : "text-indigo-100 hover:bg-white/10",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="space-y-2 px-4 py-5 text-xs text-indigo-200">
          <a className="block hover:text-white" href={MLFLOW_URL} target="_blank" rel="noreferrer">
            MLflow
          </a>
          <a className="block hover:text-white" href={AIRFLOW_URL} target="_blank" rel="noreferrer">
            Airflow
          </a>
          <a className="block hover:text-white" href={GRAFANA_URL} target="_blank" rel="noreferrer">
            Grafana
          </a>
        </div>
      </aside>

      <div className="min-w-0">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b px-4 backdrop-blur-xl md:px-8" style={{ borderColor: "var(--border)", background: "var(--glass)" }}>
          <button className="btn-ghost md:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
            Menu
          </button>
          <p className="hidden text-sm subtle md:block">Enterprise MLOps console</p>
          <div className="flex items-center gap-3">
            <button className="btn-ghost" onClick={toggle} aria-label="Toggle theme">
              {theme === "dark" ? "Light" : "Dark"}
            </button>
            <div className="text-right text-sm">
              <p className="font-medium">{user?.name}</p>
              <p className="subtle text-xs">{user?.role}</p>
            </div>
            <button className="btn-ghost" onClick={() => dispatch(logout())}>
              Sign out
            </button>
          </div>
        </header>
        <main className="px-4 py-6 md:px-8">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
