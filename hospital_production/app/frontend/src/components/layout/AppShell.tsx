import React, { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  Activity,
  Users,
  ClipboardList,
  BarChart2,
  UserCheck,
  Brain,
  MessageSquare,
  Database,
  TrendingUp,
  Menu,
  X,
  Hospital,
  ChevronRight,
} from "lucide-react";

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactNode;
}

const NAV_ITEMS: NavItem[] = [
  { path: "/", label: "Overview", icon: <Activity size={18} /> },
  { path: "/patients", label: "Patient Analytics", icon: <Users size={18} /> },
  { path: "/patient-management", label: "Patient Management", icon: <ClipboardList size={18} /> },
  { path: "/operations", label: "Hospital Operations", icon: <Hospital size={18} /> },
  { path: "/staff", label: "Staff Analytics", icon: <UserCheck size={18} /> },
  { path: "/models", label: "ML Models", icon: <Brain size={18} /> },
  { path: "/insights", label: "AI Insights", icon: <MessageSquare size={18} /> },
  { path: "/data", label: "Data Explorer", icon: <Database size={18} /> },
  { path: "/health", label: "System Health", icon: <TrendingUp size={18} /> },
];

export const Sidebar: React.FC<{
  collapsed: boolean;
  onToggle: () => void;
}> = ({ collapsed, onToggle }) => {
  const location = useLocation();

  return (
    <aside className={`sidebar${collapsed ? " sidebar--collapsed" : ""}`}>
      {/* Logo / toggle */}
      <div className="sidebar-header">
        {!collapsed && (
          <span className="sidebar-logo">
            <Hospital size={18} />
            <span>Health Care</span>
          </span>
        )}
        <button className="sidebar-toggle" onClick={onToggle} aria-label="Toggle sidebar">
          {collapsed ? <Menu size={18} /> : <X size={18} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV_ITEMS.map(({ path, label, icon }) => (
          <NavLink
            key={path}
            to={path}
            end={path === "/"}
            className={({ isActive }) =>
              `nav-item${isActive ? " nav-item--active" : ""}`
            }
            title={collapsed ? label : undefined}
          >
            <span className="nav-icon">{icon}</span>
            {!collapsed && <span className="nav-label">{label}</span>}
            {!collapsed && location.pathname === path && (
              <ChevronRight size={14} className="nav-chevron" />
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="sidebar-footer">
          <p className="sidebar-version">v1.0.0</p>
        </div>
      )}
    </aside>
  );
};

export const TopBar: React.FC<{ title: string }> = ({ title }) => (
  <header className="topbar">
    <h1 className="topbar-title">{title}</h1>
    <div className="topbar-actions">
      <span className="topbar-env">Hospital Production</span>
    </div>
  </header>
);

export const AppShell: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  const currentLabel =
    NAV_ITEMS.find(({ path }) =>
      path === "/"
        ? location.pathname === "/"
        : location.pathname.startsWith(path)
    )?.label ?? "Dashboard";

  return (
    <div className={`app-shell${collapsed ? " app-shell--collapsed" : ""}`}>
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <div className="main-area">
        <TopBar title={currentLabel} />
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
};

