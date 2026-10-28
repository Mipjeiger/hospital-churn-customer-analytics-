import React from "react";
import { Routes, Route } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "./lib/queryClient";
import { AppShell } from "./components/layout/AppShell";

// Pages (lazy-loaded for code splitting)
import Overview from "./pages/Overview";
import PatientAnalytics from "./pages/PatientAnalytics";
import PatientManagement from "./pages/PatientManagement";
import HospitalOperations from "./pages/HospitalOperations";
import StaffAnalytics from "./pages/StaffAnalytics";
import MLModels from "./pages/MLModels";
import AIInsights from "./pages/AIInsights";
import DataExplorer from "./pages/DataExplorer";
import SystemHealth from "./pages/Health";

const App: React.FC = () => (
  <QueryClientProvider client={queryClient}>
    <AppShell>
      <Routes>
        <Route index element={<Overview />} />
        <Route path="patients" element={<PatientAnalytics />} />
        <Route path="patient-management" element={<PatientManagement />} />
        <Route path="operations" element={<HospitalOperations />} />
        <Route path="staff" element={<StaffAnalytics />} />
        <Route path="models" element={<MLModels />} />
        <Route path="insights" element={<AIInsights />} />
        <Route path="data" element={<DataExplorer />} />
        <Route path="health" element={<SystemHealth />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </AppShell>
  </QueryClientProvider>
);

export default App;
