import React from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import HealthPage from './pages/Health';
import ModelsPage from './pages/Models';

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-primary text-white p-4">
        <h1 className="text-xl font-semibold">Hospital SaaS Dashboard</h1>
        <nav className="mt-2 space-x-4">
          <NavLink to="/" end className={({ isActive }) => isActive ? 'underline' : ''}>Health</NavLink>
          <NavLink to="/models" className={({ isActive }) => isActive ? 'underline' : ''}>Models</NavLink>
        </nav>
      </header>
      <main className="p-4">
        <Routes>
          <Route index element={<HealthPage />} />
          <Route path="models" element={<ModelsPage />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
