import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { ContactPage } from './pages/ContactPage';
import { OutcomePage } from './pages/OutcomePage';
import { PerformancePage } from './pages/PerformancePage';
import { DocsPage } from './pages/DocsPage';
import './index.css';

function BottomNavigation() {
  return (
    <nav className="bottom-nav fixed bottom-0 left-0 right-0 z-50">
      <div className="flex justify-around items-center h-16 max-w-2xl mx-auto">
        <NavLink
          to="/"
          end
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <span className="material-symbols-rounded">home</span>
          <span className="label">Home</span>
        </NavLink>
        <NavLink
          to="/contact"
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <span className="material-symbols-rounded">sports_baseball</span>
          <span className="label">Contact</span>
        </NavLink>
        <NavLink
          to="/outcome"
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <span className="material-symbols-rounded">query_stats</span>
          <span className="label">Outcome</span>
        </NavLink>
        <NavLink
          to="/performance"
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <span className="material-symbols-rounded">monitoring</span>
          <span className="label">Metrics</span>
        </NavLink>
        <NavLink
          to="/docs"
          className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
        >
          <span className="material-symbols-rounded">description</span>
          <span className="label">Docs</span>
        </NavLink>
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background pb-20">
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/contact" element={<ContactPage />} />
            <Route path="/outcome" element={<OutcomePage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/docs" element={<DocsPage />} />
          </Routes>
        </main>
        <BottomNavigation />
      </div>
    </BrowserRouter>
  );
}

export default App;
