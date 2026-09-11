import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import ConfigureDataset from "./pages/ConfigureDataset";
import TrainDataset from "./pages/TrainDataset";
import TrainingProgress from "./pages/TrainingProgress";
import SavedModels from "./pages/SavedModels";
import Predict from "./pages/Predict";
import Explain from "./pages/Explain";
import "./App.css";

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <p>Loading...</p>;
  if (!user) return <Navigate to="/login" />;
  return <>{children}</>;
}

function AppRoutes() {
  const { user, loading } = useAuth();
  if (loading) return <p>Loading...</p>;

  return (
    <Routes>
      <Route
        path="/login"
        element={user ? <Navigate to="/dashboard" /> : <Login />}
      />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/configure/:id"
        element={
          <ProtectedRoute>
            <ConfigureDataset />
          </ProtectedRoute>
        }
      />
      <Route
        path="/train/:id"
        element={
          <ProtectedRoute>
            <TrainDataset />
          </ProtectedRoute>
        }
      />
      <Route
        path="/results/:id"
        element={
          <ProtectedRoute>
            <TrainingProgress />
          </ProtectedRoute>
        }
      />
      <Route
        path="/trained-models"
        element={
          <ProtectedRoute>
            <SavedModels />
          </ProtectedRoute>
        }
      />
      <Route
        path="/predict/:id"
        element={
          <ProtectedRoute>
            <Predict />
          </ProtectedRoute>
        }
      />
      <Route
        path="/explain/:id"
        element={
          <ProtectedRoute>
            <Explain />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to="/dashboard" />} />
    </Routes>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;