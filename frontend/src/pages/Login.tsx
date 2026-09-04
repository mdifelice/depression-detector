import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || "";

export default function Login() {
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSuccess = async (credentialResponse: { credential?: string }) => {
    if (credentialResponse.credential) {
      await login(credentialResponse.credential);
      navigate("/dashboard");
    }
  };

  return (
    <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
      <div className="login-container">
        <h1>Depression Detector</h1>
        <p>Sign in with your Google account to continue.</p>
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={() => console.error("Login failed")}
        />
      </div>
    </GoogleOAuthProvider>
  );
}
