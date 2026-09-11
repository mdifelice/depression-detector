import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";
import { useAuth } from "../contexts/AuthContext";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import LanguageSelector from "../components/LanguageSelector";

const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || "";

export default function Login() {
  const { login } = useAuth();
  const { t } = useTranslation();
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
        <div className="login-topbar">
          <LanguageSelector />
        </div>
        <img src="/gemis.png" alt="Gemis" className="gemis-logo login-logo" />
        <h1>{t("login.title")}</h1>
        <p>{t("login.subtitle")}</p>
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={() => console.error("Login failed")}
        />
      </div>
    </GoogleOAuthProvider>
  );
}
