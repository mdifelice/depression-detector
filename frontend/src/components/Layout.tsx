import { useTranslation } from "react-i18next";
import LanguageSelector from "./LanguageSelector";

export default function Layout({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  return (
    <div className="app-layout">
      <div className="topbar">
        <LanguageSelector />
      </div>
      {children}
      <footer className="app-footer">{t("footer.builtBy")}</footer>
    </div>
  );
}