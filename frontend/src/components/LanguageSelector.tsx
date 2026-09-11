import { useTranslation } from "react-i18next";

export default function LanguageSelector() {
  const { i18n } = useTranslation();
  return (
    <select
      className="language-selector"
      value={i18n.resolvedLanguage}
      onChange={(e) => i18n.changeLanguage(e.target.value)}
      aria-label="Language"
    >
      <option value="en">English</option>
      <option value="es">Español</option>
    </select>
  );
}