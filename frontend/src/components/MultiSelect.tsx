import { useState, useRef, useEffect } from "react";

interface MultiSelectProps {
  options: string[];
  selected: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
}

export default function MultiSelect({
  options,
  selected,
  onChange,
  placeholder = "Select values...",
}: MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleClickOutside = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("pointerdown", handleClickOutside);
    return () => document.removeEventListener("pointerdown", handleClickOutside);
  }, [open]);

  const toggle = (val: string) => {
    if (selected.includes(val)) {
      onChange(selected.filter((v) => v !== val));
    } else {
      onChange([...selected, val]);
    }
  };

  const remove = (val: string) => {
    onChange(selected.filter((v) => v !== val));
  };

  return (
    <div className="multi-select" ref={ref}>
      <button
        type="button"
        className="multi-select-trigger"
        onClick={() => setOpen(!open)}
      >
        <span>
          {selected.length > 0
            ? `${selected.length} selected`
            : placeholder}
        </span>
        <span className="multi-select-arrow">{open ? "▲" : "▼"}</span>
      </button>

      {selected.length > 0 && (
        <div className="multi-select-tags">
          {selected.map((val) => (
            <span key={val} className="tag">
              {val}
              <button type="button" onClick={() => remove(val)}>
                &times;
              </button>
            </span>
          ))}
        </div>
      )}

      {open && (
        <div className="multi-select-dropdown">
          <ul className="multi-select-options">
            {options.length === 0 && (
              <li className="multi-select-empty">No values available</li>
            )}
            {options.map((val) => (
              <li
                key={val}
                className={`multi-select-option ${
                  selected.includes(val) ? "selected" : ""
                }`}
                onClick={() => toggle(val)}
              >
                <span className="multi-select-check">
                  {selected.includes(val) ? "✓" : ""}
                </span>
                {val}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
