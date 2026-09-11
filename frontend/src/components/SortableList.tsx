import { useState, useRef } from "react";
import { useTranslation } from "react-i18next";

interface SortableListProps {
  values: string[];
  onChange: (values: string[]) => void;
}

export default function SortableList({ values, onChange }: SortableListProps) {
  const { t } = useTranslation();
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const dragItem = useRef<number | null>(null);

  const handleDragStart = (index: number) => {
    dragItem.current = index;
    setDragIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (dragItem.current === null || dragItem.current === index) return;
    const next = [...values];
    const [removed] = next.splice(dragItem.current, 1);
    next.splice(index, 0, removed);
    dragItem.current = index;
    onChange(next);
  };

  const handleDragEnd = () => {
    dragItem.current = null;
    setDragIndex(null);
  };

  const moveUp = (index: number) => {
    if (index === 0) return;
    const next = [...values];
    [next[index - 1], next[index]] = [next[index], next[index - 1]];
    onChange(next);
  };

  const moveDown = (index: number) => {
    if (index === values.length - 1) return;
    const next = [...values];
    [next[index], next[index + 1]] = [next[index + 1], next[index]];
    onChange(next);
  };

  if (values.length === 0) {
    return <p className="sortable-empty">{t("configure.noValuesToSort")}</p>;
  }

  return (
    <ul className="sortable-list">
      {values.map((val, i) => (
        <li
          key={`${val}-${i}`}
          className={`sortable-item ${dragIndex === i ? "dragging" : ""}`}
          draggable
          onDragStart={() => handleDragStart(i)}
          onDragOver={(e) => handleDragOver(e, i)}
          onDragEnd={handleDragEnd}
        >
          <span className="sortable-grip">⠿</span>
          <span className="sortable-index">{i + 1}.</span>
          <span className="sortable-value">{val}</span>
          <span className="sortable-buttons">
            <button
              type="button"
              className="sort-btn"
              onClick={() => moveUp(i)}
              disabled={i === 0}
              title={t("configure.moveUp")}
            >
              ↑
            </button>
            <button
              type="button"
              className="sort-btn"
              onClick={() => moveDown(i)}
              disabled={i === values.length - 1}
              title={t("configure.moveDown")}
            >
              ↓
            </button>
          </span>
        </li>
      ))}
    </ul>
  );
}
