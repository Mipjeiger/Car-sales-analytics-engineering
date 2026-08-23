import type { ReactNode } from "react";
import type { SearchResult } from "./api.types";

export interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  trend?: number;
  icon?: ReactNode;
}

export interface CarCardProps {
  car: SearchResult;
  selected?: boolean;
  onSelect?: (car: SearchResult) => void;
  onCompare?: (car: SearchResult) => void;
}

export interface ImageUploaderProps {
  onFile: (file: File) => void;
  title?: string;
  subtitle?: string;
  disabled?: boolean;
  previewUrl?: string | null;
}
