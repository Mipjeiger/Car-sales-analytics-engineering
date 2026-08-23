import { apiClient } from "@/api/client";
import type { DamageDetectResponse, SimilarDamageCase } from "@/types/api.types";

const SEVERITY_MAP: Record<string, DamageDetectResponse["severity"]> = {
  none: "None",
  no_damage: "None",
  low: "Low",
  "01-minor": "Low",
  bumper_scrape: "Low",
  door_scratch: "Low",
  medium: "Medium",
  "02-moderate": "Medium",
  dent: "Medium",
  head_lamp: "Medium",
  high: "High",
  "03-severe": "High",
  broken_headlight: "High",
};

export function mapSeverity(damageType: string): DamageDetectResponse["severity"] {
  return SEVERITY_MAP[damageType.toLowerCase()] ?? "Medium";
}

export function qaFromSeverity(severity: DamageDetectResponse["severity"]): DamageDetectResponse["qa_status"] {
  if (severity === "None" || severity === "Low") return "Pass";
  if (severity === "Medium") return "Rework";
  return "Fail";
}

export const damageApi = {
  detect: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const { data } = await apiClient.post<DamageDetectResponse>("/damage/detect", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },
  similar: async (file: File, k = 5) => {
    const form = new FormData();
    form.append("file", file);
    const { data } = await apiClient.post<{ cases: SimilarDamageCase[] }>("/damage/similar", form, {
      params: { k },
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },
};
