/**
 * Prediction API types and functions
 */

import { apiClient } from './client';

// Contact Prediction Types
export interface ContactPredictionRequest {
  pitch_type: string;
  release_speed: number;
  pfx_x: number;
  pfx_z: number;
  plate_x: number;
  plate_z: number;
  balls: number;
  strikes: number;
  stand: 'L' | 'R';
  p_throws: 'L' | 'R';
}

export interface FeatureContribution {
  name: string;
  contribution: number;
}

export interface ContactPredictionResponse {
  prob_contact: number;
  threshold: number;
  predicted: 0 | 1;
  top_features?: FeatureContribution[];
  request_id: string;
  latency_ms: number;
}

// Outcome Prediction Types
export interface OutcomePredictionRequest {
  launch_speed: number;
  launch_angle: number;
  spray_angle?: number;
  stand?: 'L' | 'R';
  sprint_speed?: number;
}

export interface OutcomeProbabilities {
  out: number;
  single: number;
  double: number;
  triple: number;
  home_run: number;
}

export interface OutcomePredictionResponse {
  probs: OutcomeProbabilities;
  predicted: string;
  xwoba?: number;
  request_id: string;
  latency_ms: number;
}

// Health Check Types
export interface HealthResponse {
  status: string;
  models_loaded: {
    contact: boolean;
    outcome: boolean;
  };
  version: string;
}

// API Functions
export async function predictContact(
  data: ContactPredictionRequest
): Promise<ContactPredictionResponse> {
  return apiClient.post<ContactPredictionResponse>('/api/predict/contact', data);
}

export async function predictOutcome(
  data: OutcomePredictionRequest
): Promise<OutcomePredictionResponse> {
  return apiClient.post<OutcomePredictionResponse>('/api/predict/outcome', data);
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiClient.get<HealthResponse>('/health');
}

// Pitch Types
export const PITCH_TYPES = [
  { value: 'FF', label: 'Four-Seam Fastball' },
  { value: 'SI', label: 'Sinker' },
  { value: 'FC', label: 'Cutter' },
  { value: 'SL', label: 'Slider' },
  { value: 'CU', label: 'Curveball' },
  { value: 'CH', label: 'Changeup' },
  { value: 'FS', label: 'Splitter' },
] as const;

// Outcome Labels and Colors - Stitch palette
export const OUTCOME_COLORS: Record<string, string> = {
  out: '#6b7280',      // Gray
  single: '#19c3e6',   // Cyan (accent)
  double: '#2A9DF4',   // Electric blue (predicted)
  triple: '#a855f7',   // Purple
  home_run: '#f97316', // Orange
};

export const OUTCOME_LABELS: Record<string, string> = {
  out: 'Out',
  single: 'Single',
  double: 'Double',
  triple: 'Triple',
  home_run: 'Home Run',
};
