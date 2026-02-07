import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { StrikeZonePlot } from '../components/charts/StrikeZonePlot';
import { ProbBar } from '../components/charts/ProbBar';
import { predictContact, PITCH_TYPES } from '../api/predict';
import type { ContactPredictionRequest, ContactPredictionResponse } from '../api/predict';

// Form validation schema
const contactSchema = z.object({
  pitch_type: z.string(),
  release_speed: z.number().min(50).max(105),
  pfx_x: z.number().min(-25).max(25),
  pfx_z: z.number().min(-25).max(25),
  plate_x: z.number().min(-2.5).max(2.5),
  plate_z: z.number().min(0).max(5),
  balls: z.number().min(0).max(3),
  strikes: z.number().min(0).max(2),
  stand: z.enum(['L', 'R']),
  p_throws: z.enum(['L', 'R']),
});

type ContactFormData = z.infer<typeof contactSchema>;

const defaultValues: ContactFormData = {
  pitch_type: 'FF',
  release_speed: 92,
  pfx_x: 0,
  pfx_z: 10,
  plate_x: 0,
  plate_z: 2.5,
  balls: 0,
  strikes: 0,
  stand: 'R',
  p_throws: 'R',
};

// Presets for quick input
const PRESETS = [
  { name: 'Fastball Middle', values: { ...defaultValues, release_speed: 95, plate_x: 0, plate_z: 2.5 } },
  { name: 'Breaking Ball Corner', values: { ...defaultValues, pitch_type: 'SL', release_speed: 82, pfx_x: 5, pfx_z: 0, plate_x: 0.8, plate_z: 1.8 } },
  { name: 'Two-Strike Count', values: { ...defaultValues, strikes: 2, balls: 1 } },
];

export const ContactPage: React.FC = () => {
  const [result, setResult] = useState<ContactPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [threshold, setThreshold] = useState(0.5);

  const {
    register,
    handleSubmit,
    watch,
    reset,
    formState: { errors },
  } = useForm<ContactFormData>({
    resolver: zodResolver(contactSchema),
    defaultValues,
  });

  const plateX = watch('plate_x');
  const plateZ = watch('plate_z');

  const onSubmit = async (data: ContactFormData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await predictContact(data as ContactPredictionRequest);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const applyPreset = (preset: typeof PRESETS[number]) => {
    reset(preset.values);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Contact Prediction</h1>
        <p className="text-slate-400">
          Predict whether a batter will make contact based on pitch characteristics
        </p>
      </div>

      {/* Presets */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {PRESETS.map((preset) => (
          <button
            key={preset.name}
            onClick={() => applyPreset(preset)}
            className="btn-secondary text-sm py-2 px-4"
          >
            {preset.name}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-700 text-red-400 px-4 py-3 rounded-lg mb-6">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <div className="card">
          <h2 className="text-xl font-semibold text-white mb-4">Pitch Parameters</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {/* Pitch Type */}
            <div>
              <label className="input-label">Pitch Type</label>
              <select {...register('pitch_type')} className="input-field">
                {PITCH_TYPES.map((pt) => (
                  <option key={pt.value} value={pt.value}>
                    {pt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Velocity */}
            <div>
              <label className="input-label">Velocity (mph)</label>
              <input
                type="number"
                step="0.5"
                {...register('release_speed', { valueAsNumber: true })}
                className="input-field"
              />
              {errors.release_speed && (
                <p className="input-error">{errors.release_speed.message}</p>
              )}
            </div>

            {/* Movement */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="input-label">H Movement (in)</label>
                <input
                  type="number"
                  step="0.5"
                  {...register('pfx_x', { valueAsNumber: true })}
                  className="input-field"
                />
              </div>
              <div>
                <label className="input-label">V Movement (in)</label>
                <input
                  type="number"
                  step="0.5"
                  {...register('pfx_z', { valueAsNumber: true })}
                  className="input-field"
                />
              </div>
            </div>

            {/* Location */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="input-label">H Location (ft)</label>
                <input
                  type="number"
                  step="0.1"
                  {...register('plate_x', { valueAsNumber: true })}
                  className="input-field"
                />
              </div>
              <div>
                <label className="input-label">V Location (ft)</label>
                <input
                  type="number"
                  step="0.1"
                  {...register('plate_z', { valueAsNumber: true })}
                  className="input-field"
                />
              </div>
            </div>

            {/* Count */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="input-label">Balls</label>
                <select {...register('balls', { valueAsNumber: true })} className="input-field">
                  {[0, 1, 2, 3].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="input-label">Strikes</label>
                <select {...register('strikes', { valueAsNumber: true })} className="input-field">
                  {[0, 1, 2].map((n) => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Handedness */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="input-label">Batter</label>
                <select {...register('stand')} className="input-field">
                  <option value="L">Left</option>
                  <option value="R">Right</option>
                </select>
              </div>
              <div>
                <label className="input-label">Pitcher</label>
                <select {...register('p_throws')} className="input-field">
                  <option value="L">Left</option>
                  <option value="R">Right</option>
                </select>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? 'Predicting...' : 'Predict Contact'}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {/* Strike Zone */}
          <div className="card">
            <h2 className="text-xl font-semibold text-white mb-4">Strike Zone</h2>
            <StrikeZonePlot
              plateX={plateX}
              plateZ={plateZ}
              probability={result?.prob_contact}
            />
          </div>

          {/* Prediction Result */}
          {result && (
            <div className="card">
              <h2 className="text-xl font-semibold text-white mb-4">Prediction</h2>

              {/* Prediction Badge */}
              <div className="flex items-center justify-center mb-4">
                <span
                  className={`px-6 py-2 rounded-full text-lg font-bold ${
                    result.predicted === 1
                      ? 'bg-green-900/50 text-green-400 border border-green-700'
                      : 'bg-red-900/50 text-red-400 border border-red-700'
                  }`}
                >
                  {result.predicted === 1 ? 'CONTACT' : 'NO CONTACT'}
                </span>
              </div>

              {/* Probability Bar */}
              <ProbBar probability={result.prob_contact} threshold={threshold} />

              {/* Threshold Slider */}
              <div className="mt-4">
                <label className="input-label">
                  Threshold: {Math.round(threshold * 100)}%
                </label>
                <input
                  type="range"
                  min="10"
                  max="90"
                  value={threshold * 100}
                  onChange={(e) => setThreshold(Number(e.target.value) / 100)}
                  className="w-full accent-accent"
                />
              </div>

              {/* Latency */}
              <p className="text-sm text-slate-500 mt-4">
                Latency: {result.latency_ms.toFixed(1)}ms
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ContactPage;
