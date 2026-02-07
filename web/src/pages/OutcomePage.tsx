import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { OutcomeProbs } from '../components/charts/OutcomeProbs';
import { BaseballFieldPlot } from '../components/charts/BaseballFieldPlot';
import { predictOutcome, OUTCOME_LABELS, OUTCOME_COLORS } from '../api/predict';
import type { OutcomePredictionRequest, OutcomePredictionResponse } from '../api/predict';

// Form validation schema
const outcomeSchema = z.object({
  launch_speed: z.number().min(30).max(120),
  launch_angle: z.number().min(-90).max(90),
  spray_angle: z.number().min(-45).max(45).optional(),
  stand: z.enum(['L', 'R']).optional(),
  sprint_speed: z.number().min(24).max(31).optional(),
});

type OutcomeFormData = z.infer<typeof outcomeSchema>;

const defaultValues: OutcomeFormData = {
  launch_speed: 90,
  launch_angle: 15,
  spray_angle: 0,
  stand: 'R',
  sprint_speed: 27,
};

// Presets for quick input
const PRESETS = [
  { name: 'Barrel', values: { ...defaultValues, launch_speed: 105, launch_angle: 28 } },
  { name: 'Line Drive', values: { ...defaultValues, launch_speed: 95, launch_angle: 12 } },
  { name: 'Fly Ball', values: { ...defaultValues, launch_speed: 85, launch_angle: 35 } },
  { name: 'Ground Ball', values: { ...defaultValues, launch_speed: 80, launch_angle: -5 } },
];

export const OutcomePage: React.FC = () => {
  const [result, setResult] = useState<OutcomePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    reset,
    formState: { errors },
  } = useForm<OutcomeFormData>({
    resolver: zodResolver(outcomeSchema),
    defaultValues,
  });

  const launchSpeed = watch('launch_speed');
  const launchAngle = watch('launch_angle');
  const sprayAngle = watch('spray_angle') ?? 0;

  const onSubmit = async (data: OutcomeFormData) => {
    setLoading(true);
    setError(null);

    try {
      const response = await predictOutcome(data as OutcomePredictionRequest);
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

  // Determine batted ball type based on launch angle
  const getBattedBallType = (angle: number): string => {
    if (angle < 10) return 'Ground Ball';
    if (angle > 25 && angle < 50) return 'Fly Ball';
    if (angle >= 50) return 'Pop Up';
    return 'Line Drive';
  };

  // Check if barrel zone
  const isBarrel = launchSpeed >= 98 && launchAngle >= 26 && launchAngle <= 30;

  // xwOBA interpretation
  const getXwobaInterpretation = (xwoba?: number): { label: string; color: string } => {
    if (xwoba === undefined) return { label: '', color: '' };
    if (xwoba >= 0.400) return { label: 'Elite', color: 'text-purple-400' };
    if (xwoba >= 0.350) return { label: 'Above Average', color: 'text-green-400' };
    if (xwoba >= 0.300) return { label: 'Average', color: 'text-slate-400' };
    return { label: 'Below Average', color: 'text-red-400' };
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">Hit Outcome Prediction</h1>
        <p className="text-slate-400">
          Predict the outcome type based on batted ball characteristics
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
          <h2 className="text-xl font-semibold text-white mb-4">Batted Ball Parameters</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {/* Exit Velocity */}
            <div>
              <label className="input-label">Exit Velocity (mph)</label>
              <input
                type="number"
                step="0.5"
                {...register('launch_speed', { valueAsNumber: true })}
                className="input-field"
              />
              {errors.launch_speed && (
                <p className="input-error">{errors.launch_speed.message}</p>
              )}
              <p className="text-xs text-slate-500 mt-1">
                Range: 30-120 mph • Hard hit: 95+ mph
              </p>
            </div>

            {/* Launch Angle */}
            <div>
              <label className="input-label">Launch Angle (°)</label>
              <input
                type="number"
                step="1"
                {...register('launch_angle', { valueAsNumber: true })}
                className="input-field"
              />
              {errors.launch_angle && (
                <p className="input-error">{errors.launch_angle.message}</p>
              )}
              <p className="text-xs text-slate-500 mt-1">
                Range: -90° to 90° • Sweet spot: 8-32°
              </p>
            </div>

            {/* Spray Angle */}
            <div>
              <label className="input-label">Spray Angle (°)</label>
              <input
                type="number"
                step="1"
                {...register('spray_angle', { valueAsNumber: true })}
                className="input-field"
              />
              <p className="text-xs text-slate-500 mt-1">
                Negative = pull • Positive = opposite field
              </p>
            </div>

            {/* Advanced Options */}
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-accent hover:underline"
            >
              {showAdvanced ? '− Hide' : '+ Show'} advanced options
            </button>

            {showAdvanced && (
              <div className="space-y-4 pt-2 border-t border-slate-700">
                <div>
                  <label className="input-label">Batter Handedness</label>
                  <select {...register('stand')} className="input-field">
                    <option value="L">Left</option>
                    <option value="R">Right</option>
                  </select>
                </div>
                <div>
                  <label className="input-label">Sprint Speed (ft/s)</label>
                  <input
                    type="number"
                    step="0.1"
                    {...register('sprint_speed', { valueAsNumber: true })}
                    className="input-field"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    Elite: 29+ • Average: 27
                  </p>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? 'Predicting...' : 'Predict Outcome'}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {/* Input Summary with Field */}
          <div className="card">
            <h2 className="text-xl font-semibold text-white mb-4">Input Summary</h2>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p className="text-sm text-slate-500">Exit Velocity</p>
                <p className="stat-value">{launchSpeed} mph</p>
              </div>
              <div>
                <p className="text-sm text-slate-500">Launch Angle</p>
                <p className="stat-value">{launchAngle}°</p>
              </div>
            </div>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-slate-400">Type:</span>
              <span className="font-medium text-white">{getBattedBallType(launchAngle)}</span>
              {isBarrel && (
                <span className="bg-orange-900/50 text-orange-400 border border-orange-700 px-2 py-0.5 rounded text-sm font-medium">
                  BARREL
                </span>
              )}
            </div>

            {/* Baseball Field Visualization */}
            <BaseballFieldPlot
              sprayAngle={sprayAngle}
              launchAngle={launchAngle}
              exitVelocity={launchSpeed}
              predictedOutcome={result?.predicted}
            />
          </div>

          {/* Prediction Result */}
          {result && (
            <>
              <div className="card">
                <h2 className="text-xl font-semibold text-white mb-4">Prediction</h2>

                {/* Predicted Outcome Badge */}
                <div className="flex items-center justify-center mb-6">
                  <span
                    className="px-6 py-3 rounded-lg text-xl font-bold text-white"
                    style={{ backgroundColor: OUTCOME_COLORS[result.predicted] }}
                  >
                    {OUTCOME_LABELS[result.predicted]}
                  </span>
                </div>

                {/* xwOBA */}
                {result.xwoba !== undefined && (
                  <div className="text-center mb-6">
                    <p className="text-sm text-slate-500">Expected wOBA</p>
                    <p className="font-mono text-3xl font-bold text-white">
                      {result.xwoba.toFixed(3)}
                    </p>
                    <p className={`text-sm font-medium ${getXwobaInterpretation(result.xwoba).color}`}>
                      {getXwobaInterpretation(result.xwoba).label}
                    </p>
                  </div>
                )}

                {/* Latency */}
                <p className="text-sm text-slate-500 text-center">
                  Latency: {result.latency_ms.toFixed(1)}ms
                </p>
              </div>

              {/* Probability Distribution */}
              <div className="card">
                <h2 className="text-xl font-semibold text-white mb-4">Probability Distribution</h2>
                <OutcomeProbs probs={result.probs} predicted={result.predicted} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default OutcomePage;
