import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell
} from 'recharts';

interface EvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
}

interface PerClassMetrics {
  [key: string]: {
    support: number;
    precision: number;
    recall: number;
    f1: number;
  };
}

interface FeatureImportance {
  feature: string;
  importance: number;
  importance_pct: number;
}

interface EvaluationData {
  contact_prediction: {
    model_name: string;
    test_samples: number;
    metrics: EvaluationMetrics;
    confusion_matrix: {
      labels: string[];
      matrix: number[][];
    };
    feature_importance: FeatureImportance[];
  };
  hit_outcome: {
    model_name: string;
    test_samples: number;
    metrics: {
      accuracy: number;
      f1_macro: number;
      f1_weighted: number;
    };
    per_class_metrics: PerClassMetrics;
    confusion_matrix: {
      labels: string[];
      matrix: number[][];
    };
    feature_importance: FeatureImportance[];
  };
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const MetricCard: React.FC<{ label: string; value: string | number; subtext?: string }> = ({
  label,
  value,
  subtext,
}) => (
  <div className="card">
    <div className="text-sm text-slate-400 mb-1">{label}</div>
    <div className="text-2xl font-bold text-white">{value}</div>
    {subtext && <div className="text-xs text-slate-500 mt-1">{subtext}</div>}
  </div>
);

const ConfusionMatrix: React.FC<{ labels: string[]; matrix: number[][] }> = ({
  labels,
  matrix,
}) => {
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="p-2 text-slate-400"></th>
            {labels.map((label) => (
              <th key={label} className="p-2 text-slate-400 text-center">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-slate-400 font-medium">{labels[i]}</td>
              {row.map((val, j) => {
                const intensity = val / maxVal;
                const isCorrect = i === j;
                return (
                  <td
                    key={j}
                    className="p-2 text-center font-mono"
                    style={{
                      backgroundColor: isCorrect
                        ? `rgba(26, 163, 176, ${0.2 + intensity * 0.6})`
                        : `rgba(239, 68, 68, ${intensity * 0.4})`,
                      color: intensity > 0.5 ? 'white' : 'inherit',
                    }}
                  >
                    {val.toLocaleString()}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export const PerformancePage: React.FC = () => {
  const [data, setData] = useState<EvaluationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeModel, setActiveModel] = useState<'contact' | 'outcome'>('contact');

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`${API_URL}/api/evaluation`);
        if (!response.ok) throw new Error('Failed to fetch evaluation data');
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-slate-400">Loading evaluation data...</div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-red-400">{error || 'No data available'}</div>
      </div>
    );
  }

  const contactMetrics = data.contact_prediction.metrics;
  const outcomeMetrics = data.hit_outcome.metrics;

  const contactFeatures = data.contact_prediction.feature_importance.slice(0, 10);
  const outcomeFeatures = data.hit_outcome.feature_importance.slice(0, 10);

  const COLORS = ['#1aa3b0', '#2a9df4', '#a855f7', '#f97316', '#22c55e'];

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Model Performance</h1>
          <p className="text-slate-400">
            Evaluation metrics for Contact and Hit Outcome prediction models
          </p>
        </div>

        {/* Model Selector */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveModel('contact')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeModel === 'contact'
                ? 'bg-primary text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Contact Model
          </button>
          <button
            onClick={() => setActiveModel('outcome')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              activeModel === 'outcome'
                ? 'bg-primary text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Outcome Model
          </button>
        </div>

        {/* Contact Model View */}
        {activeModel === 'contact' && (
          <>
            {/* Metrics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
              <MetricCard
                label="Accuracy"
                value={`${(contactMetrics.accuracy * 100).toFixed(1)}%`}
              />
              <MetricCard
                label="ROC-AUC"
                value={contactMetrics.roc_auc.toFixed(3)}
              />
              <MetricCard
                label="Precision"
                value={contactMetrics.precision.toFixed(3)}
              />
              <MetricCard
                label="Recall"
                value={contactMetrics.recall.toFixed(3)}
              />
              <MetricCard
                label="F1 Score"
                value={contactMetrics.f1_score.toFixed(3)}
              />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Confusion Matrix */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">Confusion Matrix</h3>
                <ConfusionMatrix
                  labels={data.contact_prediction.confusion_matrix.labels}
                  matrix={data.contact_prediction.confusion_matrix.matrix}
                />
                <p className="text-xs text-slate-500 mt-4">
                  Test samples: {data.contact_prediction.test_samples.toLocaleString()}
                </p>
              </div>

              {/* Feature Importance */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={contactFeatures}
                    layout="vertical"
                    margin={{ left: 100, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      stroke="#9ca3af"
                      width={90}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                      formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Importance']}
                    />
                    <Bar dataKey="importance_pct" fill="#1aa3b0" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}

        {/* Outcome Model View */}
        {activeModel === 'outcome' && (
          <>
            {/* Metrics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              <MetricCard
                label="Accuracy"
                value={`${(outcomeMetrics.accuracy * 100).toFixed(1)}%`}
              />
              <MetricCard
                label="F1 Macro"
                value={outcomeMetrics.f1_macro.toFixed(3)}
              />
              <MetricCard
                label="F1 Weighted"
                value={outcomeMetrics.f1_weighted.toFixed(3)}
              />
              <MetricCard
                label="Test Samples"
                value={data.hit_outcome.test_samples.toLocaleString()}
              />
            </div>

            {/* Per-Class Metrics */}
            <div className="card mb-6">
              <h3 className="text-lg font-semibold text-white mb-4">Per-Class Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                {Object.entries(data.hit_outcome.per_class_metrics).map(([label, metrics], i) => (
                  <div
                    key={label}
                    className="p-3 rounded-lg"
                    style={{ backgroundColor: `${COLORS[i]}20` }}
                  >
                    <div className="text-sm font-medium capitalize" style={{ color: COLORS[i] }}>
                      {label.replace('_', ' ')}
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      P: {metrics.precision.toFixed(2)} | R: {metrics.recall.toFixed(2)}
                    </div>
                    <div className="text-xs text-slate-400">
                      F1: {metrics.f1.toFixed(2)} | n={metrics.support}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              {/* Confusion Matrix */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">Confusion Matrix</h3>
                <ConfusionMatrix
                  labels={data.hit_outcome.confusion_matrix.labels}
                  matrix={data.hit_outcome.confusion_matrix.matrix}
                />
              </div>

              {/* Feature Importance */}
              <div className="card">
                <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={outcomeFeatures}
                    layout="vertical"
                    margin={{ left: 120, right: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis type="number" stroke="#9ca3af" />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      stroke="#9ca3af"
                      width={110}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1f2937', border: 'none' }}
                      formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Importance']}
                    />
                    <Bar dataKey="importance_pct" radius={[0, 4, 4, 0]}>
                      {outcomeFeatures.map((_, index) => (
                        <Cell key={index} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PerformancePage;
