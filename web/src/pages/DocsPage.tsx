import React, { useState } from 'react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface PipelineStep {
  icon: string;
  title: string;
  description: string;
}

const pipelineSteps: PipelineStep[] = [
  {
    icon: 'database',
    title: 'Data Sources',
    description: 'MLB Statcast data via pybaseball API. 1.5M+ pitches and 256K+ batted balls from 2023-2024 seasons.',
  },
  {
    icon: 'cleaning_services',
    title: 'Data Processing',
    description: 'Missing value imputation, outlier detection, and validation. Temporal train/val/test split.',
  },
  {
    icon: 'settings',
    title: 'Feature Engineering',
    description: '16 engineered features per model including location, movement, velocity, count, and matchup features.',
  },
  {
    icon: 'model_training',
    title: 'Model Training',
    description: 'LightGBM for contact prediction, XGBoost for hit outcome. SMOTE for class balancing.',
  },
  {
    icon: 'api',
    title: 'API Deployment',
    description: 'FastAPI backend with prediction endpoints. <50ms latency per prediction.',
  },
  {
    icon: 'web',
    title: 'Frontend UI',
    description: 'React + TypeScript + Vite. Interactive visualizations with Recharts.',
  },
];

const docSections = [
  {
    title: 'Data Sources',
    icon: 'storage',
    content: '2023-2024 MLB Statcast pitch-by-pitch and batted ball data',
    stats: '1.5M+ pitches',
  },
  {
    title: 'Feature Engineering',
    icon: 'build',
    content: '16 engineered features per model based on domain expertise',
    stats: '32 total features',
  },
  {
    title: 'Models',
    icon: 'psychology',
    content: 'LightGBM (Contact) + XGBoost (Outcome) classifiers',
    stats: '2 models',
  },
  {
    title: 'Deployment',
    icon: 'cloud_upload',
    content: 'FastAPI backend with React frontend',
    stats: '<50ms latency',
  },
];

const quickStats = [
  { label: 'Pitches Analyzed', value: '1.5M+' },
  { label: 'Batted Balls', value: '256K+' },
  { label: 'Model Accuracy', value: '85%+' },
  { label: 'API Latency', value: '<50ms' },
];

export const DocsPage: React.FC = () => {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const handleDownloadReport = async () => {
    setDownloading(true);
    setDownloadError(null);

    try {
      const response = await fetch(`${API_URL}/api/report/download`);

      if (!response.ok) {
        throw new Error('Failed to download report');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'Baseball_ML_Project_Report.pdf';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : 'Download failed');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-white mb-4">
            Baseball ML Documentation
          </h1>
          <p className="text-lg text-slate-400 mb-8 max-w-2xl mx-auto">
            Predicting contact and hit outcomes using MLB Statcast data with
            machine learning models
          </p>

          {/* Download Report Button */}
          <button
            onClick={handleDownloadReport}
            disabled={downloading}
            className="inline-flex items-center gap-3 px-8 py-4 bg-primary hover:bg-primary/80
                       disabled:bg-slate-600 disabled:cursor-not-allowed
                       text-white font-semibold rounded-xl transition-colors
                       shadow-lg shadow-primary/25"
          >
            <span className="material-symbols-rounded text-2xl">
              {downloading ? 'hourglass_empty' : 'download'}
            </span>
            <span className="text-lg">
              {downloading ? 'Downloading...' : 'Download Full Report (PDF)'}
            </span>
          </button>

          {downloadError && (
            <p className="mt-4 text-red-400 text-sm">{downloadError}</p>
          )}
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
          {quickStats.map((stat) => (
            <div key={stat.label} className="card text-center">
              <div className="text-2xl font-bold text-accent mb-1">{stat.value}</div>
              <div className="text-sm text-slate-400">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Pipeline Architecture */}
        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Pipeline Architecture
          </h2>
          <div className="relative">
            {/* Connection Line */}
            <div className="hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-slate-700 -translate-y-1/2 z-0" />

            <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
              {pipelineSteps.map((step, index) => (
                <div key={step.title} className="relative z-10">
                  <div className="card text-center h-full flex flex-col items-center">
                    <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-3">
                      <span className="material-symbols-rounded text-accent">
                        {step.icon}
                      </span>
                    </div>
                    <h3 className="text-sm font-semibold text-white mb-2">
                      {step.title}
                    </h3>
                    <p className="text-xs text-slate-400">{step.description}</p>
                  </div>
                  {index < pipelineSteps.length - 1 && (
                    <div className="hidden md:flex absolute top-1/2 -right-2 w-4 h-4 items-center justify-center z-20">
                      <span className="material-symbols-rounded text-accent text-sm">
                        arrow_forward
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Documentation Cards */}
        <div className="mb-12">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Project Overview
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {docSections.map((section) => (
              <div key={section.title} className="card hover:border-accent transition-colors">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="material-symbols-rounded text-accent">
                      {section.icon}
                    </span>
                  </div>
                  <div>
                    <h3 className="font-semibold text-white mb-1">{section.title}</h3>
                    <p className="text-sm text-slate-400 mb-2">{section.content}</p>
                    <span className="text-xs font-medium text-accent">{section.stats}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          {/* Contact Model */}
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-rounded text-accent">sports_baseball</span>
              Contact Prediction Model
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Algorithm</span>
                <span className="text-white">LightGBM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Type</span>
                <span className="text-white">Binary Classification</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Training Samples</span>
                <span className="text-white">1.1M+</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Features</span>
                <span className="text-white">16</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">ROC-AUC</span>
                <span className="text-accent font-medium">0.805</span>
              </div>
            </div>
          </div>

          {/* Outcome Model */}
          <div className="card">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <span className="material-symbols-rounded text-accent">query_stats</span>
              Hit Outcome Model
            </h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Algorithm</span>
                <span className="text-white">XGBoost</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Type</span>
                <span className="text-white">Multi-class (5 classes)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Training Samples</span>
                <span className="text-white">658K (SMOTE)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Features</span>
                <span className="text-white">16</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Accuracy</span>
                <span className="text-accent font-medium">85.4%</span>
              </div>
            </div>
          </div>
        </div>

        {/* API Reference */}
        <div className="card mb-12">
          <h3 className="text-lg font-semibold text-white mb-4">API Endpoints</h3>
          <div className="space-y-4 font-mono text-sm">
            <div className="p-3 bg-slate-800 rounded-lg">
              <span className="text-green-400">GET</span>
              <span className="text-white ml-2">/health</span>
              <span className="text-slate-400 ml-4">- Health check and model status</span>
            </div>
            <div className="p-3 bg-slate-800 rounded-lg">
              <span className="text-blue-400">POST</span>
              <span className="text-white ml-2">/api/predict/contact</span>
              <span className="text-slate-400 ml-4">- Contact probability prediction</span>
            </div>
            <div className="p-3 bg-slate-800 rounded-lg">
              <span className="text-blue-400">POST</span>
              <span className="text-white ml-2">/api/predict/outcome</span>
              <span className="text-slate-400 ml-4">- Hit outcome classification</span>
            </div>
            <div className="p-3 bg-slate-800 rounded-lg">
              <span className="text-green-400">GET</span>
              <span className="text-white ml-2">/api/evaluation</span>
              <span className="text-slate-400 ml-4">- Model evaluation metrics</span>
            </div>
            <div className="p-3 bg-slate-800 rounded-lg">
              <span className="text-green-400">GET</span>
              <span className="text-white ml-2">/api/report/download</span>
              <span className="text-slate-400 ml-4">- Download PDF report</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-slate-500 text-sm">
          <p>Stitch Project ID: 9081103112165199837</p>
          <p className="mt-1">Built with FastAPI + React + Stitch</p>
        </div>
      </div>
    </div>
  );
};

export default DocsPage;
