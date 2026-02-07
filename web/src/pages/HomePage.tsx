import React from 'react';
import { Link } from 'react-router-dom';

export const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section with Blueprint Background */}
      <section className="blueprint-bg py-16 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Baseball Analytics
          </h1>
          <p className="text-lg text-slate-400 mb-8 max-w-2xl mx-auto">
            Machine learning models for predicting pitch contact and batted ball outcomes
            using MLB Statcast data
          </p>
        </div>
      </section>

      {/* Model Cards */}
      <section className="py-12 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl font-semibold text-white mb-6">
            Prediction Models
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Contact Model Card */}
            <Link to="/contact" className="block">
              <div className="card hover:border-accent transition-colors cursor-pointer h-full">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="material-symbols-rounded text-accent">sports_baseball</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Contact Prediction
                    </h3>
                    <p className="text-slate-400 text-sm mb-4">
                      Predict whether a batter will make contact based on pitch characteristics
                      including velocity, movement, and location.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Pitch Type
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Velocity
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Location
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Count
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>

            {/* Outcome Model Card */}
            <Link to="/outcome" className="block">
              <div className="card hover:border-accent transition-colors cursor-pointer h-full">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="material-symbols-rounded text-accent">query_stats</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Hit Outcome
                    </h3>
                    <p className="text-slate-400 text-sm mb-4">
                      Predict the outcome type (out, single, double, triple, home run) based on
                      batted ball characteristics.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Exit Velocity
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Launch Angle
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Spray Angle
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>

            {/* Performance Metrics Card */}
            <Link to="/performance" className="block">
              <div className="card hover:border-accent transition-colors cursor-pointer h-full">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="material-symbols-rounded text-accent">monitoring</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Model Performance
                    </h3>
                    <p className="text-slate-400 text-sm mb-4">
                      View detailed evaluation metrics, confusion matrices, and feature
                      importance for both models.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Accuracy
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        ROC-AUC
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Confusion Matrix
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>

            {/* Documentation Card */}
            <Link to="/docs" className="block">
              <div className="card hover:border-accent transition-colors cursor-pointer h-full">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-lg bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <span className="material-symbols-rounded text-accent">description</span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-white mb-2">
                      Documentation
                    </h3>
                    <p className="text-slate-400 text-sm mb-4">
                      Explore the ML pipeline, architecture, and download the comprehensive
                      project report.
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        Pipeline
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        API Docs
                      </span>
                      <span className="px-2 py-1 text-xs rounded bg-slate-700 text-slate-300">
                        PDF Report
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Info Section */}
      <section className="py-12 px-6 border-t border-slate-700">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                Data Source
              </h4>
              <p className="text-slate-400 text-sm">
                MLB Statcast pitch-by-pitch and batted ball data from 2023-2024 seasons
              </p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                Models
              </h4>
              <p className="text-slate-400 text-sm">
                XGBoost classifiers trained on engineered features with calibrated probabilities
              </p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-accent uppercase tracking-wide mb-2">
                Metrics
              </h4>
              <p className="text-slate-400 text-sm">
                Expected wOBA calculations based on outcome probability distributions
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;
