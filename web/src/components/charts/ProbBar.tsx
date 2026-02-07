import React from 'react';

interface ProbBarProps {
  probability: number;
  threshold?: number;
  showLabel?: boolean;
}

/**
 * Probability Bar Component
 * Displays a horizontal bar showing probability with dark theme styling
 */
export const ProbBar: React.FC<ProbBarProps> = ({
  probability,
  threshold = 0.5,
  showLabel = true,
}) => {
  // Calculate color based on probability - using Stitch palette
  const getColor = (prob: number): string => {
    if (prob >= 0.6) return '#22c55e'; // Success green
    if (prob >= 0.4) return '#f97316'; // Warning orange
    return '#ef4444'; // Danger red
  };

  const percentage = Math.round(probability * 100);
  const fillColor = getColor(probability);
  const thresholdPosition = threshold * 100;

  return (
    <div className="w-full">
      {/* Bar container */}
      <div className="relative h-8 bg-gray-900 rounded-full overflow-hidden">
        {/* Fill */}
        <div
          className="absolute left-0 top-0 h-full transition-all duration-300 ease-out rounded-full"
          style={{
            width: `${percentage}%`,
            backgroundColor: fillColor,
          }}
        />

        {/* Threshold marker */}
        <div
          className="absolute top-0 w-0.5 h-full bg-slate-300"
          style={{ left: `${thresholdPosition}%` }}
        />

        {/* Threshold label */}
        <div
          className="absolute -top-5 text-xs text-slate-400 transform -translate-x-1/2"
          style={{ left: `${thresholdPosition}%` }}
        >
          {Math.round(threshold * 100)}%
        </div>
      </div>

      {/* Labels */}
      {showLabel && (
        <div className="flex justify-between mt-2 text-sm">
          <span className="text-slate-400">No Contact</span>
          <span className="font-mono font-bold text-lg" style={{ color: fillColor }}>
            {percentage}%
          </span>
          <span className="text-slate-400">Contact</span>
        </div>
      )}
    </div>
  );
};

/**
 * Probability Gauge Component
 * Displays a semicircular gauge showing probability with dark theme
 */
export const ProbGauge: React.FC<ProbBarProps> = ({
  probability,
  threshold = 0.5,
}) => {
  const percentage = Math.round(probability * 100);
  const angle = probability * 180; // 0 to 180 degrees

  const getColor = (prob: number): string => {
    if (prob >= 0.6) return '#22c55e';
    if (prob >= 0.4) return '#f97316';
    return '#ef4444';
  };

  const fillColor = getColor(probability);

  // SVG parameters
  const size = 200;
  const strokeWidth = 20;
  const radius = (size - strokeWidth) / 2;
  const circumference = Math.PI * radius;
  const progress = (angle / 180) * circumference;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 20} className="overflow-visible">
        {/* Background arc */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2}
              A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke="#1f2937"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Progress arc */}
        <path
          d={`M ${strokeWidth / 2} ${size / 2}
              A ${radius} ${radius} 0 0 1 ${size - strokeWidth / 2} ${size / 2}`}
          fill="none"
          stroke={fillColor}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={`${progress} ${circumference}`}
          className="transition-all duration-500 ease-out"
        />

        {/* Threshold marker */}
        <circle
          cx={size / 2 + radius * Math.cos(Math.PI - threshold * Math.PI)}
          cy={size / 2 - radius * Math.sin(threshold * Math.PI)}
          r="4"
          fill="#19c3e6"
        />

        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2 - 10}
          textAnchor="middle"
          className="fill-current"
          style={{ fill: fillColor }}
          fontSize="32"
          fontWeight="bold"
          fontFamily="JetBrains Mono, monospace"
        >
          {percentage}%
        </text>
        <text
          x={size / 2}
          y={size / 2 + 15}
          textAnchor="middle"
          fill="#94a3b8"
          fontSize="14"
        >
          Contact Probability
        </text>
      </svg>
    </div>
  );
};

export default ProbBar;
