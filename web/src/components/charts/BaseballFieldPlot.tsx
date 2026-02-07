import React from 'react';

interface BaseballFieldPlotProps {
  sprayAngle?: number;
  launchAngle?: number;
  exitVelocity?: number;
  predictedOutcome?: string;
}

/**
 * Baseball Field Plot Component
 * SVG visualization of a baseball field with ball trajectory
 */
export const BaseballFieldPlot: React.FC<BaseballFieldPlotProps> = ({
  sprayAngle = 0,
  launchAngle = 15,
  exitVelocity = 90,
  predictedOutcome,
}) => {
  const width = 300;
  const height = 300;
  const centerX = width / 2;
  const centerY = height - 30;

  // Field dimensions (scaled)
  const infieldRadius = 90;
  const outfieldRadius = 200;

  // Calculate ball landing position based on spray angle and estimated distance
  // Distance is a simplified function of exit velocity and launch angle
  const estimatedDistance = Math.min(
    ((exitVelocity / 100) * (1 - Math.abs(launchAngle - 25) / 50)) * outfieldRadius,
    outfieldRadius - 10
  );

  // Convert spray angle to radians (0 = center, negative = pull, positive = oppo)
  const angleRad = ((90 - sprayAngle) * Math.PI) / 180;
  const ballX = centerX + Math.cos(angleRad) * estimatedDistance;
  const ballY = centerY - Math.sin(angleRad) * estimatedDistance;

  // Outcome color
  const getOutcomeColor = (outcome?: string): string => {
    switch (outcome) {
      case 'home_run':
        return '#f97316';
      case 'triple':
        return '#a855f7';
      case 'double':
        return '#2A9DF4';
      case 'single':
        return '#19c3e6';
      case 'out':
      default:
        return '#6b7280';
    }
  };

  const ballColor = getOutcomeColor(predictedOutcome);

  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={height} className="rounded-lg">
        {/* Dark background */}
        <rect width={width} height={height} fill="#111827" rx="8" />

        {/* Outfield grass */}
        <path
          d={`M ${centerX - outfieldRadius} ${centerY}
              A ${outfieldRadius} ${outfieldRadius} 0 0 1 ${centerX + outfieldRadius} ${centerY}
              L ${centerX} ${centerY} Z`}
          fill="#6B9B3C"
          opacity="0.8"
        />

        {/* Infield dirt */}
        <path
          d={`M ${centerX - infieldRadius} ${centerY}
              A ${infieldRadius} ${infieldRadius} 0 0 1 ${centerX + infieldRadius} ${centerY}
              L ${centerX} ${centerY} Z`}
          fill="#8B6914"
          opacity="0.8"
        />

        {/* Infield diamond */}
        <polygon
          points={`
            ${centerX},${centerY - 63}
            ${centerX + 45},${centerY - 18}
            ${centerX},${centerY + 27}
            ${centerX - 45},${centerY - 18}
          `}
          fill="#8B6914"
          stroke="#ffffff"
          strokeWidth="1"
          opacity="0.9"
        />

        {/* Foul lines */}
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX - outfieldRadius}
          y2={centerY}
          stroke="#ffffff"
          strokeWidth="2"
        />
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + outfieldRadius}
          y2={centerY}
          stroke="#ffffff"
          strokeWidth="2"
        />

        {/* Base paths */}
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX + 45}
          y2={centerY - 45}
          stroke="#ffffff"
          strokeWidth="1"
        />
        <line
          x1={centerX}
          y1={centerY}
          x2={centerX - 45}
          y2={centerY - 45}
          stroke="#ffffff"
          strokeWidth="1"
        />
        <line
          x1={centerX - 45}
          y1={centerY - 45}
          x2={centerX}
          y2={centerY - 90}
          stroke="#ffffff"
          strokeWidth="1"
        />
        <line
          x1={centerX + 45}
          y1={centerY - 45}
          x2={centerX}
          y2={centerY - 90}
          stroke="#ffffff"
          strokeWidth="1"
        />

        {/* Bases */}
        <rect
          x={centerX - 4}
          y={centerY - 8}
          width={8}
          height={8}
          fill="#ffffff"
          transform={`rotate(45, ${centerX}, ${centerY - 4})`}
        />
        <rect
          x={centerX + 41}
          y={centerY - 49}
          width={8}
          height={8}
          fill="#ffffff"
          transform={`rotate(45, ${centerX + 45}, ${centerY - 45})`}
        />
        <rect
          x={centerX - 49}
          y={centerY - 49}
          width={8}
          height={8}
          fill="#ffffff"
          transform={`rotate(45, ${centerX - 45}, ${centerY - 45})`}
        />
        <rect
          x={centerX - 4}
          y={centerY - 94}
          width={8}
          height={8}
          fill="#ffffff"
          transform={`rotate(45, ${centerX}, ${centerY - 90})`}
        />

        {/* Trajectory line */}
        <line
          x1={centerX}
          y1={centerY}
          x2={ballX}
          y2={ballY}
          stroke={ballColor}
          strokeWidth="2"
          strokeDasharray="4,4"
          opacity="0.6"
        />

        {/* Glow filter for ball */}
        <defs>
          <filter id="ballGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Ball landing spot */}
        <circle
          cx={ballX}
          cy={ballY}
          r="10"
          fill={ballColor}
          filter="url(#ballGlow)"
        />

        {/* Spray angle arc indicator */}
        <path
          d={`M ${centerX + 30} ${centerY}
              A 30 30 0 0 0 ${centerX + 30 * Math.cos(angleRad)} ${centerY - 30 * Math.sin(angleRad)}`}
          fill="none"
          stroke="#19c3e6"
          strokeWidth="2"
          opacity="0.5"
        />

        {/* Labels */}
        <text x={20} y={centerY - 10} fill="#94a3b8" fontSize="10">
          LF
        </text>
        <text x={centerX - 8} y={30} fill="#94a3b8" fontSize="10">
          CF
        </text>
        <text x={width - 35} y={centerY - 10} fill="#94a3b8" fontSize="10">
          RF
        </text>
      </svg>

      {/* Field info */}
      <div className="mt-2 text-xs text-slate-500">
        Spray Angle: {sprayAngle > 0 ? '+' : ''}{sprayAngle}°
      </div>
    </div>
  );
};

export default BaseballFieldPlot;
