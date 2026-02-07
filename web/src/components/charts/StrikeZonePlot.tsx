import React from 'react';

interface StrikeZonePlotProps {
  plateX: number;
  plateZ: number;
  probability?: number;
}

/**
 * Strike Zone Plot Component
 * Displays the pitch location relative to the strike zone
 * Dark theme with teal accents
 */
export const StrikeZonePlot: React.FC<StrikeZonePlotProps> = ({
  plateX,
  plateZ,
  probability,
}) => {
  // SVG dimensions
  const width = 250;
  const height = 300;
  const padding = 30;

  // Strike zone dimensions (in feet)
  const zoneWidth = 17 / 12; // 17 inches = 1.42 feet
  const zoneBottom = 1.5;
  const zoneTop = 3.5;
  const zoneHeight = zoneTop - zoneBottom;

  // Scale factors
  const xRange = 3; // -1.5 to 1.5 feet
  const yRange = 4; // 0.5 to 4.5 feet

  const scaleX = (x: number) =>
    padding + ((x + xRange / 2) / xRange) * (width - 2 * padding);
  const scaleY = (y: number) =>
    height - padding - ((y - 0.5) / yRange) * (height - 2 * padding);

  // Strike zone rectangle coordinates
  const zoneX = scaleX(-zoneWidth / 2);
  const zoneY = scaleY(zoneTop);
  const zoneW = scaleX(zoneWidth / 2) - zoneX;
  const zoneH = scaleY(zoneBottom) - zoneY;

  // Ball position
  const ballX = scaleX(plateX);
  const ballY = scaleY(plateZ);

  // Color based on probability - using Stitch palette
  const getColor = (prob?: number) => {
    if (prob === undefined) return '#2A9DF4'; // Predicted blue
    if (prob >= 0.6) return '#22c55e'; // Success green
    if (prob >= 0.4) return '#f97316'; // Warning orange
    return '#ef4444'; // Danger red
  };

  const ballColor = getColor(probability);

  // Grid lines for 3x3 zone breakdown
  const gridLines = [];
  for (let i = 1; i <= 2; i++) {
    // Vertical lines
    const vx = scaleX(-zoneWidth / 2 + (i * zoneWidth) / 3);
    gridLines.push(
      <line
        key={`v${i}`}
        x1={vx}
        y1={scaleY(zoneTop)}
        x2={vx}
        y2={scaleY(zoneBottom)}
        stroke="#374151"
        strokeWidth="1"
        strokeDasharray="4,4"
      />
    );
    // Horizontal lines
    const hy = scaleY(zoneBottom + (i * zoneHeight) / 3);
    gridLines.push(
      <line
        key={`h${i}`}
        x1={scaleX(-zoneWidth / 2)}
        y1={hy}
        x2={scaleX(zoneWidth / 2)}
        y2={hy}
        stroke="#374151"
        strokeWidth="1"
        strokeDasharray="4,4"
      />
    );
  }

  return (
    <div className="flex flex-col items-center">
      <svg width={width} height={height} className="rounded-lg">
        {/* Dark Background */}
        <rect width={width} height={height} fill="#111827" rx="8" />

        {/* Strike zone */}
        <rect
          x={zoneX}
          y={zoneY}
          width={zoneW}
          height={zoneH}
          fill="none"
          stroke="#19c3e6"
          strokeWidth="2"
        />

        {/* Grid lines */}
        {gridLines}

        {/* Home plate (simplified pentagon) */}
        <polygon
          points={`${scaleX(0)},${height - 10} ${scaleX(-0.35)},${height - 20} ${scaleX(-0.35)},${height - 25} ${scaleX(0.35)},${height - 25} ${scaleX(0.35)},${height - 20}`}
          fill="#1f2937"
          stroke="#94a3b8"
          strokeWidth="2"
        />

        {/* Glow filter for ball */}
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Ball location with glow */}
        <circle
          cx={ballX}
          cy={ballY}
          r="12"
          fill={ballColor}
          filter="url(#glow)"
          opacity="0.95"
        />

        {/* Probability label on ball */}
        {probability !== undefined && (
          <text
            x={ballX}
            y={ballY + 4}
            textAnchor="middle"
            fill="white"
            fontSize="10"
            fontWeight="bold"
          >
            {Math.round(probability * 100)}
          </text>
        )}

        {/* Axis labels */}
        <text x={width / 2} y={height - 2} textAnchor="middle" fill="#64748b" fontSize="10">
          Horizontal (ft)
        </text>
        <text
          x={10}
          y={height / 2}
          textAnchor="middle"
          fill="#64748b"
          fontSize="10"
          transform={`rotate(-90, 10, ${height / 2})`}
        >
          Height (ft)
        </text>
      </svg>

      {/* Catcher's Perspective Label */}
      <p className="text-xs text-slate-500 mt-2">Catcher's Perspective</p>

      {/* Legend */}
      <div className="mt-2 text-sm text-slate-400 flex gap-4">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-green-500 rounded-full" /> High
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-orange-500 rounded-full" /> Medium
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-3 bg-red-500 rounded-full" /> Low
        </span>
      </div>
    </div>
  );
};

export default StrikeZonePlot;
