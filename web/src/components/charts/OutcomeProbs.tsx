import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { OUTCOME_COLORS, OUTCOME_LABELS } from '../../api/predict';
import type { OutcomeProbabilities } from '../../api/predict';

interface OutcomeProbsProps {
  probs: OutcomeProbabilities;
  predicted?: string;
}

interface ChartData {
  name: string;
  label: string;
  probability: number;
  color: string;
}

/**
 * Outcome Probabilities Bar Chart
 * Displays probability distribution for hit outcomes with dark theme
 */
export const OutcomeProbs: React.FC<OutcomeProbsProps> = ({ probs, predicted }) => {
  // Convert probabilities to chart data
  const data: ChartData[] = Object.entries(probs)
    .map(([name, probability]) => ({
      name,
      label: OUTCOME_LABELS[name] || name,
      probability: probability * 100,
      color: OUTCOME_COLORS[name] || '#6b7280',
    }))
    .sort((a, b) => b.probability - a.probability);

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={220}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
        >
          <XAxis
            type="number"
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
            tick={{ fontSize: 12, fill: '#94a3b8' }}
            axisLine={{ stroke: '#374151' }}
            tickLine={{ stroke: '#374151' }}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fontSize: 12, fill: '#f8fafc' }}
            width={75}
            axisLine={{ stroke: '#374151' }}
            tickLine={{ stroke: '#374151' }}
          />
          <Tooltip
            formatter={(value) => [`${(value as number).toFixed(1)}%`, 'Probability']}
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#f8fafc',
            }}
            labelStyle={{ color: '#f8fafc' }}
            itemStyle={{ color: '#f8fafc' }}
          />
          <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color}
                opacity={entry.name === predicted ? 1 : 0.7}
                stroke={entry.name === predicted ? '#19c3e6' : 'none'}
                strokeWidth={entry.name === predicted ? 2 : 0}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Color Legend */}
      <div className="flex flex-wrap justify-center gap-3 mt-2 text-sm">
        {Object.entries(OUTCOME_COLORS).map(([key, color]) => (
          <div key={key} className="flex items-center">
            <div
              className="w-3 h-3 rounded mr-1"
              style={{ backgroundColor: color }}
            />
            <span className="text-slate-400">{OUTCOME_LABELS[key]}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default OutcomeProbs;
