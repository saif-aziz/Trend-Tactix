// File: src/Components/NotificationCard.jsx
import React from 'react';

export default function NotificationCard({ title, text, date, type }) {
  const borderColor = getBorderColor(type);

  return (
    <div className="bg-white p-4 rounded-lg shadow border-l-4" style={{ borderColor }}>
      <h3 className="text-lg font-semibold text-gray-800">{title}</h3>
      <p className="text-sm text-gray-600 mt-1">{text}</p>
      <p className="text-xs text-gray-400 mt-2">{date}</p>
    </div>
  );
}

function getBorderColor(type) {
  switch (type) {
    case 'critical': return '#ef4444';  // red
    case 'warning': return '#facc15';   // yellow
    case 'info': return '#3b82f6';      // blue
    case 'notice': return '#10b981';    // green
    case 'reminder': return '#8b5cf6';  // purple
    default: return '#d1d5db';          // gray
  }
}
