import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

export default function KPICard({ title, value, last, trend, category }) {
  const isUp = trend === 'up';
  const isPositive = (isUp && title !== 'Return Rate' && title !== 'Shrinkage') ||
                     (!isUp && (title === 'Return Rate' || title === 'Shrinkage'));

  const calculateChange = () => {
    if (last.includes('%') && value.includes('%')) {
      const lastVal = parseFloat(last.replace('%', ''));
      const currentVal = parseFloat(value.replace('%', ''));
      return (currentVal - lastVal).toFixed(1) + '%';
    }
    if (last.includes('$') && value.includes('$')) {
      const lastVal = parseFloat(last.replace('$', '').replace('K', '000'));
      const currentVal = parseFloat(value.replace('$', '').replace('K', '000'));
      return ((currentVal - lastVal) / lastVal * 100).toFixed(1) + '%';
    }
    const lastVal = parseFloat(last);
    const currentVal = parseFloat(value);
    return (!isNaN(lastVal) && !isNaN(currentVal)) ? ((currentVal - lastVal) / lastVal * 100).toFixed(1) + '%' : 'N/A';
  };

  const changeValue = calculateChange();
  const changePrefix = changeValue.startsWith('-') ? '' : '+';

  const getCategoryColor = () => {
    switch (category) {
      case 'profitability': return 'from-teal-500 to-emerald-500';
      case 'inventory': return 'from-blue-500 to-cyan-500';
      case 'sales': return 'from-violet-500 to-purple-500';
      case 'pricing': return 'from-amber-500 to-yellow-500';
      case 'customer': return 'from-rose-500 to-pink-500';
      case 'operations': return 'from-gray-500 to-slate-500';
      default: return 'from-indigo-500 to-blue-500';
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300">
      <div className={`h-1 bg-gradient-to-r ${getCategoryColor()}`}></div>
      <div className="p-6">
        <div className="flex justify-between items-start">
          <h3 className="text-gray-600 uppercase text-xs font-semibold tracking-wider">{title}</h3>
          <span className={`text-xs font-medium px-2 py-1 rounded-full ${isPositive ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {changePrefix}{changeValue}
          </span>
        </div>
        <p className="mt-2 text-3xl font-bold text-gray-900">{value}</p>
        <div className="flex items-center mt-4 text-sm">
          <span className="text-gray-500 mr-2">Previous: {last}</span>
          {isUp ? (
            <ArrowUp className={`w-4 h-4 ${isPositive ? 'text-green-500' : 'text-red-500'}`} />
          ) : (
            <ArrowDown className={`w-4 h-4 ${isPositive ? 'text-green-500' : 'text-red-500'}`} />
          )}
        </div>
      </div>
    </div>
  );
}
