import React from 'react';
import { Filter, Layers, RefreshCw } from 'lucide-react';

export default function FilterPanel({
  categories, selectedCategory, setSelectedCategory,
  shops, selectedShop, setSelectedShop,
  timeframes, selectedTimeframe, setSelectedTimeframe
}) {
  return (
    <div className="mb-6 bg-white rounded-xl shadow-sm p-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between space-y-4 md:space-y-0">
        <div className="space-x-2 flex items-center flex-wrap gap-2">
          <span className="text-sm font-medium text-gray-500 flex items-center">
            <Filter className="w-4 h-4 mr-1" /> Filter by:
          </span>
          {categories.map((category) => (
            <button
              key={category.name}
              onClick={() => setSelectedCategory(category.name)}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                selectedCategory === category.name
                  ? 'bg-indigo-100 text-indigo-800 font-medium'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {category.label}
            </button>
          ))}
        </div>
        <div className="flex space-x-3 items-center">
          <span className="text-sm font-medium text-gray-500 mr-1">
            <Layers className="w-4 h-4 inline mr-1" /> View:
          </span>
          <select
            value={selectedShop}
            onChange={(e) => setSelectedShop(e.target.value)}
            className="bg-white border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {shops.map(shop => (
              <option key={shop.id} value={shop.id}>{shop.name}</option>
            ))}
          </select>
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value)}
            className="bg-white border border-gray-300 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {timeframes.map(t => (
              <option key={t.id} value={t.id}>{t.name}</option>
            ))}
          </select>
          <button className="p-2 rounded-lg hover:bg-gray-100">
            <RefreshCw className="h-4 w-4 text-gray-500" />
          </button>
        </div>
      </div>
    </div>
  );
}
