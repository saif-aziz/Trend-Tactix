// KPIDashboard.jsx
import React, { useState } from 'react';
import Header from './Header';
import FilterPanel from './FilterPanel';
import KPICard from './KPICard';
import NotificationCard from './NotificationCard';

import { metrics, alerts, categories, shops, timeframes } from './Data';

export default function KPIDashboard() {
  const [activeTab, setActiveTab] = useState('kpis');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedShop, setSelectedShop] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('monthly');

  const filteredMetrics = metrics.filter(m => selectedCategory === 'all' || m.category === selectedCategory);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      {activeTab === 'kpis' ? (
        <>
          <FilterPanel
            categories={categories}
            selectedCategory={selectedCategory}
            setSelectedCategory={setSelectedCategory}
            shops={shops}
            selectedShop={selectedShop}
            setSelectedShop={setSelectedShop}
            timeframes={timeframes}
            selectedTimeframe={selectedTimeframe}
            setSelectedTimeframe={setSelectedTimeframe}
          />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6">
            {filteredMetrics.map(m => (
              <KPICard key={m.title} {...m} />
            ))}
          </div>
        </>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 p-6">
          {alerts.map(a => <NotificationCard key={a.id} {...a} />)}
        </div>
      )}
    </div>
  );
}
