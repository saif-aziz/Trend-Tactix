// Updated KPIDashboard.jsx using centralized API configuration
import React, { useState, useEffect } from 'react';
import Header from './Header';
import FilterPanel from './FilterPanel';
import KPICard from './KPICard';
import NotificationCard from './NotificationCard';

// Import centralized API configuration
import { api, testConnection } from '../config/api';

// Import static data as fallback
import { categories, shops, timeframes } from './Data';

// Fallback data in case API is not available
const fallbackMetrics = [
  { title: 'GMROI', value: '5.8', last: '5.5', trend: 'up', category: 'profitability' },
  { title: 'Gross Margin', value: '42%', last: '45%', trend: 'down', category: 'profitability' },
  { title: 'Inventory Turnover', value: '3.2', last: '3.0', trend: 'up', category: 'inventory' },
  { title: 'Weeks of Stock', value: '8', last: '9', trend: 'down', category: 'inventory' },
  { title: 'Markdown', value: '10%', last: '12%', trend: 'up', category: 'pricing' },
  { title: 'Return Rate', value: '5%', last: '4%', trend: 'down', category: 'customer' },
  { title: 'Shrinkage', value: '2%', last: '1.8%', trend: 'down', category: 'operations' },
  { title: 'Avg Monthly Sales', value: '$125K', last: '$120K', trend: 'up', category: 'sales' },
  { title: 'Avg Sell Thru', value: '60%', last: '58%', trend: 'up', category: 'inventory' },
  { title: 'Avg Basket Value', value: '$45', last: '$50', trend: 'down', category: 'sales' },
  { title: 'Avg Invoice Value', value: '$150', last: '$140', trend: 'up', category: 'sales' },
  { title: 'Stock to Sales Ratio', value: '3.5', last: '3.8', trend: 'down', category: 'inventory' },
];

const fallbackAlerts = [
  { id: 1, title: 'Variants Out Of Stock!!!', text: '22 of your active variants are out of stock.', date: 'Mar 12, 2025 • 07:40 AM', type: 'critical' },
  { id: 2, title: 'Stock Reallocation Needed', text: 'Stock levels are unbalanced across stores.', date: 'Mar 10, 2025 • 06:30 PM', type: 'warning' },
  { id: 3, title: 'New Discount Strategy Suggested', text: 'Competitor sales detected.', date: 'Mar 8, 2025 • 03:15 PM', type: 'info' },
  { id: 4, title: 'Excess Stock Warning', text: 'Inventory levels are higher than required.', date: 'Mar 7, 2025 • 01:45 PM', type: 'notice' },
  { id: 5, title: 'Restock Reminder', text: 'Reorder threshold reached for multiple products.', date: 'Mar 5, 2025 • 09:20 AM', type: 'reminder' },
];

// Helper functions
function formatCurrency(amount) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(amount || 0);
}

function formatNumber(number) {
  return new Intl.NumberFormat().format(number || 0);
}

export default function KPIDashboard() {
  const [activeTab, setActiveTab] = useState('kpis');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedShop, setSelectedShop] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('monthly');
  const [selectedYear, setSelectedYear] = useState('2025');
  
  // State for API data
  const [kpiData, setKpiData] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [usingFallback, setUsingFallback] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('checking');

  // Test backend connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      const isConnected = await testConnection();
      setConnectionStatus(isConnected ? 'connected' : 'disconnected');
    };
    
    checkConnection();
  }, []);

  // Fetch data on component mount and when year changes
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      setUsingFallback(false);
      
      try {
        // Try to fetch KPI data using centralized API
        console.log(`Fetching KPIs for year: ${selectedYear}`);
        const kpis = await api.getKPIs(selectedYear);
        setKpiData(kpis);
        
        // Try to fetch notifications
        console.log('Fetching notifications...');
        const notifs = await api.getNotifications();
        setNotifications(notifs);
        
        setConnectionStatus('connected');
      } catch (err) {
        setError(err.message);
        setUsingFallback(true);
        setConnectionStatus('disconnected');
        console.warn('Using fallback data due to API error:', err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [selectedYear]);

  // Convert API data to metrics format or use fallback
  const metrics = kpiData ? [
    {
      title: 'Total Revenue',
      value: formatCurrency(kpiData.revenue),
      last: formatCurrency(kpiData.revenue * 0.95),
      trend: kpiData.revenue > (kpiData.revenue * 0.95) ? 'up' : 'down',
      category: 'profitability'
    },
    {
      title: 'Gross Margin',
      value: `${((kpiData.revenue - (kpiData.revenue * 0.7)) / kpiData.revenue * 100).toFixed(1)}%`,
      last: '40%',
      trend: 'up',
      category: 'profitability'
    },
    {
      title: 'Inventory Turnover',
      value: kpiData.inventoryTurnover ? kpiData.inventoryTurnover.toFixed(1) : '0.0',
      last: kpiData.inventoryTurnover ? (kpiData.inventoryTurnover * 0.9).toFixed(1) : '0.0',
      trend: 'up',
      category: 'inventory'
    },
    {
      title: 'Total Stock Value',
      value: formatCurrency(kpiData.stockValue),
      last: formatCurrency(kpiData.stockValue * 1.05),
      trend: 'down',
      category: 'inventory'
    },
    {
      title: 'Low Stock Items',
      value: formatNumber(kpiData.lowStockItems),
      last: formatNumber(kpiData.lowStockItems + 5),
      trend: kpiData.lowStockItems < (kpiData.lowStockItems + 5) ? 'up' : 'down',
      category: 'inventory'
    },
    {
      title: 'Out of Stock',
      value: formatNumber(kpiData.outOfStockItems),
      last: formatNumber(kpiData.outOfStockItems + 3),
      trend: kpiData.outOfStockItems < (kpiData.outOfStockItems + 3) ? 'up' : 'down',
      category: 'inventory'
    },
    {
      title: 'Avg Monthly Sales',
      value: formatCurrency(kpiData.revenue / 12),
      last: formatCurrency((kpiData.revenue / 12) * 0.92),
      trend: 'up',
      category: 'sales'
    },
    {
      title: 'Active Products',
      value: formatNumber(kpiData.activeProducts),
      last: formatNumber(kpiData.activeProducts * 1.02),
      trend: 'down',
      category: 'inventory'
    },
    {
      title: 'Avg Basket Value',
      value: formatCurrency(kpiData.avgBasketValue),
      last: formatCurrency(kpiData.avgBasketValue * 0.98),
      trend: 'up',
      category: 'sales'
    },
    {
      title: 'Total Transactions',
      value: formatNumber(kpiData.transactionCount),
      last: formatNumber(kpiData.transactionCount * 0.94),
      trend: 'up',
      category: 'sales'
    }
  ] : fallbackMetrics;

  // Filter metrics based on selected category
  const filteredMetrics = metrics.filter(m => selectedCategory === 'all' || m.category === selectedCategory);

  // Convert notifications to alerts format or use fallback
  const alerts = notifications.length > 0 ? notifications.map(notif => ({
    id: notif.id,
    title: notif.title,
    text: notif.text,
    date: new Date(notif.date).toLocaleDateString(),
    type: notif.type
  })) : fallbackAlerts;

  // Loading state
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading KPI data...</p>
          <p className="mt-2 text-sm text-gray-500">
            Connection Status: {connectionStatus}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      
      {/* Connection Status Indicator */}
      <div className="px-6 pt-4">
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' : 
            connectionStatus === 'disconnected' ? 'bg-red-500' : 
            'bg-yellow-500 animate-pulse'
          }`}></div>
          <span className="text-sm text-gray-600">
            Backend: {connectionStatus === 'connected' ? 'Connected' : 
                     connectionStatus === 'disconnected' ? 'Disconnected' : 
                     'Checking...'}
          </span>
          {connectionStatus === 'connected' && (
            <span className="text-xs text-green-600">• Live Data</span>
          )}
        </div>
      </div>
      
      {/* Show warning if using fallback data */}
      {usingFallback && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 m-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">
                Using Demo Data
              </h3>
              <div className="mt-2 text-sm text-yellow-700">
                <p>Cannot connect to backend API. Displaying demo data. Please ensure your backend server is running on port 5000.</p>
                <p className="mt-1">Error: {error}</p>
              </div>
              <div className="mt-4">
                <button
                  onClick={() => window.location.reload()}
                  className="bg-yellow-100 px-3 py-2 rounded-md text-sm font-medium text-yellow-800 hover:bg-yellow-200"
                >
                  Retry Connection
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

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
          
          {/* Year Selector */}
          <div className="px-6 pb-4">
            <div className="flex items-center space-x-4">
              <label className="text-sm font-medium text-gray-700">Data Year:</label>
              <select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
                className="bg-white border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="2025">2025</option>
                <option value="2024">2024</option>
                <option value="2023">2023</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 p-6">
            {filteredMetrics.map(m => (
              <KPICard key={m.title} {...m} />
            ))}
          </div>
        </>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3 p-6">
          {alerts.map(a => (
            <NotificationCard key={a.id} {...a} />
          ))}
        </div>
      )}
    </div>
  );
}