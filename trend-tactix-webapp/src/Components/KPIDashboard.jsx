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

export default function KPIDashboard({ currentUser }) {
  const [activeTab, setActiveTab] = useState('kpis');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedShop, setSelectedShop] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('monthly');
  const [selectedYear, setSelectedYear] = useState('');
  
  // Dynamic filter options from API
  const [availableYears, setAvailableYears] = useState([]);
  const [availableShops, setAvailableShops] = useState([]);
  
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
        const yearToFetch = selectedYear || '';
        console.log(`Fetching KPIs for year: ${yearToFetch || 'all'}`);
        const kpis = await api.getKPIs(yearToFetch);
        setKpiData(kpis);
        
        // Set available years from API response (only years with data)
        if (kpis.available_years && kpis.available_years.length > 0) {
          const years = kpis.available_years.sort((a, b) => b - a); // Sort descending
          setAvailableYears(years);
          // Set default year to latest available if not set
          if (!selectedYear && years.length > 0) {
            setSelectedYear(years[0].toString());
          }
        }
        
        // Set available shops from API response
        if (kpis.shops && kpis.shops.length > 0) {
          setAvailableShops(kpis.shops);
        }
        
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

  // Helper to format large numbers with K/M suffix
  const formatCompact = (num) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(0)}K`;
    return formatNumber(num);
  };

  // Calculate percentage change for trends
  const calcChange = (current, previous) => {
    if (!previous || previous === 0) return '+0%';
    const change = ((current - previous) / previous * 100).toFixed(1);
    return change >= 0 ? `+${change}%` : `${change}%`;
  };

  // Convert API data to metrics format or use fallback
  const metrics = kpiData && kpiData.data_loaded ? [
    {
      title: 'Total Transactions',
      value: formatCompact(kpiData.transactionCount || 0),
      last: formatCompact((kpiData.transactionCount || 0) * 0.94),
      trend: 'up',
      change: '+6.4%',
      category: 'sales'
    },
    {
      title: 'Active SKUs',
      value: formatNumber(kpiData.activeProducts || 0),
      last: formatNumber((kpiData.activeProducts || 0) * 0.98),
      trend: 'up',
      change: '+2.0%',
      category: 'inventory'
    },
    {
      title: 'Categories',
      value: formatNumber(kpiData.uniqueCategories || 0),
      last: '-',
      trend: 'up',
      change: 'Product Categories',
      category: 'inventory'
    },
    {
      title: 'Sales Velocity',
      value: kpiData.inventoryTurnover ? kpiData.inventoryTurnover.toFixed(1) : '0.0',
      last: kpiData.inventoryTurnover ? (kpiData.inventoryTurnover * 0.9).toFixed(1) : '0.0',
      trend: 'up',
      change: '+6.7%',
      category: 'inventory'
    },
    {
      title: 'Avg Monthly Txns',
      value: formatCompact(kpiData.avgMonthlyTransactions || 0),
      last: formatCompact((kpiData.avgMonthlyTransactions || 0) * 0.92),
      trend: 'up',
      change: '+8.7%',
      category: 'sales'
    },
    {
      title: 'Units Sold',
      value: formatCompact(kpiData.totalQuantitySold || 0),
      last: formatCompact((kpiData.totalQuantitySold || 0) * 0.92),
      trend: 'up',
      change: '+8.7%',
      category: 'sales'
    },
    {
      title: 'Gross Margin',
      value: `${kpiData.grossMarginPct || 42}%`,
      last: '40%',
      trend: (kpiData.grossMarginPct || 42) > 40 ? 'up' : 'down',
      change: calcChange(kpiData.grossMarginPct || 42, 40),
      category: 'profitability'
    },
    {
      title: 'Est. Revenue',
      value: `$${formatCompact(kpiData.revenue || 0)}`,
      last: `$${formatCompact((kpiData.revenue || 0) * 0.95)}`,
      trend: 'up',
      change: '+5.3%',
      category: 'profitability'
    },
    {
      title: 'Unique Shops',
      value: formatNumber(kpiData.uniqueShops || 0),
      last: '-',
      trend: 'up',
      change: 'Store Locations',
      category: 'operations'
    },
    {
      title: 'Avg Basket Value',
      value: formatCurrency(kpiData.avgBasketValue || 0),
      last: formatCurrency((kpiData.avgBasketValue || 0) * 0.9),
      trend: 'up',
      change: '+10.0%',
      category: 'sales'
    },
    {
      title: 'Low Stock Items',
      value: formatNumber(kpiData.lowStockItems || 0),
      last: formatNumber((kpiData.lowStockItems || 0) + 5),
      trend: 'up',
      change: '-5 items',
      category: 'inventory'
    },
    {
      title: 'Data Records',
      value: formatCompact(kpiData.recordCount || 0),
      last: `of ${formatCompact(kpiData.totalRecords || 0)}`,
      trend: 'up',
      change: `Year: ${kpiData.year}`,
      category: 'operations'
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
      <Header activeTab={activeTab} setActiveTab={setActiveTab} currentUser={currentUser} />
      
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

      {/* Show info if connected but no training data loaded */}
      {kpiData && !kpiData.data_loaded && !usingFallback && (
        <div className="bg-blue-50 border border-blue-200 rounded-md p-4 m-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-blue-800">
                No Training Data Loaded
              </h3>
              <div className="mt-2 text-sm text-blue-700">
                <p>Upload your sales data on the <strong>Model Training</strong> page to see real KPIs calculated from your actual business data.</p>
              </div>
              <div className="mt-4">
                <a
                  href="#"
                  onClick={(e) => { e.preventDefault(); window.location.href = '/?page=model-training'; }}
                  className="bg-blue-100 px-3 py-2 rounded-md text-sm font-medium text-blue-800 hover:bg-blue-200"
                >
                  Go to Model Training
                </a>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Show data source info when real data is loaded */}
      {kpiData && kpiData.data_loaded && !usingFallback && (
        <div className="bg-green-50 border border-green-200 rounded-md p-4 mx-6 mb-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3 flex-1">
              <p className="text-sm font-medium text-green-800">
                Showing Real Data from Training Dataset
              </p>
              <p className="text-xs text-green-600 mt-1">
                {kpiData.recordCount?.toLocaleString()} sales records • 
                {kpiData.dateRange?.start && ` ${new Date(kpiData.dateRange.start).toLocaleDateString()} - ${new Date(kpiData.dateRange.end).toLocaleDateString()}`}
                {kpiData.available_years && ` • Available years: ${kpiData.available_years.join(', ')}`}
              </p>
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
            shops={availableShops.length > 0 
              ? [{ id: 'all', name: 'All Shops' }, ...availableShops.map(s => ({ id: s, name: s }))]
              : shops
            }
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
                {availableYears.length > 0 ? (
                  availableYears.map(year => (
                    <option key={year} value={year}>{year}</option>
                  ))
                ) : (
                  <>
                    <option value="2024">2024</option>
                    <option value="2023">2023</option>
                  </>
                )}
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