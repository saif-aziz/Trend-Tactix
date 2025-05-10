import React, { useState, useEffect } from 'react';
import { Filter, ChevronDown, RefreshCw, Download, Calendar, Search, AlertTriangle } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, AreaChart, Area } from 'recharts';

// Sample data with more meaningful structure
const shops = [
  { id: 'all', name: 'All Shops' },
  { id: 'shop_a', name: 'Shop A' },
  { id: 'shop_b', name: 'Shop B' },
  { id: 'shop_c', name: 'Shop C' },
  { id: 'shop_d', name: 'Shop D' }
];

const timeframes = [
  { id: 'daily', name: 'Daily' },
  { id: 'weekly', name: 'Weekly' },
  { id: 'monthly', name: 'Monthly' },
  { id: 'yearly', name: 'Yearly' }
];

// Sample data - In a real app, this would come from an API
const bestShop = {
  name: 'Shop A',
  delta: '1.5%',
  trend: 'up',
  series: [12000, 14000, 16000, 18000, 20000, 22000, 25000, 27000]
};

const worstShop = {
  name: 'Shop B',
  delta: '0.7%',
  trend: 'down',
  series: [15000, 14500, 14000, 13500, 13000, 12500, 12000, 11500]
};

const totalSales = {
  value: 2753,
  delta: '114',
  trend: 'up',
  segments: [
    { name: 'New Customer', value: 60 },
    { name: 'Loyal Customer', value: 40 }
  ]
};

const storeTypes = [
  { name: 'Retail Shops', value: 67 },
  { name: 'E-commerce', value: 23 },
  { name: 'Factory Outlet', value: 10 }
];

const colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

// Monthly comparison data
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const makeSeries = (base, variance = 5000, trend = 0) => months.map((m, i) => ({ 
  month: m, 
  last: base + Math.random() * variance - (trend * i * 100), 
  running: base + Math.random() * variance + (trend * i * 100),
  target: base + variance / 2 + (trend * i * 200)
}));

const dataStock = makeSeries(15000, 3000, 1);
const dataLoss = makeSeries(12000, 2000, -0.5);
const dataProfit = makeSeries(10000, 4000, 1.2);
const dataMarkdown = makeSeries(8000, 2500, 0.3);

// Top product data
const topProducts = [
  { name: 'Product A', sales: 1245, growth: 12.5 },
  { name: 'Product B', sales: 1120, growth: 8.3 },
  { name: 'Product C', sales: 980, growth: -2.1 },
  { name: 'Product D', sales: 840, growth: 5.7 },
  { name: 'Product E', sales: 650, growth: 1.2 }
];

export default function SalesDashboard() {
  const [selectedShop, setSelectedShop] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('monthly');
  const [isLoading, setIsLoading] = useState(true);
  const [showFilterPanel, setShowFilterPanel] = useState(false);
  const [dateRange, setDateRange] = useState({ start: '2025-01-01', end: '2025-05-07' });

  // Simulate data loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  // Simulate data refresh when filters change
  useEffect(() => {
    if (!isLoading) {
      setIsLoading(true);
      const timer = setTimeout(() => {
        setIsLoading(false);
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [selectedShop, selectedTimeframe]);

  const handleDataRefresh = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 800);
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm z-10">
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center">
              <h2 className="text-2xl font-bold text-gray-800">Sales Dashboard</h2>
              <div className="ml-4 text-sm text-gray-500 flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                <span>{dateRange.start} to {dateRange.end}</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Search */}
              <div className="relative rounded-md shadow-sm hidden md:block">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Search className="h-4 w-4 text-gray-400" />
                </div>
                <input
                  type="text"
                  className="focus:ring-indigo-500 focus:border-indigo-500 block w-full pl-10 sm:text-sm border-gray-300 rounded-md"
                  placeholder="Search dashboard"
                />
              </div>

              {/* Filter Button */}
              <button 
                onClick={() => setShowFilterPanel(!showFilterPanel)}
                className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
              >
                <Filter className="h-4 w-4 mr-2" />
                Filters
                <ChevronDown className="h-4 w-4 ml-1" />
              </button>
              
              {/* Refresh Button */}
              <button 
                onClick={handleDataRefresh}
                className={`inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              
              {/* Export Button */}
              <button className="inline-flex items-center px-3 py-2 border border-gray-300 text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50">
                <Download className="h-4 w-4 mr-2" />
                Export
              </button>
            </div>
          </div>
          
          {/* Filter Panel */}
          {showFilterPanel && (
            <div className="p-4 bg-gray-50 border-b border-gray-200 grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Shop Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Shop</label>
                <select
                  value={selectedShop}
                  onChange={(e) => setSelectedShop(e.target.value)}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                >
                  {shops.map((shop) => (
                    <option key={shop.id} value={shop.id}>{shop.name}</option>
                  ))}
                </select>
              </div>
              
              {/* Timeframe Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Timeframe</label>
                <select
                  value={selectedTimeframe}
                  onChange={(e) => setSelectedTimeframe(e.target.value)}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                >
                  {timeframes.map((timeframe) => (
                    <option key={timeframe.id} value={timeframe.id}>{timeframe.name}</option>
                  ))}
                </select>
              </div>
              
              {/* Date Range */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Date Range</label>
                <div className="flex space-x-2">
                  <input
                    type="date"
                    value={dateRange.start}
                    onChange={(e) => setDateRange({...dateRange, start: e.target.value})}
                    className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                  />
                  <input
                    type="date"
                    value={dateRange.end}
                    onChange={(e) => setDateRange({...dateRange, end: e.target.value})}
                    className="mt-1 block w-full pl-3 pr-3 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                  />
                </div>
              </div>
            </div>
          )}
        </header>
  
        {/* Dashboard Content */}
        <main className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <RefreshCw className="w-12 h-12 mx-auto text-indigo-500 animate-spin" />
                <p className="mt-2 text-gray-500">Loading dashboard data...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <MetricCard 
                  title="Best Selling Shop" 
                  data={bestShop} 
                  accent="from-green-400 to-teal-500" 
                />
                <MetricCard 
                  title="Worst Selling Shop" 
                  data={worstShop} 
                  accent="from-red-400 to-rose-500" 
                />
                <TotalSalesCard 
                  title="Total Sales" 
                  data={totalSales} 
                  accent="from-indigo-400 to-blue-500" 
                />
                <PieCard 
                  title="Store Type Distribution" 
                  data={storeTypes} 
                  colors={colors} 
                  accent="from-purple-400 to-violet-500" 
                />
              </div>
              
            
             

              {/* Main Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <EnhancedTrendChart 
                  title="Sales Stock Ratio" 
                  data={dataStock} 
                  accent="#4F46E5" 
                  type="area"
                />
                <EnhancedTrendChart 
                  title="Loss Indicator" 
                  data={dataLoss} 
                  accent="#EF4444" 
                  type="line"
                />
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <EnhancedTrendChart 
                  title="Profit Margin" 
                  data={dataProfit} 
                  accent="#10B981" 
                  type="bar"
                />
                <EnhancedTrendChart 
                  title="Mark Down" 
                  data={dataMarkdown} 
                  accent="#F59E0B" 
                  type="area"
                />
              </div>
              
              {/* Top Products Table */}
              <div className="bg-white shadow rounded-lg overflow-hidden mb-6">
                <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Top Selling Products</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Product</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sales</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Growth</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {topProducts.map((product, idx) => (
                        <tr key={idx}>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{product.name}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${product.sales.toLocaleString()}</td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <span className={product.growth >= 0 ? 'text-green-600' : 'text-red-600'}>
                              {product.growth >= 0 ? '+' : ''}{product.growth}%
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            <div className="w-20 h-8">
                              <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={[0,1,2,3,4,5].map(i => ({
                                  x: i, 
                                  y: 50 + Math.random() * 50 * (product.growth >= 0 ? 1 : -1)
                                }))}>
                                  <Line 
                                    type="monotone" 
                                    dataKey="y" 
                                    stroke={product.growth >= 0 ? '#10B981' : '#EF4444'} 
                                    strokeWidth={2} 
                                    dot={false} 
                                  />
                                </LineChart>
                              </ResponsiveContainer>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function MetricCard({ title, data, accent }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden transform transition hover:shadow-lg">
      <div className={`h-1 bg-gradient-to-r ${accent}`}></div>
      <div className="p-6 space-y-2">
        <div className="flex justify-between items-center">
          <h3 className="text-gray-600 font-medium text-sm">{title}</h3>
          <span className={`text-sm font-semibold ${data.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>  
            {data.trend === 'up' ? '↑' : '↓'} {data.delta}
          </span>
        </div>
        <div className="flex items-center">
          <h2 className="text-2xl font-bold text-gray-800">{data.name}</h2>
        </div>
        <ResponsiveContainer width="100%" height={60}>
          <LineChart data={data.series.map((v,i)=>({ x: i, y: v }))} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <Line 
              type="monotone" 
              dataKey="y" 
              stroke={accent.split(' ')[1]} 
              strokeWidth={2} 
              dot={false} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TotalSalesCard({ title, data, accent }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden transform transition hover:shadow-lg">
      <div className={`h-1 bg-gradient-to-r ${accent}`}></div>
      <div className="p-6 space-y-2 flex items-center justify-between">
        <div>
          <h3 className="text-gray-600 font-medium text-sm">{title}</h3>
          <p className="mt-1 text-3xl font-bold text-gray-900">{data.value}</p>
          <span className={`text-sm ${data.trend === 'up' ? 'text-green-600' : 'text-red-600'}`}>
            {data.trend === 'up' ? '↑' : '↓'} {data.delta} vs last month
          </span>
        </div>
        <div className="w-24 h-24">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie 
                data={data.segments} 
                dataKey="value" 
                nameKey="name" 
                innerRadius={30} 
                outerRadius={40} 
                paddingAngle={2}
              >
                {data.segments.map((entry, idx) => (
                  <Cell key={idx} fill={colors[idx]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value}%`} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function PieCard({ title, data, colors, accent }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden transform transition hover:shadow-lg">
      <div className={`h-1 bg-gradient-to-r ${accent}`}></div>
      <div className="p-6 space-y-2">
        <h3 className="text-gray-600 font-medium text-sm">{title}</h3>
        <div className="flex items-center">
          <div className="w-28 h-28">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie 
                  data={data} 
                  dataKey="value" 
                  nameKey="name" 
                  innerRadius={35} 
                  outerRadius={50} 
                  paddingAngle={2}
                >
                  {data.map((entry, idx) => (
                    <Cell key={idx} fill={colors[idx % colors.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <ul className="ml-4 space-y-1">
            {data.map((seg, i) => (
              <li key={i} className="flex items-center text-sm">
                <span 
                  className="w-3 h-3 mr-2 rounded-full" 
                  style={{ background: colors[i % colors.length] }} 
                />
                {seg.name}: {seg.value}%
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

function EnhancedTrendChart({ title, data, accent, type }) {
  const renderChartType = () => {
    switch(type) {
      case 'area':
        return (
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <defs>
              <linearGradient id={`colorGradient-${title}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={accent} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={accent} stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="last" stroke="#9CA3AF" strokeDasharray="5 5" fill="#F3F4F6" />
            <Area type="monotone" dataKey="running" stroke={accent} fill={`url(#colorGradient-${title})`} />
            <Area type="monotone" dataKey="target" stroke="#6B7280" strokeDasharray="3 3" fill="none" />
          </AreaChart>
        );
      case 'bar':
        return (
          <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="last" fill="#D1D5DB" />
            <Bar dataKey="running" fill={accent} />
            <Line type="monotone" dataKey="target" stroke="#6B7280" strokeDasharray="3 3" dot={false} />
          </BarChart>
        );
      default:
        return (
          <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="last" stroke="#9CA3AF" strokeDasharray="5 5" dot={false} />
            <Line type="monotone" dataKey="running" stroke={accent} dot={{ r: 3 }} activeDot={{ r: 5 }} />
            <Line type="monotone" dataKey="target" stroke="#6B7280" strokeDasharray="3 3" dot={false} />
          </LineChart>
        );
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden transform transition hover:shadow-lg">
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-gray-700 font-medium">{title}</h3>
          <div className="flex space-x-2">
            <button className="text-xs font-semibold bg-indigo-100 text-indigo-800 px-2 py-1 rounded hover:bg-indigo-200">Details</button>
            <button className="text-xs font-semibold bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200">Predict</button>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={220}>
          {renderChartType()}
        </ResponsiveContainer>
      </div>
    </div>
  );
}