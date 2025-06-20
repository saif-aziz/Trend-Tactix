import React, { useState, useEffect } from 'react';
import { Filter, ChevronDown, RefreshCw, Download, Calendar, Search, AlertTriangle, Package, TrendingUp, TrendingDown, Zap } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, AreaChart, Area, ComposedChart } from 'recharts';

// Sample data
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

const categories = [
  { id: 'all', name: 'All Categories' },
  { id: 'men', name: 'Men\'s Wear' },
  { id: 'women', name: 'Women\'s Wear' },
  { id: 'kids', name: 'Kids Wear' },
  { id: 'accessories', name: 'Accessories' }
];

// Sample inventory metrics
const inventoryMetrics = {
  totalValue: {
    value: '$2.8M',
    change: '5.2%',
    trend: 'up',
    previous: '$2.66M'
  },
  turnoverRate: {
    value: '4.2',
    change: '0.8',
    trend: 'up',
    previous: '3.4'
  },
  stockouts: {
    value: '12',
    change: '3',
    trend: 'down',
    previous: '15'
  },
  excessStock: {
    value: '8.5%',
    change: '1.2%',
    trend: 'down',
    previous: '9.7%'
  }
};

// Category distribution
const categoryDistribution = [
  { name: 'Men\'s Wear', value: 35, stock: 1250 },
  { name: 'Women\'s Wear', value: 40, stock: 1500 },
  { name: 'Kids Wear', value: 15, stock: 650 },
  { name: 'Accessories', value: 10, stock: 400 }
];

// ABC Analysis data
const abcAnalysis = [
  { category: 'A Items', percentage: 70, count: 45, color: '#10B981' },
  { category: 'B Items', percentage: 20, count: 35, color: '#F59E0B' },
  { category: 'C Items', percentage: 10, count: 120, color: '#EF4444' }
];

// Monthly data
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const inventoryTrends = months.map((month, i) => ({
  month,
  stockLevel: 1500 + Math.sin(i * 0.5) * 200 + Math.random() * 100,
  inbound: 300 + Math.random() * 150,
  outbound: 280 + Math.random() * 120,
  optimal: 1600,
  turnover: 3.5 + Math.sin(i * 0.3) * 0.5 + Math.random() * 0.3
}));

const agingAnalysis = [
  { range: '0-30 days', value: 45, amount: '$890K' },
  { range: '31-60 days', value: 25, amount: '$520K' },
  { range: '61-90 days', value: 18, amount: '$380K' },
  { range: '90+ days', value: 12, amount: '$210K' }
];

// Low stock items
const lowStockItems = [
  { name: 'Blue Denim Jacket', sku: 'BDJ-001', current: 8, reorder: 25, category: 'Men\'s Wear', status: 'critical' },
  { name: 'Floral Summer Dress', sku: 'FSD-045', current: 12, reorder: 30, category: 'Women\'s Wear', status: 'warning' },
  { name: 'Kids Superhero T-Shirt', sku: 'KST-023', current: 15, reorder: 40, category: 'Kids Wear', status: 'warning' },
  { name: 'Leather Handbag', sku: 'LHB-089', current: 5, reorder: 20, category: 'Accessories', status: 'critical' },
  { name: 'Casual Sneakers', sku: 'CS-156', current: 18, reorder: 35, category: 'Men\'s Wear', status: 'warning' }
];

// Stock movement data
const stockMovement = [
  { product: 'Blue Denim Jacket', inbound: 50, outbound: 72, net: -22 },
  { product: 'Floral Summer Dress', inbound: 80, outbound: 65, net: 15 },
  { product: 'Kids Superhero T-Shirt', inbound: 45, outbound: 58, net: -13 },
  { product: 'Leather Handbag', inbound: 30, outbound: 42, net: -12 },
  { product: 'Casual Sneakers', inbound: 60, outbound: 55, net: 5 }
];

const colors = ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

export default function InventoryDashboard() {
  const [selectedShop, setSelectedShop] = useState('all');
  const [selectedTimeframe, setSelectedTimeframe] = useState('monthly');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(true);
  const [showFilterPanel, setShowFilterPanel] = useState(false);
  const [dateRange, setDateRange] = useState({ start: '2025-01-01', end: '2025-05-15' });

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
  }, [selectedShop, selectedTimeframe, selectedCategory]);

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
              <Package className="w-6 h-6 mr-2 text-indigo-600" />
              <h2 className="text-2xl font-bold text-gray-800">Inventory Analysis</h2>
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
                  placeholder="Search inventory..."
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
            <div className="p-4 bg-gray-50 border-b border-gray-200 grid grid-cols-1 md:grid-cols-4 gap-4">
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
              
              {/* Category Filter */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                >
                  {categories.map((category) => (
                    <option key={category.id} value={category.id}>{category.name}</option>
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
                <p className="mt-2 text-gray-500">Loading inventory data...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                <InventoryMetricCard 
                  title="Total Inventory Value" 
                  data={inventoryMetrics.totalValue} 
                  icon={Package}
                  accent="from-blue-400 to-blue-500" 
                />
                <InventoryMetricCard 
                  title="Inventory Turnover" 
                  data={inventoryMetrics.turnoverRate} 
                  icon={TrendingUp}
                  accent="from-green-400 to-green-500" 
                />
                <InventoryMetricCard 
                  title="Stockouts" 
                  data={inventoryMetrics.stockouts} 
                  icon={AlertTriangle}
                  accent="from-red-400 to-red-500" 
                />
                <InventoryMetricCard 
                  title="Excess Stock" 
                  data={inventoryMetrics.excessStock} 
                  icon={TrendingDown}
                  accent="from-amber-400 to-amber-500" 
                />
              </div>

              {/* Category Distribution and ABC Analysis */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <CategoryDistributionCard 
                  title="Inventory by Category" 
                  data={categoryDistribution}
                  colors={colors}
                />
                <ABCAnalysisCard 
                  title="ABC Analysis" 
                  data={abcAnalysis}
                />
              </div>

              {/* Inventory Trends Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <InventoryTrendChart 
                  title="Stock Level Trends" 
                  data={inventoryTrends} 
                  type="area"
                />
                <InventoryTurnoverChart 
                  title="Inventory Turnover Trends" 
                  data={inventoryTrends} 
                />
              </div>

              {/* Stock Aging and Movement */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <StockAgingChart 
                  title="Stock Aging Analysis" 
                  data={agingAnalysis}
                  colors={colors}
                />
                <StockMovementChart 
                  title="Stock Movement Analysis" 
                  data={stockMovement}
                />
              </div>
              
              {/* Low Stock Alert Table */}
              <div className="bg-white shadow rounded-lg overflow-hidden">
                <div className="px-4 py-5 border-b border-gray-200 sm:px-6">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg leading-6 font-medium text-gray-900 flex items-center">
                      <AlertTriangle className="w-5 h-5 mr-2 text-amber-500" />
                      Low Stock Alerts
                    </h3>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      {lowStockItems.filter(item => item.status === 'critical').length} Critical
                    </span>
                  </div>
                </div>
                <LowStockTable data={lowStockItems} />
              </div>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function InventoryMetricCard({ title, data, icon: Icon, accent }) {
  const isPositive = data.trend === 'up';
  const isBetter = (title === 'Stockouts' || title === 'Excess Stock') ? !isPositive : isPositive;

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden transform transition hover:shadow-lg">
      <div className={`h-1 bg-gradient-to-r ${accent}`}></div>
      <div className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Icon className="w-6 h-6 text-gray-400 mr-2" />
            <h3 className="text-gray-600 font-medium text-sm">{title}</h3>
          </div>
          <span className={`text-sm font-semibold flex items-center ${isBetter ? 'text-green-600' : 'text-red-600'}`}>  
            {isPositive ? (
              <TrendingUp className="w-4 h-4 mr-1" />
            ) : (
              <TrendingDown className="w-4 h-4 mr-1" />
            )}
            {data.change}
          </span>
        </div>
        <div className="mt-3">
          <p className="text-3xl font-bold text-gray-900">{data.value}</p>
          <p className="text-sm text-gray-500 mt-1">Previous: {data.previous}</p>
        </div>
      </div>
    </div>
  );
}

function CategoryDistributionCard({ title, data, colors }) {
  const totalStock = data.reduce((sum, item) => sum + item.stock, 0);

  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <h3 className="text-gray-700 font-medium mb-4">{title}</h3>
        <div className="flex items-center">
          <div className="w-40 h-40">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie 
                  data={data} 
                  dataKey="value" 
                  nameKey="name" 
                  cx="50%" 
                  cy="50%" 
                  innerRadius={50} 
                  outerRadius={70} 
                  paddingAngle={2}
                >
                  {data.map((entry, idx) => (
                    <Cell key={idx} fill={colors[idx % colors.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value, name) => [`${value}%`, name]} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="ml-6 flex-1">
            {data.map((item, idx) => (
              <div key={idx} className="flex items-center justify-between mb-3">
                <div className="flex items-center">
                  <span 
                    className="w-3 h-3 mr-3 rounded-full" 
                    style={{ backgroundColor: colors[idx % colors.length] }} 
                  />
                  <span className="text-sm font-medium text-gray-700">{item.name}</span>
                </div>
                <div className="text-right">
                  <span className="text-sm font-bold text-gray-900">{item.stock} units</span>
                  <span className="block text-xs text-gray-500">{item.value}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function ABCAnalysisCard({ title, data }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <h3 className="text-gray-700 font-medium mb-4">{title}</h3>
        <div className="space-y-4">
          {data.map((item, idx) => (
            <div key={idx} className="flex items-center">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-gray-700">{item.category}</span>
                  <span className="text-sm font-bold text-gray-900">{item.percentage}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="h-2 rounded-full" 
                    style={{ 
                      width: `${item.percentage}%`, 
                      backgroundColor: item.color 
                    }}
                  ></div>
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-500">{item.count} products</span>
                  <span className="text-xs text-gray-500">Value contribution</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function InventoryTrendChart({ title, data, type }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-gray-700 font-medium">{title}</h3>
          <div className="flex space-x-2">
            <button className="text-xs font-semibold bg-blue-100 text-blue-800 px-2 py-1 rounded hover:bg-blue-200">Details</button>
            <button className="text-xs font-semibold bg-green-100 text-green-800 px-2 py-1 rounded hover:bg-green-200">Optimize</button>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="stockLevel" stackId="1" stroke="#4F46E5" fill="#4F46E5" fillOpacity={0.3} />
            <Bar dataKey="inbound" fill="#10B981" />
            <Bar dataKey="outbound" fill="#EF4444" />
            <Line type="monotone" dataKey="optimal" stroke="#F59E0B" strokeDasharray="5 5" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function InventoryTurnoverChart({ title, data }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-gray-700 font-medium">{title}</h3>
          <div className="flex space-x-2">
            <button className="text-xs font-semibold bg-purple-100 text-purple-800 px-2 py-1 rounded hover:bg-purple-200">Analyze</button>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip formatter={(value) => [value.toFixed(2), 'Turnover Rate']} />
            <Legend />
            <defs>
              <linearGradient id="colorTurnover" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <Area type="monotone" dataKey="turnover" stroke="#8B5CF6" fill="url(#colorTurnover)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StockAgingChart({ title, data, colors }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <h3 className="text-gray-700 font-medium mb-4">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical" margin={{ top: 10, right: 30, left: 100, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" />
            <YAxis dataKey="range" type="category" width={90} />
            <Tooltip formatter={(value, name) => [`${value}%`, 'Stock Percentage']} />
            <Bar dataKey="value" fill="#4F46E5" />
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 space-y-2">
          {data.map((item, idx) => (
            <div key={idx} className="flex justify-between items-center text-sm">
              <span className="text-gray-600">{item.range}</span>
              <span className="font-semibold text-gray-900">{item.amount}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StockMovementChart({ title, data }) {
  return (
    <div className="bg-white rounded-xl shadow-md overflow-hidden">
      <div className="p-6">
        <h3 className="text-gray-700 font-medium mb-4">{title}</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 100 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="product" angle={-45} textAnchor="end" height={100} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="inbound" stackId="a" fill="#10B981" />
            <Bar dataKey="outbound" stackId="b" fill="#EF4444" />
            <Line type="monotone" dataKey="net" stroke="#4F46E5" strokeWidth={3} dot={{ r: 4 }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function LowStockTable({ data }) {
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Product</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">SKU</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Current Stock</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reorder Level</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Category</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((item) => (
              <tr key={item.sku} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm font-medium text-gray-900">{item.name}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{item.sku}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900 font-medium">{item.current}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-900">{item.reorder}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="text-sm text-gray-500">{item.category}</div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                    item.status === 'critical' 
                      ? 'bg-red-100 text-red-800' 
                      : 'bg-yellow-100 text-yellow-800'
                  }`}>
                    {item.status === 'critical' ? 'Critical' : 'Warning'}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                  <button className="text-indigo-600 hover:text-indigo-900 mr-3">Reorder</button>
                  <button className="text-gray-600 hover:text-gray-900">Details</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  