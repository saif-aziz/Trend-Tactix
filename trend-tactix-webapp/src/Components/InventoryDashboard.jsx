import React, { useState, useEffect } from 'react';
import { 
  Filter, ChevronDown, ChevronUp, RefreshCw, Download, Calendar, Search, 
  AlertTriangle, Package, TrendingUp, TrendingDown, Zap, BarChart3, 
  PieChart as PieChartIcon, Activity, Target, ShoppingBag, Users,
  ArrowUpRight, ArrowDownRight, Layers, Clock, Award, AlertCircle,
  Lightbulb, CheckCircle, Info, Store
} from 'lucide-react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend, BarChart, Bar, AreaChart, Area, 
  ComposedChart, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';

// API base URL
const API_BASE = 'http://localhost:5000/api';

// Color palette
const COLORS = {
  primary: '#6366F1',
  secondary: '#8B5CF6',
  success: '#10B981',
  warning: '#F59E0B',
  danger: '#EF4444',
  info: '#3B82F6',
  purple: '#A855F7',
  pink: '#EC4899',
  teal: '#14B8A6',
  orange: '#F97316'
};

const CHART_COLORS = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#3B82F6', '#A855F7'];

export default function InventoryDashboard() {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [showFilterPanel, setShowFilterPanel] = useState(false);
  const [selectedYear, setSelectedYear] = useState('all');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [expandedSections, setExpandedSections] = useState({
    trends: true,
    categories: true,
    products: true,
    insights: true
  });

  // Fetch analytics data
  useEffect(() => {
    fetchAnalytics();
  }, []);

  const fetchAnalytics = async () => {
      setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/inventory/analytics`);
      const data = await response.json();
      
      if (data.status === 'success') {
        setAnalyticsData(data);
      } else {
        setError(data.message || 'Failed to load analytics');
      }
    } catch (err) {
      setError('Failed to connect to server. Please ensure backend is running.');
      console.error('Analytics fetch error:', err);
    } finally {
        setIsLoading(false);
    }
  };

  const handleRefresh = () => {
    fetchAnalytics();
  };

  // Export analytics data as CSV
  const handleExport = () => {
    if (!analyticsData) {
      alert('No data available to export');
      return;
    }

    const data = filteredData || analyticsData;
    const timestamp = new Date().toISOString().split('T')[0];
    const filterSuffix = hasActiveFilters 
      ? `_${selectedYear !== 'all' ? selectedYear : 'AllYears'}_${selectedCategory !== 'all' ? selectedCategory.replace(/\s+/g, '-') : 'AllCategories'}`
      : '';

    // Create CSV content
    let csvContent = '';

    // Summary Section
    csvContent += 'SALES & INVENTORY ANALYTICS REPORT\n';
    csvContent += `Generated: ${new Date().toLocaleString()}\n`;
    csvContent += `Data Period: ${data.summary?.dateRange?.start || 'N/A'} to ${data.summary?.dateRange?.end || 'N/A'}\n`;
    if (hasActiveFilters) {
      csvContent += `Filters Applied: Year=${selectedYear}, Category=${selectedCategory}\n`;
    }
    csvContent += '\n';

    // KPI Summary
    csvContent += '--- KEY METRICS ---\n';
    csvContent += `Total Transactions,${data.summary?.totalTransactions || 0}\n`;
    csvContent += `Total Units Sold,${data.summary?.totalUnitsSold || 0}\n`;
    csvContent += `Unique SKUs,${data.summary?.uniqueSKUs || 0}\n`;
    csvContent += `Unique Products,${data.summary?.uniqueProducts || 0}\n`;
    csvContent += `Unique Categories,${data.summary?.uniqueCategories || 0}\n`;
    csvContent += `Avg Units/Day,${data.summary?.avgUnitsPerDay || 0}\n`;
    csvContent += '\n';

    // Category Performance
    if (data.categoryPerformance && data.categoryPerformance.length > 0) {
      csvContent += '--- CATEGORY PERFORMANCE ---\n';
      csvContent += 'Category,Total Units,Transactions,SKUs,Market Share %,Velocity\n';
      data.categoryPerformance.forEach(cat => {
        csvContent += `"${cat.category}",${cat.totalUnits},${cat.transactions},${cat.uniqueSKUs},${cat.marketShare},${cat.velocity}\n`;
      });
      csvContent += '\n';
    }

    // Monthly Trends
    if (data.monthlyTrends && data.monthlyTrends.length > 0) {
      csvContent += '--- MONTHLY TRENDS ---\n';
      csvContent += 'Year,Month,Units,Active SKUs\n';
      data.monthlyTrends.forEach(m => {
        csvContent += `${m.year},${m.monthName},${m.units},${m.activeSKUs || 0}\n`;
      });
      csvContent += '\n';
    }

    // Quarterly Trends
    if (data.quarterlyTrends && data.quarterlyTrends.length > 0) {
      csvContent += '--- QUARTERLY TRENDS ---\n';
      csvContent += 'Year,Quarter,Units\n';
      data.quarterlyTrends.forEach(q => {
        csvContent += `${q.year},${q.quarter},${q.units}\n`;
      });
      csvContent += '\n';
    }

    // Top Products
    if (data.topProducts && data.topProducts.length > 0) {
      csvContent += '--- TOP PERFORMING PRODUCTS ---\n';
      csvContent += 'Rank,Product Name,Category,Total Units,SKU Count\n';
      data.topProducts.forEach(p => {
        csvContent += `${p.rank},"${p.productName}","${p.category}",${p.totalUnits},${p.skuCount}\n`;
      });
      csvContent += '\n';
    }

    // Slow Moving Products
    if (data.slowMovingProducts && data.slowMovingProducts.length > 0) {
      csvContent += '--- SLOW MOVING PRODUCTS (Bottom 10%) ---\n';
      csvContent += 'Product Name,Category,Total Units\n';
      data.slowMovingProducts.forEach(p => {
        csvContent += `"${p.productName}","${p.category}",${p.totalUnits}\n`;
      });
      csvContent += '\n';
    }

    // Seasonal Analysis
    if (data.seasonalAnalysis && data.seasonalAnalysis.length > 0) {
      csvContent += '--- SEASONAL ANALYSIS ---\n';
      csvContent += 'Season,Total Units,Transactions,SKUs\n';
      data.seasonalAnalysis.forEach(s => {
        csvContent += `"${s.season}",${s.totalUnits},${s.transactions},${s.uniqueSKUs}\n`;
      });
      csvContent += '\n';
    }

    // Gender Breakdown
    if (data.genderBreakdown && data.genderBreakdown.length > 0) {
      csvContent += '--- GENDER BREAKDOWN ---\n';
      csvContent += 'Gender,Units\n';
      data.genderBreakdown.forEach(g => {
        csvContent += `"${g.gender}",${g.units}\n`;
      });
      csvContent += '\n';
    }

    // Create download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `sales_inventory_analytics_${timestamp}${filterSuffix}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  // Get available years from data
  const availableYears = analyticsData?.yearOverYear 
    ? Object.keys(analyticsData.yearOverYear).sort() 
    : [];

  // Get available categories
  const availableCategories = analyticsData?.categoryPerformance?.map(c => c.category) || [];

  // ============ FILTER DATA BASED ON SELECTIONS ============
  const getFilteredData = () => {
    if (!analyticsData) return null;

    let filteredMonthlyTrends = analyticsData.monthlyTrends || [];
    let filteredQuarterlyTrends = analyticsData.quarterlyTrends || [];
    let filteredCategoryPerformance = analyticsData.categoryPerformance || [];
    let filteredTopProducts = analyticsData.topProducts || [];
    let filteredSlowMoving = analyticsData.slowMovingProducts || [];
    let filteredYoY = analyticsData.yearOverYear || {};

    // Filter by Year
    if (selectedYear !== 'all') {
      const yearNum = parseInt(selectedYear);
      filteredMonthlyTrends = filteredMonthlyTrends.filter(item => item.year === yearNum);
      filteredQuarterlyTrends = filteredQuarterlyTrends.filter(item => item.year === yearNum);
      filteredYoY = { [selectedYear]: filteredYoY[selectedYear] };
    }

    // Filter by Category
    if (selectedCategory !== 'all') {
      filteredCategoryPerformance = filteredCategoryPerformance.filter(
        cat => cat.category === selectedCategory
      );
      filteredTopProducts = filteredTopProducts.filter(
        p => p.category === selectedCategory
      );
      filteredSlowMoving = filteredSlowMoving.filter(
        p => p.category === selectedCategory
      );
    }

    // Recalculate summary based on filters
    let filteredSummary = { ...analyticsData.summary };
    if (selectedCategory !== 'all' && filteredCategoryPerformance.length > 0) {
      const catData = filteredCategoryPerformance[0];
      filteredSummary = {
        ...filteredSummary,
        totalTransactions: catData.transactions,
        totalUnitsSold: catData.totalUnits,
        uniqueSKUs: catData.uniqueSKUs,
        uniqueCategories: 1,
        avgUnitsPerDay: Math.round(catData.totalUnits / Math.max(filteredSummary.dateRange?.days || 1, 1))
      };
    }
    
    if (selectedYear !== 'all' && filteredYoY[selectedYear]) {
      filteredSummary = {
        ...filteredSummary,
        totalUnitsSold: filteredYoY[selectedYear].units,
        uniqueSKUs: filteredYoY[selectedYear].skus
      };
    }

    return {
      ...analyticsData,
      summary: filteredSummary,
      monthlyTrends: filteredMonthlyTrends,
      quarterlyTrends: filteredQuarterlyTrends,
      categoryPerformance: filteredCategoryPerformance,
      topProducts: filteredTopProducts,
      slowMovingProducts: filteredSlowMoving,
      yearOverYear: filteredYoY
    };
  };

  // Get filtered data
  const filteredData = getFilteredData();

  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-indigo-50">
        <div className="text-center">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-indigo-200 rounded-full animate-pulse"></div>
            <RefreshCw className="w-8 h-8 text-indigo-600 absolute top-4 left-4 animate-spin" />
          </div>
          <p className="mt-4 text-gray-600 font-medium">Loading Inventory Analytics...</p>
          <p className="text-sm text-gray-400 mt-1">Analyzing historical data</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-gradient-to-br from-slate-50 to-red-50">
        <div className="text-center max-w-md p-8 bg-white rounded-2xl shadow-xl">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <AlertCircle className="w-8 h-8 text-red-600" />
          </div>
          <h3 className="text-xl font-bold text-gray-900 mb-2">Unable to Load Analytics</h3>
          <p className="text-gray-600 mb-6">{error}</p>
          <button
            onClick={handleRefresh}
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // Use filtered data for display (filtered by year/category)
  const { summary, yearOverYear, monthlyTrends, categoryPerformance, seasonalAnalysis, 
          genderBreakdown, topProducts, slowMovingProducts, insights, quarterlyTrends,
          dayOfWeekPatterns, shopPerformance } = filteredData || {};
  
  // Check if filters are active
  const hasActiveFilters = selectedYear !== 'all' || selectedCategory !== 'all';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50">
        {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-20">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl">
                <BarChart3 className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                  Sales & Inventory Analytics
                </h1>
                <p className="text-sm text-gray-500 flex items-center mt-0.5">
                  <Calendar className="w-3.5 h-3.5 mr-1" />
                  {summary?.dateRange?.start} to {summary?.dateRange?.end}
                  <span className="mx-2">•</span>
                  <span className="text-indigo-600 font-medium">{summary?.dateRange?.days || 0} days of data</span>
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button 
                onClick={() => setShowFilterPanel(!showFilterPanel)}
                className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all relative ${
                  showFilterPanel || hasActiveFilters
                    ? 'bg-indigo-100 text-indigo-700 border border-indigo-200' 
                    : 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Filter className="w-4 h-4 mr-2" />
                Filters
                {hasActiveFilters && (
                  <span className="ml-2 px-1.5 py-0.5 bg-indigo-600 text-white text-xs rounded-full">
                    {(selectedYear !== 'all' ? 1 : 0) + (selectedCategory !== 'all' ? 1 : 0)}
                  </span>
                )}
                <ChevronDown className={`w-4 h-4 ml-1 transition-transform ${showFilterPanel ? 'rotate-180' : ''}`} />
              </button>
              
              <button 
                onClick={handleRefresh}
                disabled={isLoading}
                className="flex items-center px-4 py-2 bg-white border border-gray-200 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              
              <button 
                onClick={handleExport}
                className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors"
              >
                <Download className="w-4 h-4 mr-2" />
                Export Report
              </button>
            </div>
          </div>
          
          {/* Filter Panel */}
          {showFilterPanel && (
            <div className="mt-4 p-4 bg-gray-50 rounded-xl">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Year</label>
                  <select
                    value={selectedYear}
                    onChange={(e) => setSelectedYear(e.target.value)}
                    className={`w-full px-3 py-2 bg-white border rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 ${
                      selectedYear !== 'all' ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200'
                    }`}
                  >
                    <option value="all">All Years</option>
                    {availableYears.map(year => (
                      <option key={year} value={year}>{year}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">Category</label>
                  <select
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className={`w-full px-3 py-2 bg-white border rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 ${
                      selectedCategory !== 'all' ? 'border-indigo-400 bg-indigo-50' : 'border-gray-200'
                    }`}
                  >
                    <option value="all">All Categories</option>
                    {availableCategories.map(cat => (
                      <option key={cat} value={cat}>{cat}</option>
                    ))}
                  </select>
                </div>
                <div className="flex items-end">
                  {hasActiveFilters && (
                    <button
                      onClick={() => {
                        setSelectedYear('all');
                        setSelectedCategory('all');
                      }}
                      className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-300 transition-colors"
                    >
                      Clear Filters
                    </button>
                  )}
                </div>
              </div>
              {hasActiveFilters && (
                <div className="mt-3 flex items-center text-sm text-indigo-600">
                  <Filter className="w-4 h-4 mr-2" />
                  <span>
                    Showing data for: 
                    {selectedYear !== 'all' && <span className="font-semibold ml-1">Year {selectedYear}</span>}
                    {selectedYear !== 'all' && selectedCategory !== 'all' && <span className="mx-1">•</span>}
                    {selectedCategory !== 'all' && <span className="font-semibold">{selectedCategory}</span>}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
        </header>
  
      {/* Main Content */}
      <main className="p-6 space-y-6">
        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <KPICard
            title="Total Transactions"
            value={summary?.totalTransactions?.toLocaleString() || '0'}
            icon={ShoppingBag}
            color="indigo"
          />
          <KPICard
            title="Units Sold"
            value={summary?.totalUnitsSold?.toLocaleString() || '0'}
                  icon={Package}
            color="green"
          />
          <KPICard
            title="Active SKUs"
            value={summary?.uniqueSKUs?.toLocaleString() || '0'}
            icon={Layers}
            color="purple"
          />
          <KPICard
            title="Products"
            value={summary?.uniqueProducts?.toLocaleString() || '0'}
            icon={Target}
            color="blue"
          />
          <KPICard
            title="Categories"
            value={summary?.uniqueCategories || '0'}
            icon={PieChartIcon}
            color="pink"
          />
          <KPICard
            title="Avg Units/Day"
            value={summary?.avgUnitsPerDay?.toLocaleString() || '0'}
            icon={Activity}
            color="orange"
                />
              </div>

        {/* Year-over-Year Comparison */}
        {yearOverYear && Object.keys(yearOverYear).length > 1 && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-indigo-600" />
              Year-over-Year Performance
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(yearOverYear).map(([year, data]) => (
                <div key={year} className="p-4 bg-gradient-to-br from-gray-50 to-indigo-50 rounded-xl">
                  <div className="text-2xl font-bold text-gray-900">{year}</div>
                  <div className="text-lg font-semibold text-indigo-600 mt-1">
                    {data.units?.toLocaleString()} units
                  </div>
                  {data.change !== null && (
                    <div className={`flex items-center mt-2 text-sm font-medium ${
                      data.change >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {data.change >= 0 ? (
                        <ArrowUpRight className="w-4 h-4 mr-1" />
                      ) : (
                        <ArrowDownRight className="w-4 h-4 mr-1" />
                      )}
                      {Math.abs(data.change)}% vs previous
                    </div>
                  )}
                  <div className="text-xs text-gray-500 mt-1">{data.skus} SKUs active</div>
              </div>
              ))}
              </div>
              </div>
        )}

        {/* Insights & Recommendations */}
        {insights && insights.length > 0 && (
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 rounded-2xl p-6 text-white">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Lightbulb className="w-5 h-5 mr-2" />
              AI Insights & Recommendations
                    </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {insights.map((insight, idx) => (
                <div key={idx} className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
                  <div className="flex items-start space-x-3">
                    <div className={`p-2 rounded-lg ${
                      insight.type === 'success' ? 'bg-green-500/20' :
                      insight.type === 'warning' ? 'bg-yellow-500/20' :
                      'bg-blue-500/20'
                    }`}>
                      {insight.type === 'success' ? <CheckCircle className="w-4 h-4" /> :
                       insight.type === 'warning' ? <AlertTriangle className="w-4 h-4" /> :
                       <Info className="w-4 h-4" />}
                    </div>
                    <div>
                      <h4 className="font-semibold text-sm">{insight.title}</h4>
                      <p className="text-sm text-white/80 mt-1">{insight.message}</p>
                    </div>
                  </div>
                </div>
              ))}
    </div>
          </div>
        )}

        {/* Charts Row 1: Monthly Trends & Category Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Monthly Sales Trend */}
          <CollapsibleSection
            title="Monthly Sales Trend"
            icon={<Activity className="w-5 h-5" />}
            isExpanded={expandedSections.trends}
            onToggle={() => toggleSection('trends')}
          >
            {monthlyTrends && monthlyTrends.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={processMonthlyData(monthlyTrends)}>
                  <defs>
                    <linearGradient id="colorUnits" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={COLORS.primary} stopOpacity={0.3}/>
                      <stop offset="95%" stopColor={COLORS.primary} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                    formatter={(value) => [value.toLocaleString(), 'Units']}
                  />
                  <Area type="monotone" dataKey="units" stroke={COLORS.primary} fill="url(#colorUnits)" strokeWidth={2} />
                  <Line type="monotone" dataKey="activeSKUs" stroke={COLORS.success} strokeWidth={2} dot={false} />
                  <Legend />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="No monthly trend data available" />
            )}
          </CollapsibleSection>

          {/* Category Distribution */}
          <CollapsibleSection
            title="Category Performance"
            icon={<PieChartIcon className="w-5 h-5" />}
            isExpanded={expandedSections.categories}
            onToggle={() => toggleSection('categories')}
          >
            {categoryPerformance && categoryPerformance.length > 0 ? (
        <div className="flex items-center">
                <div className="w-1/2">
                  <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie 
                        data={categoryPerformance.slice(0, 8)}
                        dataKey="totalUnits"
                        nameKey="category"
                  cx="50%" 
                  cy="50%" 
                  innerRadius={50} 
                        outerRadius={90}
                  paddingAngle={2}
                >
                        {categoryPerformance.slice(0, 8).map((entry, idx) => (
                          <Cell key={idx} fill={CHART_COLORS[idx % CHART_COLORS.length]} />
                  ))}
                </Pie>
                      <Tooltip formatter={(value) => value.toLocaleString()} />
              </PieChart>
            </ResponsiveContainer>
          </div>
                <div className="w-1/2 space-y-2 max-h-[250px] overflow-y-auto pr-2">
                  {categoryPerformance.slice(0, 8).map((cat, idx) => (
                    <div key={cat.category} className="flex items-center justify-between text-sm">
                <div className="flex items-center">
                        <div 
                          className="w-3 h-3 rounded-full mr-2" 
                          style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                        />
                        <span className="text-gray-700 truncate max-w-[120px]">{cat.category}</span>
                </div>
                <div className="text-right">
                        <span className="font-semibold text-gray-900">{cat.totalUnits.toLocaleString()}</span>
                        <span className="text-gray-400 text-xs ml-1">({cat.marketShare}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
            ) : (
              <EmptyState message="No category data available" />
            )}
          </CollapsibleSection>
        </div>

        {/* Charts Row 2: Quarterly & Seasonal */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Quarterly Trends */}
          {quarterlyTrends && quarterlyTrends.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-purple-600" />
                Quarterly Performance
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={quarterlyTrends}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis 
                    dataKey={(d) => `${d.year} ${d.quarter}`} 
                    tick={{ fontSize: 11 }} 
                  />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                    formatter={(value) => [value.toLocaleString(), 'Units']}
                  />
                  <Bar dataKey="units" fill={COLORS.purple} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Seasonal Analysis */}
          {seasonalAnalysis && seasonalAnalysis.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Clock className="w-5 h-5 mr-2 text-orange-600" />
                Seasonal Distribution
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={seasonalAnalysis} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis dataKey="season" type="category" width={100} tick={{ fontSize: 10 }} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                    formatter={(value) => [value.toLocaleString(), 'Units']}
                  />
                  <Bar dataKey="totalUnits" fill={COLORS.orange} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
      </div>
          )}
    </div>

        {/* Gender & Day of Week Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Gender Breakdown */}
          {genderBreakdown && genderBreakdown.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Users className="w-5 h-5 mr-2 text-pink-600" />
                Gender Distribution
              </h3>
              <div className="flex items-center justify-around">
                {genderBreakdown.map((g, idx) => {
                  const total = genderBreakdown.reduce((sum, item) => sum + item.units, 0);
                  const percentage = ((g.units / total) * 100).toFixed(1);
  return (
                    <div key={g.gender} className="text-center">
                      <div 
                        className="w-20 h-20 rounded-full flex items-center justify-center mx-auto mb-3"
                        style={{ backgroundColor: `${CHART_COLORS[idx]}20` }}
                      >
                        <span className="text-2xl font-bold" style={{ color: CHART_COLORS[idx] }}>
                          {percentage}%
                        </span>
                </div>
                      <div className="font-semibold text-gray-900">{g.gender}</div>
                      <div className="text-sm text-gray-500">{g.units.toLocaleString()} units</div>
                </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Day of Week Pattern */}
          {dayOfWeekPatterns && dayOfWeekPatterns.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Calendar className="w-5 h-5 mr-2 text-teal-600" />
                Sales by Day of Week
              </h3>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={dayOfWeekPatterns}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                  <XAxis dataKey="day" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 20px rgba(0,0,0,0.1)' }}
                    formatter={(value) => [value.toLocaleString(), 'Units']}
                  />
                  <Bar dataKey="units" fill={COLORS.teal} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* Top & Slow Moving Products */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Top Products */}
          <CollapsibleSection
            title="Top Performing Products"
            icon={<Award className="w-5 h-5" />}
            isExpanded={expandedSections.products}
            onToggle={() => toggleSection('products')}
            badge={`Top ${topProducts?.length || 0}`}
            badgeColor="green"
          >
            {topProducts && topProducts.length > 0 ? (
              <div className="space-y-3 max-h-[350px] overflow-y-auto pr-2">
                {topProducts.slice(0, 10).map((product, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors">
                    <div className="flex items-center space-x-3">
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-bold text-sm ${
                        idx < 3 ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white' : 'bg-gray-200 text-gray-600'
                      }`}>
                        {product.rank}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900 text-sm">{product.productName}</div>
                        <div className="text-xs text-gray-500">{product.category} • {product.skuCount} SKUs</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-indigo-600">{product.totalUnits.toLocaleString()}</div>
                      <div className="text-xs text-gray-500">units</div>
      </div>
    </div>
                ))}
              </div>
            ) : (
              <EmptyState message="No product data available" />
            )}
          </CollapsibleSection>

          {/* Slow Moving Products */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <AlertTriangle className="w-5 h-5 mr-2 text-amber-600" />
                Slow-Moving Products
              </h3>
              <span className="px-2 py-1 bg-amber-100 text-amber-800 rounded-full text-xs font-medium">
                Bottom 10%
              </span>
            </div>
            {slowMovingProducts && slowMovingProducts.length > 0 ? (
              <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
                {slowMovingProducts.slice(0, 10).map((product, idx) => (
                  <div key={idx} className="flex items-center justify-between p-3 bg-amber-50 rounded-xl border border-amber-100">
                    <div>
                      <div className="font-medium text-gray-900 text-sm">{product.productName}</div>
                      <div className="text-xs text-gray-500">{product.category}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-bold text-amber-600">{product.totalUnits}</div>
                      <div className="text-xs text-gray-500">units</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState message="No slow-moving products identified" />
            )}
            <div className="mt-4 p-3 bg-amber-50 rounded-xl border border-amber-200">
              <p className="text-xs text-amber-800">
                <Lightbulb className="w-3 h-3 inline mr-1" />
                <strong>Recommendation:</strong> Consider promotional pricing or bundling strategies for these items.
              </p>
            </div>
          </div>
        </div>

        {/* Category Details Table */}
        {categoryPerformance && categoryPerformance.length > 0 && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
            <div className="p-6 border-b border-gray-100">
              <h3 className="text-lg font-semibold text-gray-900 flex items-center">
                <Layers className="w-5 h-5 mr-2 text-indigo-600" />
                Detailed Category Analysis
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Category</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Total Units</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Transactions</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">SKUs</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Market Share</th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-600 uppercase tracking-wider">Velocity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {categoryPerformance.map((cat, idx) => (
                    <tr key={cat.category} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-3" 
                            style={{ backgroundColor: CHART_COLORS[idx % CHART_COLORS.length] }}
                          />
                          <span className="font-medium text-gray-900">{cat.category}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right font-semibold text-gray-900">
                        {cat.totalUnits.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-right text-gray-600">
                        {cat.transactions.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-right text-gray-600">
                        {cat.uniqueSKUs}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className="px-2 py-1 bg-indigo-100 text-indigo-800 rounded-full text-xs font-medium">
                          {cat.marketShare}%
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right">
                        <span className={`font-medium ${cat.velocity > 50 ? 'text-green-600' : cat.velocity > 20 ? 'text-amber-600' : 'text-red-600'}`}>
                          {cat.velocity}
                        </span>
                        <span className="text-gray-400 text-xs ml-1">units/SKU</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center py-6 text-sm text-gray-500">
          <p>Data refreshed from training sales data • {summary?.totalTransactions?.toLocaleString()} transactions analyzed</p>
      </div>
      </main>
    </div>
  );
}

// Helper Components
function KPICard({ title, value, icon: Icon, color, change }) {
  const colorClasses = {
    indigo: 'from-indigo-500 to-indigo-600',
    green: 'from-emerald-500 to-emerald-600',
    purple: 'from-purple-500 to-purple-600',
    blue: 'from-blue-500 to-blue-600',
    pink: 'from-pink-500 to-pink-600',
    orange: 'from-orange-500 to-orange-600',
    red: 'from-red-500 to-red-600',
    teal: 'from-teal-500 to-teal-600'
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]} text-white`}>
          <Icon className="w-4 h-4" />
        </div>
        {change && (
          <span className={`text-xs font-medium ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {change >= 0 ? '+' : ''}{change}%
          </span>
        )}
      </div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-xs text-gray-500 mt-1">{title}</div>
    </div>
  );
}

function CollapsibleSection({ title, icon, isExpanded, onToggle, children, badge, badgeColor = 'blue' }) {
  const badgeColors = {
    blue: 'bg-blue-100 text-blue-800',
    green: 'bg-green-100 text-green-800',
    amber: 'bg-amber-100 text-amber-800'
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full p-6 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center space-x-3">
          <div className="text-indigo-600">{icon}</div>
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          {badge && (
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${badgeColors[badgeColor]}`}>
              {badge}
            </span>
          )}
            </div>
        {isExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
      </button>
      {isExpanded && (
        <div className="px-6 pb-6">
          {children}
        </div>
      )}
    </div>
  );
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
      <Package className="w-12 h-12 mb-3 opacity-50" />
      <p className="text-sm">{message}</p>
    </div>
  );
}

// Helper function to process monthly data
function processMonthlyData(monthlyTrends) {
  // Group by year-month and create readable labels
  const processed = monthlyTrends.map(item => ({
    ...item,
    label: `${item.monthName} ${String(item.year).slice(2)}`
  }));
  
  // Sort by year and month
  return processed.sort((a, b) => {
    if (a.year !== b.year) return a.year - b.year;
    return a.month - b.month;
  });
  }
  