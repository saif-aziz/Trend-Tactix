import React, { useState, useEffect } from 'react';
import {
  Brain,
  TrendingUp,
  Package,
  Store,
  Shuffle,
  BarChart3,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Download,
  Upload,
  Search,
  Filter,
  Target,
  MapPin,
  Calendar,
  Zap,
  Wifi,
  WifiOff,
  Database,
  AlertCircle,
  ArrowRight,
  Grid3X3,
  PieChart,
  ShoppingCart,
  ChevronDown,
  ChevronUp,
  Check,
  X,
  Plus,
  Minus,
  Eye,
  EyeOff
} from 'lucide-react';

// Mock API Service with proper async delays
const apiService = {
  processCSVData: async (csvData) => {
    // Add delay to simulate real API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const lines = csvData.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    const products = {};
    const salesData = [];
    
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      const row = {};
      
      headers.forEach((header, index) => {
        row[header] = values[index];
      });
      
      if (row['Product Code'] && row['Sale Date']) {
        salesData.push(row);
        
        const productCode = row['Product Code'];
        if (!products[productCode]) {
          products[productCode] = {
            id: productCode,
            productCode: productCode,
            name: row['Product Name'] || productCode,
            category: row['Category'] || 'Unknown',
            gender: row['Gender'] || 'Unisex',
            season: row['Season'] || 'Unknown',
            sizeName: row['Size Name'] || 'OS',
            sizeCode: row['Size Code'] || 'OS',
            colorName: row['Color Name'] || 'Default',
            colorCode: row['Color Code'] || 'DEF',
            attributes: {
              size: row['Size Name'] || 'OS',
              color: row['Color Name'] || 'Default',
              gender: row['Gender'] || 'Unisex'
            },
            totalQuantity: 0,
            historicalSales: 0
          };
        }
        products[productCode].historicalSales += 1;
      }
    }
    
    return { products: Object.values(products), salesData };
  },

  generateForecast: async (salesData, productCode) => {
    // Add delay to simulate AI processing
    await new Promise(resolve => setTimeout(resolve, 50));
    
    const productSales = salesData.filter(sale => sale['Product Code'] === productCode);
    
    if (productSales.length === 0) {
      return {
        predictedDemand: 0,
        confidence: 0,
        riskLevel: 'HIGH',
        reasoning: 'No historical sales data available'
      };
    }

    const salesByMonth = {};
    productSales.forEach(sale => {
      const month = new Date(sale['Sale Date']).getMonth();
      salesByMonth[month] = (salesByMonth[month] || 0) + 1;
    });

    const totalSales = productSales.length;
    const dateRange = productSales.map(s => new Date(s['Sale Date']));
    const minDate = new Date(Math.min(...dateRange));
    const maxDate = new Date(Math.max(...dateRange));
    const monthsSpan = Math.max(1, (maxDate - minDate) / (1000 * 60 * 60 * 24 * 30));
    
    const velocity = totalSales / monthsSpan;
    const predictedDemand = Math.round(velocity * 3);
    
    const avgMonthlySales = Object.values(salesByMonth).reduce((a, b) => a + b, 0) / Object.keys(salesByMonth).length;
    const variance = Object.values(salesByMonth).reduce((sum, sales) => sum + Math.pow(sales - avgMonthlySales, 2), 0) / Object.keys(salesByMonth).length;
    const coefficient_of_variation = Math.sqrt(variance) / avgMonthlySales;
    
    let confidence = Math.max(0, Math.min(100, 100 - (coefficient_of_variation * 50)));
    if (totalSales < 5) confidence *= 0.5;
    
    let riskLevel = 'LOW';
    if (confidence < 30 || totalSales < 3) riskLevel = 'HIGH';
    else if (confidence < 60 || totalSales < 10) riskLevel = 'MEDIUM';
    
    return {
      predictedDemand: Math.max(1, predictedDemand),
      confidence: Math.round(confidence),
      riskLevel,
      reasoning: `Based on ${totalSales} historical sales over ${monthsSpan.toFixed(1)} months. Average velocity: ${velocity.toFixed(1)} sales/month.`
    };
  },

  generateDistribution: async (salesData, productCode, forecast) => {
    // Add delay to simulate distribution calculation
    // await new Promise(resolve => setTimeout(resolve, 600));
    
    const product = salesData.find(sale => sale['Product Code'] === productCode);
    
    if (!product || !forecast) {
      return [];
    }

    const variations = salesData
      .filter(sale => sale['Product Code'] === productCode)
      .reduce((acc, sale) => {
        const key = `${sale['Size Code']}-${sale['Color Code']}`;
        if (!acc[key]) {
          acc[key] = {
            size: sale['Size Name'] || sale['Size Code'],
            color: sale['Color Name'] || sale['Color Code'],
            sizeCode: sale['Size Code'],
            colorCode: sale['Color Code'],
            salesCount: 0
          };
        }
        acc[key].salesCount += 1;
        return acc;
      }, {});

    const variationList = Object.values(variations);
    const totalVariationSales = variationList.reduce((sum, v) => sum + v.salesCount, 0);

    const distributions = variationList.map(variation => {
      const proportion = totalVariationSales > 0 ? variation.salesCount / totalVariationSales : 1 / variationList.length;
      const allocatedQuantity = Math.max(1, Math.round(forecast.predictedDemand * proportion));
      
      return {
        shopId: 1,
        productCode: productCode,
        variation: variation,
        allocatedQuantity: allocatedQuantity,
        reasoning: `${(proportion * 100).toFixed(1)}% allocation based on historical sales performance (${variation.salesCount} sales)`
      };
    });

    return distributions;
  }
};

// Loading Component with better animation
function LoadingSpinner({ message = "Loading...", subMessage = "" }) {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="text-center">
        <div className="relative">
          <RefreshCw className="w-12 h-12 animate-spin text-purple-600 mx-auto mb-4" />
          <div className="absolute inset-0 w-12 h-12 border-4 border-purple-200 rounded-full mx-auto animate-pulse"></div>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">{message}</h3>
        {subMessage && <p className="text-sm text-gray-600">{subMessage}</p>}
        <div className="mt-4 flex justify-center">
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
            <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Error Component
function ErrorMessage({ error, onRetry }) {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
      <div className="flex items-center">
        <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
        <div className="flex-1">
          <h3 className="text-red-800 font-medium">Error</h3>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-red-600 hover:text-red-800 text-sm font-medium"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

// Connection Status Component
function ConnectionStatus({ isDataLoaded, modelStatus }) {
  return (
    <div className="flex items-center space-x-4">
      <div className="flex items-center space-x-2">
        {isDataLoaded ? (
          <Database className="w-4 h-4 text-green-500" />
        ) : (
          <Database className="w-4 h-4 text-red-500" />
        )}
        <span className="text-sm text-gray-600">
          Data: {isDataLoaded ? 'Loaded' : 'Not Loaded'}
        </span>
      </div>

      <div className="flex items-center space-x-2">
        <div className={`w-3 h-3 rounded-full ${
          modelStatus === 'READY' ? 'bg-green-500' :
          modelStatus === 'PROCESSING' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
        }`}></div>
        <span className="text-sm text-gray-600">
          Model: {modelStatus === 'READY' ? 'Ready' : modelStatus || 'Not Ready'}
        </span>
      </div>
    </div>
  );
}

// Data Upload Component
function DataUploadSection({ onDataLoad, isLoading }) {
  const [dragOver, setDragOver] = useState(false);

  const handleFileUpload = async (file) => {
    if (file && file.type === 'text/csv') {
      const text = await file.text();
      await onDataLoad(text);
    } else {
      alert('Please upload a CSV file');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    handleFileUpload(file);
  };

  return (
    <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <Database className="w-5 h-5 text-blue-600 mr-2" />
          <div>
            <h3 className="font-medium text-blue-900">Historical Sales Data</h3>
            <p className="text-sm text-blue-700">Upload your sales CSV file to train the AI forecasting model</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <div
            className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${
              dragOver ? 'border-blue-500 bg-blue-100' : 'border-gray-300'
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept=".csv"
              onChange={handleFileInput}
              className="hidden"
              id="file-upload"
              disabled={isLoading}
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="w-6 h-6 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600">
                {isLoading ? 'Processing...' : 'Drop CSV file here or click to upload'}
              </p>
            </label>
          </div>
        </div>
      </div>
    </div>
  );
}

// Analytics Dashboard Component
function AnalyticsDashboard({ salesData, onShowCategories }) {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (salesData.length > 0) {
      const categoryYearData = {};
      
      salesData.forEach(sale => {
        const year = new Date(sale['Sale Date']).getFullYear();
        const category = sale['Category'];
        
        if (!categoryYearData[category]) {
          categoryYearData[category] = { 2022: 0, 2023: 0, 2024: 0, total: 0 };
        }
        
        if (year >= 2022 && year <= 2024) {
          categoryYearData[category][year] += 1;
          categoryYearData[category].total += 1;
        }
      });

      const topCategories = Object.entries(categoryYearData)
        .sort(([,a], [,b]) => b.total - a.total)
        .slice(0, 10)
        .map(([category, data]) => ({ category, ...data }));

      setChartData(topCategories);
    }
  }, [salesData]);

  const maxSales = Math.max(...chartData.map(item => item.total));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Sales Analytics Dashboard</h2>
          <p className="text-gray-600">Top 10 Categories Performance (2022-2024)</p>
        </div>
        <button
          onClick={onShowCategories}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center"
        >
          <Grid3X3 className="w-4 h-4 mr-2" />
          View Categories
        </button>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="space-y-4">
          {chartData.map((item, index) => (
            <div key={item.category} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium text-gray-900">{item.category}</span>
                <span className="text-sm text-gray-500">{item.total.toLocaleString()} sales</span>
              </div>
              
              <div className="flex h-8 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="bg-blue-500 flex items-center justify-center text-xs text-white font-medium"
                  style={{ width: `${(item[2022] / maxSales) * 100}%` }}
                  title={`2022: ${item[2022]} sales`}
                >
                  {item[2022] > 50 ? '2022' : ''}
                </div>
                <div 
                  className="bg-green-500 flex items-center justify-center text-xs text-white font-medium"
                  style={{ width: `${(item[2023] / maxSales) * 100}%` }}
                  title={`2023: ${item[2023]} sales`}
                >
                  {item[2023] > 50 ? '2023' : ''}
                </div>
                <div 
                  className="bg-purple-500 flex items-center justify-center text-xs text-white font-medium"
                  style={{ width: `${(item[2024] / maxSales) * 100}%` }}
                  title={`2024: ${item[2024]} sales`}
                >
                  {item[2024] > 50 ? '2024' : ''}
                </div>
              </div>
              
              <div className="flex justify-between text-xs text-gray-500">
                <span>2022: {item[2022]}</span>
                <span>2023: {item[2023]}</span>
                <span>2024: {item[2024]}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-center space-x-6">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded mr-2"></div>
            <span className="text-sm text-gray-600">2022</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded mr-2"></div>
            <span className="text-sm text-gray-600">2023</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-purple-500 rounded mr-2"></div>
            <span className="text-sm text-gray-600">2024</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-blue-600">{salesData.length.toLocaleString()}</div>
          <div className="text-sm text-gray-600">Total Sales</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-green-600">{chartData.length}</div>
          <div className="text-sm text-gray-600">Top Categories</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {chartData.length > 0 ? Math.round(chartData.reduce((sum, item) => sum + item.total, 0) / chartData.length) : 0}
          </div>
          <div className="text-sm text-gray-600">Avg Sales/Category</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-orange-600">3</div>
          <div className="text-sm text-gray-600">Years Data</div>
        </div>
      </div>
    </div>
  );
}

// Categories Grid Component
function CategoriesGrid({ salesData, onSelectCategory, onBackToAnalytics }) {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    if (salesData.length > 0) {
      const categoryStats = salesData.reduce((acc, sale) => {
        const category = sale['Category'];
        if (!acc[category]) {
          acc[category] = {
            name: category,
            totalSales: 0,
            uniqueProducts: new Set(),
            years: new Set()
          };
        }
        acc[category].totalSales += 1;
        acc[category].uniqueProducts.add(sale['Product Name']);
        acc[category].years.add(new Date(sale['Sale Date']).getFullYear());
        return acc;
      }, {});

      const categoriesArray = Object.values(categoryStats).map(cat => ({
        ...cat,
        uniqueProducts: cat.uniqueProducts.size,
        years: Array.from(cat.years).sort()
      })).sort((a, b) => b.totalSales - a.totalSales);

      setCategories(categoriesArray);
    }
  }, [salesData]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={onBackToAnalytics}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-2"
          >
            <ArrowRight className="w-4 h-4 mr-2 rotate-180" />
            Back to Analytics
          </button>
          <h2 className="text-2xl font-bold text-gray-900">Product Categories</h2>
          <p className="text-gray-600">{categories.length} categories available</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {categories.map((category) => (
          <div
            key={category.name}
            onClick={() => onSelectCategory(category.name)}
            className="bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer border-2 border-transparent hover:border-purple-500 p-6"
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                <Package className="w-8 h-8 text-white" />
              </div>
              
              <h3 className="text-lg font-semibold text-gray-900 mb-2">{category.name}</h3>
              
              <div className="space-y-2 text-sm text-gray-600">
                <div className="flex justify-between">
                  <span>Total Sales:</span>
                  <span className="font-medium">{category.totalSales.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Products:</span>
                  <span className="font-medium">{category.uniqueProducts}</span>
                </div>
                <div className="flex justify-between">
                  <span>Data Years:</span>
                  <span className="font-medium">{category.years.join(', ')}</span>
                </div>
              </div>
              
              <div className="mt-4 px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                Click to View Products
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Split Screen View Component with Product-Level View
function SplitScreenView({ 
  category, 
  products, 
  salesData, 
  selectedProducts,
  onSelectProduct,
  onBackToCategories,
  distributions,
  forecasts,
  onGenerateDistribution,
  isLoading,
  isGeneratingDistribution
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [showSKULevel, setShowSKULevel] = useState(false);
  const [expandedProducts, setExpandedProducts] = useState(new Set());
  const [showOnlySelected, setShowOnlySelected] = useState(false);

  // Monitor when distributions are generated to hide other products
  useEffect(() => {
    const hasAnyDistribution = selectedProducts.some(product => 
      distributions[product.productCode]?.data?.length > 0
    );
    // Also check if we're currently generating distributions for selected products
    const isGeneratingForSelected = isGeneratingDistribution && selectedProducts.length > 0;
    
    setShowOnlySelected(hasAnyDistribution || isGeneratingForSelected);
  }, [selectedProducts, distributions, isGeneratingDistribution]);

  // Group SKUs by Product Name to create product-level view
  const groupedProducts = products
    .filter(product => product.category === category)
    .reduce((acc, sku) => {
      const productName = sku.name;
      if (!acc[productName]) {
        acc[productName] = {
          name: productName,
          category: sku.category,
          gender: sku.gender,
          season: sku.season,
          skus: [],
          totalSales: 0,
          totalQuantity: 0
        };
      }
      acc[productName].skus.push(sku);
      acc[productName].totalSales += sku.historicalSales;
      acc[productName].totalQuantity += sku.totalQuantity;
      return acc;
    }, {});

  const productsList = Object.values(groupedProducts);

  const filteredProducts = productsList.filter(product => {
    const matchesSearch = product.name?.toLowerCase().includes(searchTerm.toLowerCase());
    
    // If distributions are generated or we're loading, only show selected products
    if (showOnlySelected) {
      const hasSelectedSKUs = product.skus.some(sku => 
        selectedProducts.some(selected => selected.productCode === sku.productCode)
      );
      return matchesSearch && hasSelectedSKUs;
    }
    
    return matchesSearch;
  });

  const filteredSKUs = products.filter(product => {
    const matchesSearch = product.category === category &&
      (product.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
       product.productCode?.toLowerCase().includes(searchTerm.toLowerCase()));
    
    // If distributions are generated or we're loading, only show selected SKUs
    if (showOnlySelected) {
      const isSelected = selectedProducts.some(selected => 
        selected.productCode === product.productCode
      );
      return matchesSearch && isSelected;
    }
    
    return matchesSearch;
  });

  const toggleProductExpansion = (productName) => {
    const newExpanded = new Set(expandedProducts);
    if (newExpanded.has(productName)) {
      newExpanded.delete(productName);
    } else {
      newExpanded.add(productName);
    }
    setExpandedProducts(newExpanded);
  };

  const selectedProductCodes = selectedProducts.map(p => p.productCode);

  return (
    <div className="h-full flex flex-col relative">
      {/* Header with Multi-Select Actions */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <button
              onClick={onBackToCategories}
              className="flex items-center text-gray-600 hover:text-gray-900 mb-2"
            >
              <ArrowRight className="w-4 h-4 mr-2 rotate-180" />
              Back to Categories
            </button>
            <h2 className="text-xl font-bold text-gray-900">{category}</h2>
            <p className="text-sm text-gray-600">
              {showSKULevel ? filteredSKUs.length : filteredProducts.length} {showSKULevel ? 'SKUs' : 'products'} 
              {selectedProducts.length > 0 && ` • ${selectedProducts.length} selected`}
            </p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Multi-Select Actions */}
            {selectedProducts.length > 0 && (
              <div className="flex items-center space-x-2 bg-purple-50 rounded-lg p-2">
                <span className="text-sm font-medium text-purple-700">
                  {selectedProducts.length} selected
                </span>
                <button
                  onClick={() => onGenerateDistribution(selectedProducts.map(p => p.productCode))}
                  disabled={isGeneratingDistribution}
                  className="px-3 py-1 bg-purple-600 text-white rounded text-sm font-medium hover:bg-purple-700 disabled:opacity-50 transition-colors"
                >
                  {isGeneratingDistribution ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Zap className="w-3 h-3 mr-1 inline" />
                      Generate Distribution
                    </>
                  )}
                </button>
                <button
                  onClick={() => onSelectProduct(null, 'clear')}
                  className="p-1 text-gray-500 hover:text-gray-700"
                  title="Clear selection"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Reset Button to Show All Products */}
            {showOnlySelected && !isGeneratingDistribution && (
              <button
                onClick={() => {
                  setShowOnlySelected(false);
                  onSelectProduct(null, 'clear');
                }}
                className="px-3 py-2 bg-gray-100 text-gray-700 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors"
              >
                <Eye className="w-4 h-4 mr-1 inline" />
                Show All Products
              </button>
            )}

            {/* View Toggle */}
            <button
              onClick={() => setShowSKULevel(!showSKULevel)}
              className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                showSKULevel
                  ? 'bg-orange-100 text-orange-800'
                  : 'bg-blue-100 text-blue-800'
              }`}
            >
              {showSKULevel ? (
                <>
                  <EyeOff className="w-4 h-4 mr-1 inline" />
                  Show Product Level
                </>
              ) : (
                <>
                  <Eye className="w-4 h-4 mr-1 inline" />
                  Show SKU Level
                </>
              )}
            </button>

            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder={`Search ${showSKULevel ? 'SKUs' : 'products'}...`}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Split Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Half - Products/SKUs */}
        <div className="w-1/2 border-r border-gray-200 overflow-y-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {showSKULevel ? (
              // SKU Level View
              filteredSKUs.map(sku => (
                <ProductCard
                  key={sku.id}
                  product={sku}
                  onSelect={onSelectProduct}
                  isSelected={selectedProductCodes.includes(sku.productCode)}
                  showPredictions={true}
                  distributionStatus={distributions[sku.productCode]?.status}
                  onGenerateDistribution={onGenerateDistribution}
                  forecast={forecasts[sku.productCode]}
                  compact={true}
                  multiSelect={true}
                />
              ))
            ) : (
              // Product Level View
              filteredProducts.map(product => (
                <ProductLevelCard
                  key={product.name}
                  product={product}
                  selectedProducts={selectedProducts}
                  onSelectProduct={onSelectProduct}
                  isExpanded={expandedProducts.has(product.name)}
                  onToggleExpansion={() => toggleProductExpansion(product.name)}
                  distributions={distributions}
                  forecasts={forecasts}
                />
              ))
            )}
          </div>
        </div>

        {/* Right Half - Distribution */}
        <div className="w-1/2 overflow-y-auto p-4">
          {selectedProducts.length > 0 ? (
            <MultiDistributionPanel
              selectedProducts={selectedProducts}
              distributions={distributions}
              forecasts={forecasts}
              onGenerateDistribution={() => onGenerateDistribution(selectedProducts.map(p => p.productCode))}
              isLoading={isLoading}
              isGeneratingDistribution={isGeneratingDistribution}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-center">
              <div>
                <ShoppingCart className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Select Products</h3>
                <p className="text-gray-600">
                  Choose one or more products from the left to view distribution options
                </p>
                <div className="mt-4 text-sm text-gray-500">
                  <p>💡 Tip: You can select multiple products for batch distribution</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Full Screen Loading Overlay for Distribution Generation */}
      {isGeneratingDistribution && selectedProducts.length > 0 && (
        <div className="fixed inset-0 bg-white bg-opacity-95 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-8 text-center border border-gray-200 max-w-md mx-4">
            <div className="relative mb-6">
              <RefreshCw className="w-16 h-16 animate-spin text-purple-600 mx-auto" />
              <div className="absolute inset-0 w-16 h-16 border-4 border-purple-200 rounded-full mx-auto animate-pulse"></div>
            </div>
            
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Generating AI Distribution
            </h3>
            
            <p className="text-sm text-gray-600 mb-2">
              Processing {selectedProducts.length} product{selectedProducts.length > 1 ? 's' : ''}
            </p>
            
            <p className="text-xs text-gray-500 mb-6">
              AI is analyzing sales patterns, seasonality trends, and optimizing stock allocation for maximum performance
            </p>
            
            <div className="flex justify-center mb-4">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
            
            <div className="text-xs text-purple-600 font-medium">
              Please wait, this may take a few moments...
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Product Level Card Component
function ProductLevelCard({ 
  product, 
  selectedProducts, 
  onSelectProduct, 
  isExpanded, 
  onToggleExpansion,
  distributions,
  forecasts 
}) {
  const selectedProductCodes = selectedProducts.map(p => p.productCode);
  const selectedSKUs = product.skus.filter(sku => selectedProductCodes.includes(sku.productCode));
  const allSKUsSelected = product.skus.length === selectedSKUs.length;
  const someSKUsSelected = selectedSKUs.length > 0;

  const handleProductSelect = () => {
    if (allSKUsSelected) {
      // Deselect all SKUs
      product.skus.forEach(sku => {
        onSelectProduct(sku, 'remove');
      });
    } else {
      // Select all SKUs
      product.skus.forEach(sku => {
        if (!selectedProductCodes.includes(sku.productCode)) {
          onSelectProduct(sku, 'add');
        }
      });
    }
  };

  const avgConfidence = product.skus.reduce((sum, sku) => {
    const forecast = forecasts[sku.productCode];
    return sum + (forecast?.confidence || 0);
  }, 0) / product.skus.length;

  const totalPredictedDemand = product.skus.reduce((sum, sku) => {
    const forecast = forecasts[sku.productCode];
    return sum + (forecast?.predictedDemand || 0);
  }, 0);

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 border-2 border-transparent hover:border-purple-300 p-4">
      <div className="space-y-3">
        {/* Header with Multi-Select Checkbox */}
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3">
            <button
              onClick={handleProductSelect}
              className={`mt-1 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                allSKUsSelected
                  ? 'bg-purple-600 border-purple-600 text-white'
                  : someSKUsSelected
                  ? 'bg-purple-100 border-purple-400'
                  : 'border-gray-300 hover:border-purple-400'
              }`}
            >
              {allSKUsSelected && <Check className="w-3 h-3" />}
              {someSKUsSelected && !allSKUsSelected && <Minus className="w-3 h-3 text-purple-600" />}
            </button>
            
            <div>
              <h3 className="text-base font-semibold text-gray-900">{product.name}</h3>
              <p className="text-sm text-gray-500">{product.skus.length} SKUs available</p>
            </div>
          </div>
          
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
            {product.category}
          </span>
        </div>

        {/* Product Summary */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Total Sales:</span>
            <span className="font-medium">{product.totalSales}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Predicted:</span>
            <span className="font-medium">{totalPredictedDemand}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">SKUs:</span>
            <span className="font-medium">{product.skus.length}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Avg Confidence:</span>
            <span className="font-medium">{Math.round(avgConfidence)}%</span>
          </div>
        </div>

        {/* Selection Status */}
        {someSKUsSelected && (
          <div className="p-2 bg-purple-50 rounded-lg">
            <p className="text-sm text-purple-700">
              {selectedSKUs.length} of {product.skus.length} SKUs selected
            </p>
          </div>
        )}

        {/* Expand/Collapse SKUs */}
        <button
          onClick={onToggleExpansion}
          className="w-full flex items-center justify-center py-2 text-sm font-medium text-gray-600 hover:text-gray-900 border-t border-gray-200"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="w-4 h-4 mr-1" />
              Hide SKUs
            </>
          ) : (
            <>
              <ChevronDown className="w-4 h-4 mr-1" />
              Show SKUs
            </>
          )}
        </button>

        {/* Expanded SKU List */}
        {isExpanded && (
          <div className="space-y-2 border-t border-gray-200 pt-3">
            {product.skus.map(sku => (
              <div
                key={sku.productCode}
                className={`p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                  selectedProductCodes.includes(sku.productCode)
                    ? 'border-purple-300 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-200'
                }`}
                onClick={() => onSelectProduct(sku, 
                  selectedProductCodes.includes(sku.productCode) ? 'remove' : 'add'
                )}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{sku.productCode}</p>
                    <p className="text-xs text-gray-500">
                      {sku.attributes.size} • {sku.attributes.color}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-medium">{sku.historicalSales} sales</p>
                    {forecasts[sku.productCode] && (
                      <p className="text-xs text-gray-500">
                        Pred: {forecasts[sku.productCode].predictedDemand}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Enhanced Product Card Component with Multi-Select
function ProductCard({ 
  product, 
  onSelect, 
  isSelected, 
  showPredictions, 
  distributionStatus, 
  onGenerateDistribution,
  forecast,
  compact = false,
  multiSelect = false 
}) {
  const riskColors = {
    LOW: 'text-green-600 bg-green-100',
    MEDIUM: 'text-yellow-600 bg-yellow-100',
    HIGH: 'text-red-600 bg-red-100'
  };

  const handleClick = () => {
    if (multiSelect) {
      onSelect(product, isSelected ? 'remove' : 'add');
    } else {
      onSelect(product);
    }
  };

  return (
    <div
      className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer border-2 ${
        isSelected ? 'border-purple-500 ring-2 ring-purple-200 bg-purple-50' : 'border-transparent'
      } ${compact ? 'p-4' : 'p-6'}`}
      onClick={handleClick}
    >
      <div className="space-y-3">
        <div className="flex justify-between items-start">
          <div className="flex items-start space-x-3">
            {multiSelect && (
              <div
                className={`mt-1 w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                  isSelected
                    ? 'bg-purple-600 border-purple-600 text-white'
                    : 'border-gray-300'
                }`}
              >
                {isSelected && <Check className="w-3 h-3" />}
              </div>
            )}
            
            <div>
              <h3 className={`${compact ? 'text-base' : 'text-lg'} font-semibold text-gray-900`}>
                {product.name}
              </h3>
              <p className="text-sm text-gray-500">{product.productCode}</p>
            </div>
          </div>
          
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
            {product.category}
          </span>
        </div>

        {!compact && (
          <div className="flex flex-wrap gap-2">
            <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
              {product.attributes.size}
            </span>
            <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
              {product.attributes.color}
            </span>
            <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-800">
              {product.attributes.gender}
            </span>
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Sales:</span>
          <span className="text-lg font-semibold text-gray-900">{product.historicalSales?.toLocaleString() || 0}</span>
        </div>

        {showPredictions && forecast && (
          <div className="border-t pt-3 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Demand:</span>
              <span className="font-semibold">{forecast.predictedDemand}</span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Confidence:</span>
              <span className="font-semibold">{forecast.confidence}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Risk:</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${riskColors[forecast.riskLevel]}`}>
                {forecast.riskLevel}
              </span>
            </div>
          </div>
        )}

        {distributionStatus === 'COMPLETE' && (
          <div className="p-2 bg-green-50 rounded-lg">
            <div className="flex items-center">
              <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
              <span className="text-sm font-medium text-green-800">Distribution Ready</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Multi Distribution Panel Component
function MultiDistributionPanel({ selectedProducts, distributions, forecasts, onGenerateDistribution, isLoading, isGeneratingDistribution }) {
  const [isExporting, setIsExporting] = useState(false);

  const shop = {
    id: 1,
    name: "(S-12) Packages Mall Lahore",
    location: "Lahore",
    tier: "FLAGSHIP"
  };

  const hasDistributions = selectedProducts.some(product => 
    distributions[product.productCode]?.data?.length > 0
  );

  const allDistributions = selectedProducts.reduce((acc, product) => {
    const productDistributions = distributions[product.productCode]?.data || [];
    return [...acc, ...productDistributions];
  }, []);

  const totalAllocated = allDistributions.reduce((sum, dist) => sum + dist.allocatedQuantity, 0);

  const handleExport = async () => {
    try {
      setIsExporting(true);
      
      const exportData = {
        exportDate: new Date().toISOString(),
        shop: shop,
        selectedProducts: selectedProducts.length,
        totalAllocated: totalAllocated,
        products: selectedProducts.map(product => ({
          productCode: product.productCode,
          productName: product.name,
          forecast: forecasts[product.productCode],
          distributions: distributions[product.productCode]?.data || []
        })),
        modelVersion: 'Sequential_Forecasting_v1.0'
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `multi_distribution_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

    } catch (error) {
      alert(`Export failed: ${error.message}`);
    } finally {
      setIsExporting(false);
    }
  };

  if (!hasDistributions) {
    return (
      <div className="h-full flex flex-col">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            Multi-Product Distribution Plan
          </h3>
          
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-3">Selected Products ({selectedProducts.length})</h4>
            <div className="space-y-2">
              {selectedProducts.map(product => {
                const forecast = forecasts[product.productCode];
                return (
                  <div key={product.productCode} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <div>
                      <div className="font-medium text-gray-900">{product.name}</div>
                      <div className="text-sm text-gray-500">{product.productCode}</div>
                    </div>
                    <div className="text-right">
                      {forecast ? (
                        <>
                          <div className="font-semibold text-purple-600">{forecast.predictedDemand}</div>
                          <div className="text-xs text-gray-500">{forecast.confidence}% confidence</div>
                        </>
                      ) : (
                        <div className="text-sm text-gray-400">No forecast</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="text-center py-8">
            <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 mb-4">No distributions generated yet</p>
            <button
              onClick={onGenerateDistribution}
              disabled={isGeneratingDistribution}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 transition-colors"
            >
              {isGeneratingDistribution ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2 inline" />
                  Generate Distribution for All Products
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-gray-900">
            Multi-Product Distribution Plan
          </h3>
          <div className="text-sm text-gray-500">
            Total: {totalAllocated} units across {selectedProducts.length} SKU's
          </div>
        </div>

        {/* Action Buttons - Moved to Top */}
        <div className="flex justify-between items-center mb-6 p-4 bg-gray-50 rounded-lg border">
          <div className="text-sm text-gray-600">
            <Brain className="w-4 h-4 inline mr-1" />
            AI-optimized multi-product distribution
          </div>

          <div className="flex space-x-3">
            <button 
              onClick={onGenerateDistribution}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Regenerate All
            </button>
            <button
              onClick={handleExport}
              disabled={isExporting}
              className="px-4 py-2 bg-green-600 text-white rounded-md text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
            >
              {isExporting ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4 mr-2 inline" />
                  Export All to POS
                </>
              )}
            </button>
          </div>
        </div>

        {/* Shop Info */}
        <div className="border border-gray-200 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h4 className="font-semibold text-gray-900">{shop.name}</h4>
              <p className="text-sm text-gray-600">{shop.location}</p>
            </div>
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
              {shop.tier}
            </span>
          </div>

          {/* Products Summary - Scrollable Area */}
          <div className="space-y-4">
            <h5 className="font-medium text-gray-700">Distribution Summary</h5>
            <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
              {selectedProducts.map(product => {
                const productDistributions = distributions[product.productCode]?.data || [];
                const productTotal = productDistributions.reduce((sum, dist) => sum + dist.allocatedQuantity, 0);
                const forecast = forecasts[product.productCode];
                
                return (
                  <div key={product.productCode} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <div className="font-medium text-gray-900">{product.name}</div>
                        <div className="text-sm text-gray-500">{product.productCode}</div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-semibold text-purple-600">{productTotal}</div>
                        <div className="text-xs text-gray-500">units allocated</div>
                      </div>
                    </div>
                    
                    {productDistributions.length > 0 && (
                      <div className="mt-2 text-xs text-gray-600">
                        {productDistributions.length} variations distributed
                      </div>
                    )}
                    
                    {forecast && (
                      <div className="mt-2 flex justify-between text-xs text-gray-500">
                        <span>Predicted: {forecast.predictedDemand}</span>
                        <span>Confidence: {forecast.confidence}%</span>
                        <span className={`px-2 py-1 rounded-full ${
                          forecast.riskLevel === 'LOW' ? 'bg-green-100 text-green-700' :
                          forecast.riskLevel === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-red-100 text-red-700'
                        }`}>
                          {forecast.riskLevel}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main Component
function InitialStockDistribution() {
  // View State Management
  const [currentView, setCurrentView] = useState('upload');
  const [selectedCategory, setSelectedCategory] = useState(null);
  
  // Data State Management
  const [selectedProducts, setSelectedProducts] = useState([]);
  const [products, setProducts] = useState([]);
  const [salesData, setSalesData] = useState([]);
  const [distributions, setDistributions] = useState({});
  const [forecasts, setForecasts] = useState({});
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [modelStatus, setModelStatus] = useState('NOT_READY');
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingDistribution, setIsGeneratingDistribution] = useState(false);
  const [error, setError] = useState(null);

  const handleDataLoad = async (csvData) => {
    try {
      setIsLoading(true);
      setError(null);
      setModelStatus('PROCESSING');

      const { products: processedProducts, salesData: processedSalesData } = await apiService.processCSVData(csvData);
      
      const productsWithQuantities = processedProducts.map(product => ({
        ...product,
        totalQuantity: Math.max(50, product.historicalSales * 2)
      }));

      setProducts(productsWithQuantities);
      setSalesData(processedSalesData);
      setIsDataLoaded(true);
      setModelStatus('READY');
      setCurrentView('analytics');

      console.log(`Loaded ${processedProducts.length} products and ${processedSalesData.length} sales records`);

    } catch (error) {
      setError(`Data processing failed: ${error.message}`);
      setModelStatus('ERROR');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateDistribution = async (productCodes) => {
    console.log('🚀 Generate Distribution clicked!', productCodes);
    
    try {
      // Set loading state IMMEDIATELY for instant feedback
      setIsGeneratingDistribution(true);
      console.log('✅ Loading state set to true');
      setError(null);

      // Handle both single product and multi-product generation
      const codes = Array.isArray(productCodes) ? productCodes : [productCodes];
      console.log(`📦 Processing ${codes.length} products...`);

      for (const productCode of codes) {
        console.log(`🔄 Generating forecast for ${productCode}...`);
        const forecast = await apiService.generateForecast(salesData, productCode);
        setForecasts(prev => ({ ...prev, [productCode]: forecast }));

        console.log(`📊 Generating distribution for ${productCode}...`);
        const distribution = await apiService.generateDistribution(salesData, productCode, forecast);

        setDistributions(prev => ({
          ...prev,
          [productCode]: {
            data: distribution,
            status: 'COMPLETE',
            forecast: forecast
          }
        }));
      }

      console.log(`✅ Generated distribution for ${codes.length} products`);

    } catch (error) {
      console.error('❌ Distribution generation failed:', error);
      setError(`Distribution generation failed: ${error.message}`);
    } finally {
      console.log('🏁 Setting loading state to false');
      setIsGeneratingDistribution(false);
    }
  };

  const handleProductSelect = async (product, action = 'toggle') => {
    if (action === 'clear') {
      setSelectedProducts([]);
      return;
    }

    if (!product) return;

    setSelectedProducts(prev => {
      const isCurrentlySelected = prev.some(p => p.productCode === product.productCode);
      
      if (action === 'add' && !isCurrentlySelected) {
        return [...prev, product];
      } else if (action === 'remove' && isCurrentlySelected) {
        return prev.filter(p => p.productCode !== product.productCode);
      } else if (action === 'toggle') {
        if (isCurrentlySelected) {
          return prev.filter(p => p.productCode !== product.productCode);
        } else {
          return [...prev, product];
        }
      }
      
      return prev;
    });

    // Generate forecast if not already generated
    if (!forecasts[product.productCode] && salesData.length > 0) {
      try {
        const forecast = await apiService.generateForecast(salesData, product.productCode);
        setForecasts(prev => ({ ...prev, [product.productCode]: forecast }));
      } catch (error) {
        console.error('Failed to generate forecast:', error);
      }
    }
  };

  const handleShowCategories = () => {
    setCurrentView('categories');
  };

  const handleSelectCategory = (category) => {
    setSelectedCategory(category);
    setSelectedProducts([]);
    setCurrentView('split');
  };

  const handleBackToAnalytics = () => {
    setCurrentView('analytics');
    setSelectedCategory(null);
    setSelectedProducts([]);
  };

  const handleBackToCategories = () => {
    setCurrentView('categories');
    setSelectedProducts([]);
  };

  // Render based on current view
  const renderContent = () => {
    switch (currentView) {
      case 'upload':
        return (
          <div className="flex-1 overflow-y-auto p-6">
            {error && (
              <ErrorMessage
                error={error}
                onRetry={() => {
                  setError(null);
                  setModelStatus('NOT_READY');
                }}
              />
            )}

            {isLoading && (
              <LoadingSpinner 
                message="Processing Your Sales Data" 
                subMessage="Training AI model and extracting product insights..."
              />
            )}

            {!isLoading && !isDataLoaded && !error && (
              <div className="text-center py-12">
                <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Welcome to AI Stock Distribution</h3>
                <p className="text-gray-600 mb-4">
                  Upload your historical sales CSV file to begin AI-powered inventory forecasting and distribution.
                </p>
                <div className="text-sm text-gray-500 mt-4">
                  <p className="mb-2">Expected CSV format:</p>
                  <div className="bg-gray-100 rounded p-3 text-left inline-block">
                    <code className="text-xs">
                      Shop Id, Shop, Sale Date, Product Name, Product Code,<br />
                      Size Name, Size Code, Color Name, Color Code, Category,<br />
                      Gender, Season, LineItem
                    </code>
                  </div>
                </div>
              </div>
            )}
          </div>
        );

      case 'analytics':
        return (
          <div className="flex-1 overflow-y-auto p-6">
            <AnalyticsDashboard 
              salesData={salesData} 
              onShowCategories={handleShowCategories}
            />
          </div>
        );

      case 'categories':
        return (
          <div className="flex-1 overflow-y-auto p-6">
            <CategoriesGrid 
              salesData={salesData}
              onSelectCategory={handleSelectCategory}
              onBackToAnalytics={handleBackToAnalytics}
            />
          </div>
        );

      case 'split':
        return (
          <SplitScreenView
            category={selectedCategory}
            products={products}
            salesData={salesData}
            selectedProducts={selectedProducts}
            onSelectProduct={handleProductSelect}
            onBackToCategories={handleBackToCategories}
            distributions={distributions}
            forecasts={forecasts}
            onGenerateDistribution={handleGenerateDistribution}
            isLoading={isLoading}
            isGeneratingDistribution={isGeneratingDistribution}
          />
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-gray-50">
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm z-10">
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center">
              <Brain className="w-6 h-6 mr-2 text-purple-600" />
              <h2 className="text-2xl font-bold text-gray-800">AI Stock Distribution</h2>
              <div className="ml-4 text-sm text-gray-500">
                {currentView === 'analytics' && `Season: SS25 • Analytics Dashboard`}
                {currentView === 'categories' && `Season: SS25 • Product Categories`}
                {currentView === 'split' && `Season: SS25 • ${selectedCategory}`}
                {currentView === 'upload' && `Season: SS25 • Data Upload`}
              </div>
            </div>

            <ConnectionStatus isDataLoaded={isDataLoaded} modelStatus={modelStatus} />
          </div>

          {/* Data Upload Section - Only show when no data is loaded */}
          {currentView === 'upload' && (
            <DataUploadSection onDataLoad={handleDataLoad} isLoading={isLoading} />
          )}
        </header>

        {/* Main Content */}
        {renderContent()}
      </div>
    </div>
  );
}

export default InitialStockDistribution;