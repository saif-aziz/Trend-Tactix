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
  ShoppingCart
} from 'lucide-react';

// Mock API Service (same as before)
const apiService = {
  processCSVData: async (csvData) => {
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

// Loading Component
function LoadingSpinner({ message = "Loading..." }) {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="text-center">
        <RefreshCw className="w-8 h-8 animate-spin text-purple-600 mx-auto mb-2" />
        <p className="text-gray-600">{message}</p>
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
      // Process data for top 10 categories over 3 years
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

      // Get top 10 categories
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
      {/* Header */}
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

      {/* Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="space-y-4">
          {chartData.map((item, index) => (
            <div key={item.category} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium text-gray-900">{item.category}</span>
                <span className="text-sm text-gray-500">{item.total.toLocaleString()} sales</span>
              </div>
              
              {/* Stacked Bar */}
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
              
              {/* Year breakdown */}
              <div className="flex justify-between text-xs text-gray-500">
                <span>2022: {item[2022]}</span>
                <span>2023: {item[2023]}</span>
                <span>2024: {item[2024]}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
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

      {/* Summary Stats */}
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
        acc[category].uniqueProducts.add(sale['Product Code']);
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
      {/* Header */}
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

      {/* Categories Grid */}
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

// Split Screen View Component
function SplitScreenView({ 
  category, 
  products, 
  salesData, 
  selectedProduct, 
  onSelectProduct, 
  onBackToCategories,
  distributions,
  forecasts,
  onGenerateDistribution,
  isLoading 
}) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredProducts = products.filter(product => 
    product.category === category &&
    (product.name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
     product.productCode?.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <button
              onClick={onBackToCategories}
              className="flex items-center text-gray-600 hover:text-gray-900 mb-2"
            >
              <ArrowRight className="w-4 h-4 mr-2 rotate-180" />
              Back to Categories
            </button>
            <h2 className="text-xl font-bold text-gray-900">{category}</h2>
            <p className="text-sm text-gray-600">{filteredProducts.length} products</p>
          </div>
          
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search products..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
            />
          </div>
        </div>
      </div>

      {/* Split Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Half - Products */}
        <div className="w-1/2 border-r border-gray-200 overflow-y-auto p-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {filteredProducts.map(product => (
              <ProductCard
                key={product.id}
                product={product}
                onSelect={onSelectProduct}
                isSelected={selectedProduct?.productCode === product.productCode}
                showPredictions={true}
                distributionStatus={distributions[product.productCode]?.status}
                onGenerateDistribution={onGenerateDistribution}
                forecast={forecasts[product.productCode]}
                compact={true}
              />
            ))}
          </div>
        </div>

        {/* Right Half - Distribution */}
        <div className="w-1/2 overflow-y-auto p-4">
          {selectedProduct ? (
            <DistributionPanel
              product={selectedProduct}
              distributions={distributions[selectedProduct.productCode]?.data}
              forecast={forecasts[selectedProduct.productCode]}
              onGenerateDistribution={() => onGenerateDistribution(selectedProduct.productCode)}
              isLoading={isLoading}
            />
          ) : (
            <div className="h-full flex items-center justify-center text-center">
              <div>
                <ShoppingCart className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Select a Product</h3>
                <p className="text-gray-600">Choose a product from the left to view distribution options</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Compact Product Card Component
function ProductCard({ 
  product, 
  onSelect, 
  isSelected, 
  showPredictions, 
  distributionStatus, 
  onGenerateDistribution,
  forecast,
  compact = false 
}) {
  const riskColors = {
    LOW: 'text-green-600 bg-green-100',
    MEDIUM: 'text-yellow-600 bg-yellow-100',
    HIGH: 'text-red-600 bg-red-100'
  };

  return (
    <div
      className={`bg-white rounded-lg shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer border-2 ${
        isSelected ? 'border-purple-500 ring-2 ring-purple-200' : 'border-transparent'
      } ${compact ? 'p-4' : 'p-6'}`}
      onClick={() => onSelect(product)}
    >
      <div className="space-y-3">
        <div className="flex justify-between items-start">
          <div>
            <h3 className={`${compact ? 'text-base' : 'text-lg'} font-semibold text-gray-900`}>
              {product.name}
            </h3>
            <p className="text-sm text-gray-500">{product.productCode}</p>
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

// Distribution Panel Component
function DistributionPanel({ product, distributions, forecast, onGenerateDistribution, isLoading }) {
  const [isExporting, setIsExporting] = useState(false);

  const shop = {
    id: 1,
    name: "(S-12) Packages Mall Lahore",
    location: "Lahore",
    tier: "FLAGSHIP"
  };

  const handleExport = async () => {
    try {
      setIsExporting(true);
      
      const exportData = {
        productCode: product.productCode,
        productName: product.name,
        shop: shop,
        totalAllocated: distributions?.reduce((sum, dist) => sum + dist.allocatedQuantity, 0) || 0,
        forecast: forecast,
        distributions: distributions?.map(dist => ({
          variation: `${dist.variation.size} - ${dist.variation.color}`,
          quantity: dist.allocatedQuantity,
          reasoning: dist.reasoning
        })) || [],
        exportDate: new Date().toISOString(),
        modelVersion: 'Sequential_Forecasting_v1.0'
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `distribution_${product.productCode}_${new Date().toISOString().split('T')[0]}.json`;
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

  if (!distributions || distributions.length === 0) {
    return (
      <div className="h-full flex flex-col">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-4">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            Distribution Plan: {product.name}
          </h3>
          
          {forecast && (
            <div className="mb-6 p-4 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-3">AI Forecast</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-blue-900">{forecast.predictedDemand}</div>
                  <div className="text-blue-700">Predicted Demand</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-blue-900">{forecast.confidence}%</div>
                  <div className="text-blue-700">Confidence</div>
                </div>
              </div>
            </div>
          )}

          <div className="text-center py-8">
            <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500 mb-4">No distribution generated yet</p>
            <button
              onClick={onGenerateDistribution}
              disabled={isLoading}
              className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 transition-colors"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2 inline" />
                  Generate Smart Distribution
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  const totalAllocated = distributions.reduce((sum, dist) => sum + dist.allocatedQuantity, 0);

  return (
    <div className="h-full flex flex-col">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-semibold text-gray-900">
            Distribution Plan: {product.name}
          </h3>
          <div className="text-sm text-gray-500">
            Total: {totalAllocated} units
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

          {/* Variations */}
          <div className="space-y-3">
            <h5 className="font-medium text-gray-700">Variation Breakdown</h5>
            <div className="space-y-2">
              {distributions.map((item, idx) => (
                <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium text-gray-900">
                      {item.variation.size} - {item.variation.color}
                    </div>
                    <div className="text-xs text-gray-500">
                      {item.variation.sizeCode} / {item.variation.colorCode}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-lg font-semibold text-purple-600">
                      {item.allocatedQuantity}
                    </div>
                    <div className="text-xs text-gray-500">units</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Forecast Summary */}
        {forecast && (
          <div className="mb-6 p-4 bg-blue-50 rounded-lg">
            <h5 className="font-medium text-blue-900 mb-3">AI Forecast Summary</h5>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="text-center">
                <div className="font-semibold text-blue-900">{forecast.predictedDemand}</div>
                <div className="text-blue-700">Predicted Demand</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-blue-900">{forecast.confidence}%</div>
                <div className="text-blue-700">Confidence</div>
              </div>
              <div className="text-center">
                <div className={`font-semibold ${forecast.riskLevel === 'LOW' ? 'text-green-600' : 
                  forecast.riskLevel === 'MEDIUM' ? 'text-yellow-600' : 'text-red-600'}`}>
                  {forecast.riskLevel}
                </div>
                <div className="text-blue-700">Risk Level</div>
              </div>
              <div className="text-center">
                <div className="font-semibold text-blue-900">{totalAllocated}</div>
                <div className="text-blue-700">Total Allocated</div>
              </div>
            </div>
            
            {forecast.reasoning && (
              <div className="mt-3 p-2 bg-white rounded text-xs text-gray-600">
                <Brain className="w-3 h-3 inline mr-1" />
                {forecast.reasoning}
              </div>
            )}
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-between items-center">
          <div className="text-sm text-gray-600">
            <Brain className="w-4 h-4 inline mr-1" />
            AI-optimized distribution
          </div>

          <div className="flex space-x-3">
            <button 
              onClick={onGenerateDistribution}
              className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Regenerate
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
                  Export to POS
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main Component
function InitialStockDistribution() {
  // View State Management
  const [currentView, setCurrentView] = useState('upload'); // 'upload', 'analytics', 'categories', 'split'
  const [selectedCategory, setSelectedCategory] = useState(null);
  
  // Data State Management
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [products, setProducts] = useState([]);
  const [salesData, setSalesData] = useState([]);
  const [distributions, setDistributions] = useState({});
  const [forecasts, setForecasts] = useState({});
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  const [modelStatus, setModelStatus] = useState('NOT_READY');
  const [isLoading, setIsLoading] = useState(false);
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
      setCurrentView('analytics'); // Move to analytics view after upload

      console.log(`Loaded ${processedProducts.length} products and ${processedSalesData.length} sales records`);

    } catch (error) {
      setError(`Data processing failed: ${error.message}`);
      setModelStatus('ERROR');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateDistribution = async (productCode) => {
    try {
      setIsLoading(true);
      setError(null);

      const forecast = await apiService.generateForecast(salesData, productCode);
      setForecasts(prev => ({ ...prev, [productCode]: forecast }));

      const distribution = await apiService.generateDistribution(salesData, productCode, forecast);

      setDistributions(prev => ({
        ...prev,
        [productCode]: {
          data: distribution,
          status: 'COMPLETE',
          forecast: forecast
        }
      }));

      console.log(`Generated distribution for ${productCode}:`, { forecast, distribution });

    } catch (error) {
      setError(`Distribution generation failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProductSelect = async (product) => {
    setSelectedProduct(product);

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
    setSelectedProduct(null);
    setCurrentView('split');
  };

  const handleBackToAnalytics = () => {
    setCurrentView('analytics');
    setSelectedCategory(null);
    setSelectedProduct(null);
  };

  const handleBackToCategories = () => {
    setCurrentView('categories');
    setSelectedProduct(null);
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

            {isLoading && <LoadingSpinner message="Processing your sales data..." />}

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
            selectedProduct={selectedProduct}
            onSelectProduct={handleProductSelect}
            onBackToCategories={handleBackToCategories}
            distributions={distributions}
            forecasts={forecasts}
            onGenerateDistribution={handleGenerateDistribution}
            isLoading={isLoading}
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