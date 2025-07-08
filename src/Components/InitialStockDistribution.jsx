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
  EyeOff,
  Settings,
  Activity,
  BarChart2,
  Clock,
  Layers,
  Sliders,
  TrendingDown,
  Award,
  Lock,
  Unlock,
  GitBranch,
  History,
  PlayCircle,
  PauseCircle,
  RotateCcw
} from 'lucide-react';

// Updated API Service for new training/prediction workflow
const apiService = {
  loadTrainingData: async (salesFile, inventoryFile = null) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const formData = new FormData();
    formData.append('sales_file', salesFile);
    if (inventoryFile) {
      formData.append('inventory_file', inventoryFile);
    }
    
    const response = await fetch('http://localhost:5000/api/load-training-data', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Failed to load training data');
    }
    
    return await response.json();
  },

  loadPredictionData: async (productsFile) => {
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const formData = new FormData();
    formData.append('products_file', productsFile);
    
    const response = await fetch('http://localhost:5000/api/load-prediction-data', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error('Failed to load prediction data');
    }
    
    return await response.json();
  },

  trainModel: async (trainConfig = {}) => {
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const response = await fetch('http://localhost:5000/api/train-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trainConfig)
    });
    
    if (!response.ok) {
      throw new Error('Model training failed');
    }
    
    return await response.json();
  },

  generatePredictions: async (productCodes = []) => {
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const response = await fetch('http://localhost:5000/api/generate-predictions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_codes: productCodes })
    });
    
    if (!response.ok) {
      throw new Error('Prediction generation failed');
    }
    
    return await response.json();
  },

  processCSVData: async (csvData) => {
    // Fallback for backward compatibility
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const lines = csvData.split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    
    const products = {};
    
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      
      const values = lines[i].split(',').map(v => v.trim().replace(/"/g, ''));
      const row = {};
      
      headers.forEach((header, index) => {
        row[header] = values[index];
      });
      
      if (row['Product Code']) {
        const productCode = row['Product Code'];
        if (!products[productCode]) {
          products[productCode] = {
            id: productCode,
            productCode: productCode,
            name: row['Product Name'] || productCode,
            category: row['Category'] || 'Unknown',
            gender: row['Gender'] || 'Unisex',
            season: row['Season'] || 'Unknown',
            attributes: {
              size: row['Size Name'] || 'OS',
              color: row['Color Name'] || 'Default',
              gender: row['Gender'] || 'Unisex'
            },
            totalQuantity: 0,
            historicalSales: 0,
            predictedDemand: 0
          };handleTrainingDataUpload
        }
      }
    }handleTrainingDataUpload
    
    return { products: Object.values(products) };
  },

  generateForecast: async (productCode, productData) => {
    await new Promise(resolve => setTimeout(resolve, 50));
    
    // Use the new API endpoint
    const response = await fetch('http://localhost:5000/api/generate-distribution', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_id: productCode })
    });
    
    if (!response.ok) {
      return {
        predictedDemand: 0,
        confidence: 0,
        riskLevel: 'HIGH',
        reasoning: 'Forecast generation failed'
      };
    }
    
    const result = await response.json();
    return result.forecast;
  },

  generateDistribution: async (productCode, forecast) => {
    await new Promise(resolve => setTimeout(resolve, 600));
    
    const response = await fetch('http://localhost:5000/api/generate-distribution', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_id: productCode })
    });
    
    if (!response.ok) {
      return [];
    }
    
    const result = await response.json();
    return result.distribution;
  },

  // Add this new method to apiService object (around line 60):
  validateModel: async (validationConfig = {}) => {
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    const response = await fetch('http://localhost:5000/api/validate-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(validationConfig)
    });
    
    if (!response.ok) {
      throw new Error('Model validation failed');
    }
    
    return await response.json();
  },

// Add these new methods to the existing apiService object:

optimizeHyperparameters: async (method = 'optuna', nTrials = 50) => {
  const response = await fetch('http://localhost:5000/api/optimize-hyperparameters', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method, n_trials: nTrials })
  });
  if (!response.ok) throw new Error('Hyperparameter optimization failed');
  return await response.json();
},

updateEnsembleWeights: async () => {
  const response = await fetch('http://localhost:5000/api/update-ensemble-weights', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!response.ok) throw new Error('Weight update failed');
  return await response.json();
},

optimizeCategoryModels: async () => {
  const response = await fetch('http://localhost:5000/api/optimize-category-models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!response.ok) throw new Error('Category optimization failed');
  return await response.json();
},

monitorModelDrift: async () => {
  const response = await fetch('http://localhost:5000/api/monitor-model-drift', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' }
  });
  if (!response.ok) throw new Error('Drift monitoring failed');
  return await response.json();
},

saveModelVersion: async (versionName = null) => {
  const response = await fetch('http://localhost:5000/api/save-model-version', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version_name: versionName })
  });
  if (!response.ok) throw new Error('Model saving failed');
  return await response.json();
},

loadModelVersion: async (version) => {
  const response = await fetch('http://localhost:5000/api/load-model-version', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version })
  });
  if (!response.ok) throw new Error('Model loading failed');
  return await response.json();
},

getAvailableModelVersions: async () => {
  const response = await fetch('http://localhost:5000/api/get-available-model-versions');
  if (!response.ok) throw new Error('Failed to get model versions');
  return await response.json();
},

getOptimizationReport: async () => {
  const response = await fetch('http://localhost:5000/api/get-optimization-report');
  if (!response.ok) throw new Error('Failed to get optimization report');
  return await response.json();
},

updateWithNewData: async (newSalesData, retrainThreshold = 100) => {
  const response = await fetch('http://localhost:5000/api/update-with-new-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_sales_data: newSalesData, retrain_threshold: retrainThreshold })
  });
  if (!response.ok) throw new Error('Data update failed');
  return await response.json();
},

setupABTesting: async (testRatio = 0.2) => {
  const response = await fetch('http://localhost:5000/api/setup-ab-testing', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ test_ratio: testRatio })
  });
  if (!response.ok) throw new Error('A/B testing setup failed');
  return await response.json();
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



// Advanced Optimization Panel Component
function AdvancedOptimizationPanel({ 
  modelStatus, 
  onOptimizationComplete, 
  brandConfig,
  isModelTrained 
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [activeOptimizations, setActiveOptimizations] = useState(new Set());
  const [optimizationReport, setOptimizationReport] = useState(null);
  const [availableVersions, setAvailableVersions] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState('');
  const [driftReport, setDriftReport] = useState(null);

  const runOptimization = async (type, config = {}) => {
    try {
      setActiveOptimizations(prev => new Set([...prev, type]));
      
      let result;
      switch (type) {
        case 'hyperparameters':
          result = await apiService.optimizeHyperparameters(
            config.method || 'optuna', 
            config.nTrials || 50
          );
          break;
        case 'weights':
          result = await apiService.updateEnsembleWeights();
          break;
        case 'category':
          result = await apiService.optimizeCategoryModels();
          break;
        case 'drift':
          result = await apiService.monitorModelDrift();
          setDriftReport(result.drift_report);
          break;
        default:
          throw new Error(`Unknown optimization type: ${type}`);
      }
      
      setOptimizationResults(prev => ({ ...prev, [type]: result }));
      onOptimizationComplete(type, result);
      
    } catch (error) {
      alert(`${type} optimization failed: ${error.message}`);
    } finally {
      setActiveOptimizations(prev => {
        const newSet = new Set(prev);
        newSet.delete(type);
        return newSet;
      });
    }
  };

  const loadOptimizationReport = async () => {
    try {
      const report = await apiService.getOptimizationReport();
      setOptimizationReport(report.report);
    } catch (error) {
      console.error('Failed to load optimization report:', error);
    }
  };

  const loadAvailableVersions = async () => {
    try {
      const versions = await apiService.getAvailableModelVersions();
      setAvailableVersions(versions.versions);
    } catch (error) {
      console.error('Failed to load model versions:', error);
    }
  };

  const saveCurrentModel = async () => {
    try {
      const versionName = `optimized_${new Date().toISOString().split('T')[0]}`;
      await apiService.saveModelVersion(versionName);
      await loadAvailableVersions();
      alert('Model version saved successfully!');
    } catch (error) {
      alert(`Failed to save model: ${error.message}`);
    }
  };

  const loadModelVersion = async () => {
    if (!selectedVersion) return;
    try {
      await apiService.loadModelVersion(selectedVersion);
      alert('Model version loaded successfully!');
      onOptimizationComplete('version_load', { version: selectedVersion });
    } catch (error) {
      alert(`Failed to load model version: ${error.message}`);
    }
  };

  useEffect(() => {
    if (isModelTrained) {
      loadOptimizationReport();
      loadAvailableVersions();
    }
  }, [isModelTrained]);

  if (!isModelTrained) {
    return null;
  }

  return (
    <div className="mt-6 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-200">
      <div className="p-4">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center justify-between w-full text-left"
        >
          <div className="flex items-center">
            <Settings className="w-5 h-5 mr-2 text-purple-600" />
            <h3 className="font-semibold text-purple-900">Advanced Model Optimization</h3>
            <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
              Pro Features
            </span>
          </div>
          {showAdvanced ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>

        {showAdvanced && (
          <div className="mt-4 space-y-6">
            {/* Optimization Actions Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Hyperparameter Optimization */}
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center mb-3">
                  <Sliders className="w-5 h-5 mr-2 text-blue-600" />
                  <h4 className="font-medium text-gray-900">Hyperparameters</h4>
                </div>
                <p className="text-sm text-gray-600 mb-3">
                  Optimize model parameters using Bayesian optimization
                </p>
                <button
                  onClick={() => runOptimization('hyperparameters', { method: 'optuna', nTrials: 50 })}
                  disabled={activeOptimizations.has('hyperparameters')}
                  className="w-full px-3 py-2 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
                >
                  {activeOptimizations.has('hyperparameters') ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    'Optimize'
                  )}
                </button>
              </div>

              {/* Dynamic Weight Updates */}
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center mb-3">
                  <BarChart2 className="w-5 h-5 mr-2 text-green-600" />
                  <h4 className="font-medium text-gray-900">Ensemble Weights</h4>
                </div>
                <p className="text-sm text-gray-600 mb-3">
                  Update model weights based on recent performance
                </p>
                <button
                  onClick={() => runOptimization('weights')}
                  disabled={activeOptimizations.has('weights')}
                  className="w-full px-3 py-2 bg-green-600 text-white rounded text-sm font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
                >
                  {activeOptimizations.has('weights') ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Updating...
                    </>
                  ) : (
                    'Update Weights'
                  )}
                </button>
              </div>

              {/* Category-Specific Models */}
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center mb-3">
                  <Layers className="w-5 h-5 mr-2 text-orange-600" />
                  <h4 className="font-medium text-gray-900">Category Models</h4>
                </div>
                <p className="text-sm text-gray-600 mb-3">
                  Train specialized models for each category
                </p>
                <button
                  onClick={() => runOptimization('category')}
                  disabled={activeOptimizations.has('category')}
                  className="w-full px-3 py-2 bg-orange-600 text-white rounded text-sm font-medium hover:bg-orange-700 disabled:opacity-50 transition-colors"
                >
                  {activeOptimizations.has('category') ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Training...
                    </>
                  ) : (
                    'Train Category Models'
                  )}
                </button>
              </div>

              {/* Model Drift Monitoring */}
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center mb-3">
                  <Activity className="w-5 h-5 mr-2 text-red-600" />
                  <h4 className="font-medium text-gray-900">Drift Monitor</h4>
                </div>
                <p className="text-sm text-gray-600 mb-3">
                  Check for model performance degradation
                </p>
                <button
                  onClick={() => runOptimization('drift')}
                  disabled={activeOptimizations.has('drift')}
                  className="w-full px-3 py-2 bg-red-600 text-white rounded text-sm font-medium hover:bg-red-700 disabled:opacity-50 transition-colors"
                >
                  {activeOptimizations.has('drift') ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Monitoring...
                    </>
                  ) : (
                    'Monitor Drift'
                  )}
                </button>
              </div>
            </div>

            {/* Model Versioning */}
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <GitBranch className="w-5 h-5 mr-2 text-purple-600" />
                  <h4 className="font-medium text-gray-900">Model Versioning</h4>
                </div>
                <button
                  onClick={saveCurrentModel}
                  className="px-3 py-2 bg-purple-600 text-white rounded text-sm font-medium hover:bg-purple-700 transition-colors"
                >
                  <History className="w-3 h-3 mr-1 inline" />
                  Save Current Model
                </button>
              </div>
              
              <div className="flex items-center space-x-3">
                <select
                  value={selectedVersion}
                  onChange={(e) => setSelectedVersion(e.target.value)}
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">Select a model version...</option>
                  {availableVersions.map(version => (
                    <option key={version.version} value={version.version}>
                      {version.version} {version.timestamp && `(${new Date(version.timestamp).toLocaleDateString()})`}
                    </option>
                  ))}
                </select>
                <button
                  onClick={loadModelVersion}
                  disabled={!selectedVersion}
                  className="px-4 py-2 bg-gray-600 text-white rounded text-sm font-medium hover:bg-gray-700 disabled:opacity-50 transition-colors"
                >
                  Load Version
                </button>
              </div>
            </div>

            {/* Optimization Results */}
            {optimizationResults && (
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                  <Award className="w-5 h-5 mr-2 text-yellow-600" />
                  Recent Optimization Results
                </h4>
                <div className="space-y-2 text-sm">
                  {Object.entries(optimizationResults).map(([type, result]) => (
                    <div key={type} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                      <span className="font-medium capitalize">{type.replace('_', ' ')}</span>
                      <span className="text-green-600">✓ Completed</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Drift Report */}
            {driftReport && (
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                  <TrendingDown className="w-5 h-5 mr-2 text-red-600" />
                  Model Drift Analysis
                </h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="font-medium">Current Performance</div>
                    <div>MAE: {driftReport.current_performance?.mae?.toFixed(4) || 'N/A'}</div>
                    <div>MAPE: {driftReport.current_performance?.mape?.toFixed(2) || 'N/A'}%</div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="font-medium">Recommendation</div>
                    <div className={`font-semibold ${
                      driftReport.retrain_recommended ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {driftReport.retrain_recommended ? 'Retrain Required' : 'Model Stable'}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Optimization Report */}
            {optimizationReport && (
              <div className="bg-white rounded-lg p-4 border border-gray-200">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                  Optimization Summary
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center p-3 bg-blue-50 rounded">
                    <div className="text-2xl font-bold text-blue-600">
                      {optimizationReport.learning_rate?.toFixed(3) || 'N/A'}
                    </div>
                    <div className="text-gray-600">Learning Rate</div>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded">
                    <div className="text-2xl font-bold text-green-600">
                      {optimizationReport.category_models || 0}
                    </div>
                    <div className="text-gray-600">Category Models</div>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded">
                    <div className="text-2xl font-bold text-purple-600">
                      {optimizationReport.model_versions_saved || 0}
                    </div>
                    <div className="text-gray-600">Saved Versions</div>
                  </div>
                  <div className="text-center p-3 bg-orange-50 rounded">
                    <div className="text-2xl font-bold text-orange-600">
                      {optimizationReport.performance_history?.length || 0}
                    </div>
                    <div className="text-gray-600">Performance History</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
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
          modelStatus === 'PROCESSING' || modelStatus === 'TRAINING' ? 'bg-yellow-500 animate-pulse' : 'bg-red-500'
        }`}></div>
        <span className="text-sm text-gray-600">
          Model: {modelStatus === 'READY' ? 'Ready' : modelStatus || 'Not Ready'}
        </span>
      </div>
    </div>
  );
}

// Multi-Stage Data Upload Component
function DataUploadSection({ onDataLoad, isLoading, currentStage, onStageComplete }) {
  const [dragOver, setDragOver] = useState(false);
  const [trainingDataLoaded, setTrainingDataLoaded] = useState(false);
  const [predictionDataLoaded, setPredictionDataLoaded] = useState(false);
  const [modelValidated, setModelValidated] = useState(false);
  const [modelTrained, setModelTrained] = useState(false);
  const [validationResults, setValidationResults] = useState(null);
  
  // Individual loading states for each step
  const [isLoadingTraining, setIsLoadingTraining] = useState(false);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  const handleTrainingDataUpload = async (salesFile, inventoryFile = null) => {
    try {
      setIsLoadingTraining(true);
      console.log('🚀 Starting training data upload...', salesFile.name);
      
      const result = await apiService.loadTrainingData(salesFile, inventoryFile);
      
      console.log('✅ Training data upload successful:', result);
      setTrainingDataLoaded(true);
      onStageComplete('training', result);
    } catch (error) {
      console.error('❌ Training data upload failed:', error);
      alert(`Training data upload failed: ${error.message}`);
    } finally {
      setIsLoadingTraining(false);
    }
  };

  const handlePredictionDataUpload = async (file) => {
    try {
      setIsLoadingPrediction(true);
      console.log('🚀 Starting prediction data upload...', file.name);
      
      const result = await apiService.loadPredictionData(file);
      
      console.log('✅ Prediction data upload successful:', result);
      setPredictionDataLoaded(true);
      onStageComplete('prediction', result);
    } catch (error) {
      console.error('❌ Prediction data upload failed:', error);
      alert(`Prediction data upload failed: ${error.message}`);
    } finally {
      setIsLoadingPrediction(false);
    }
  };

  const handleModelValidation = async () => {
    try {
      setIsValidating(true);
      console.log('🚀 Starting model validation...');
      
      const result = await apiService.validateModel();
      
      console.log('✅ Model validation successful:', result);
      setModelValidated(true);
      setValidationResults(result);
      onStageComplete('validation', result);
    } catch (error) {
      console.error('❌ Model validation failed:', error);
      alert(`Model validation failed: ${error.message}`);
    } finally {
      setIsValidating(false);
    }
  };

  const handleModelTraining = async () => {
    try {
      setIsTraining(true);
      console.log('🚀 Starting model training...');
      
      const result = await apiService.trainModel();
      
      console.log('✅ Model training successful:', result);
      setModelTrained(true);  // ← Make sure this is here
      onStageComplete('training_complete', result);
    } catch (error) {
      console.error('❌ Model training failed:', error);
      alert(`Model training failed: ${error.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleFileUpload = async (files, type) => {
    if (type === 'training') {
      const salesFile = files[0];
      const inventoryFile = files[1] || null;
      if (salesFile && salesFile.type === 'text/csv') {
        await handleTrainingDataUpload(salesFile, inventoryFile);
      } else {
        alert('Please upload CSV files');
      }
    } else if (type === 'prediction') {
      const file = files[0];
      if (file && file.type === 'text/csv') {
        await handlePredictionDataUpload(file);
      } else {
        alert('Please upload a CSV file');
      }
    }
  };

  return (
    <>
      <div className="mb-6 space-y-4">
        {/* Stage 1: Training Data */}
        <div className={`p-4 rounded-lg border transition-all duration-300 ${
          trainingDataLoaded 
            ? 'bg-green-50 border-green-200 shadow-sm' 
            : 'bg-blue-50 border-blue-200 hover:border-blue-300'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Database className={`w-5 h-5 mr-3 ${
                trainingDataLoaded ? 'text-green-600' : 'text-blue-600'
              }`} />
              <div>
                <h3 className={`font-medium ${
                  trainingDataLoaded ? 'text-green-900' : 'text-blue-900'
                }`}>
                  Step 1: Training Data {trainingDataLoaded && '✓'}
                </h3>
                <p className={`text-sm ${
                  trainingDataLoaded ? 'text-green-700' : 'text-blue-700'
                }`}>
                  Upload historical sales data (required) and inventory data (optional)
                </p>
              </div>
            </div>

            {!trainingDataLoaded && (
              <div className="flex items-center space-x-3">
                <input
                  type="file"
                  accept=".csv"
                  multiple
                  onChange={(e) => handleFileUpload(Array.from(e.target.files), 'training')}
                  className="hidden"
                  id="training-upload"
                  disabled={isLoadingTraining}
                />
                <label 
                  htmlFor="training-upload" 
                  className={`cursor-pointer px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 font-medium ${
                    isLoadingTraining ? 'opacity-50 cursor-not-allowed transform scale-95' : 'hover:transform hover:scale-105'
                  }`}
                >
                  {isLoadingTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2 inline" />
                      Upload Training Data
                    </>
                  )}
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Stage 2: Prediction Data */}
        <div className={`p-4 rounded-lg border transition-all duration-300 ${
          predictionDataLoaded 
            ? 'bg-green-50 border-green-200 shadow-sm' 
            : trainingDataLoaded 
              ? 'bg-blue-50 border-blue-200 hover:border-blue-300' 
              : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Package className={`w-5 h-5 mr-3 ${
                predictionDataLoaded 
                  ? 'text-green-600' 
                  : trainingDataLoaded 
                    ? 'text-blue-600' 
                    : 'text-gray-400'
              }`} />
              <div>
                <h3 className={`font-medium ${
                  predictionDataLoaded 
                    ? 'text-green-900' 
                    : trainingDataLoaded 
                      ? 'text-blue-900' 
                      : 'text-gray-500'
                }`}>
                  Step 2: New Products Data {predictionDataLoaded && '✓'}
                </h3>
                <p className={`text-sm ${
                  predictionDataLoaded 
                    ? 'text-green-700' 
                    : trainingDataLoaded 
                      ? 'text-blue-700' 
                      : 'text-gray-500'
                }`}>
                  Upload 2025 products for demand prediction
                </p>
              </div>
            </div>

            {trainingDataLoaded && !predictionDataLoaded && (
              <div className="flex items-center space-x-3">
                <input
                  type="file"
                  accept=".csv"
                  onChange={(e) => handleFileUpload(Array.from(e.target.files), 'prediction')}
                  className="hidden"
                  id="prediction-upload"
                  disabled={isLoadingPrediction}
                />
                <label 
                  htmlFor="prediction-upload" 
                  className={`cursor-pointer px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all duration-200 font-medium ${
                    isLoadingPrediction ? 'opacity-50 cursor-not-allowed transform scale-95' : 'hover:transform hover:scale-105'
                  }`}
                >
                  {isLoadingPrediction ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2 inline" />
                      Upload Products
                    </>
                  )}
                </label>
              </div>
            )}
          </div>
        </div>

        {/* Stage 3: Model Validation (Optional) */}
        <div className={`p-4 rounded-lg border transition-all duration-300 ${
          modelValidated 
            ? 'bg-green-50 border-green-200 shadow-sm' 
            : predictionDataLoaded 
              ? 'bg-blue-50 border-blue-200 hover:border-blue-300' 
              : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <BarChart3 className={`w-5 h-5 mr-3 ${
                modelValidated 
                  ? 'text-green-600' 
                  : predictionDataLoaded 
                    ? 'text-blue-600' 
                    : 'text-gray-400'
              }`} />
              <div>
                <h3 className={`font-medium ${
                  modelValidated 
                    ? 'text-green-900' 
                    : predictionDataLoaded 
                      ? 'text-blue-900' 
                      : 'text-gray-500'
                }`}>
                  Step 3: Validate Model Accuracy {modelValidated && '✓'} 
                  <span className="text-xs font-normal text-orange-500 ml-2 bg-orange-100 px-2 py-1 rounded-full">
                    Optional
                  </span>
                </h3>
                <p className={`text-sm ${
                  modelValidated 
                    ? 'text-green-700' 
                    : predictionDataLoaded 
                      ? 'text-blue-700' 
                      : 'text-gray-500'
                }`}>
                  Test model accuracy on historical data (recommended for confidence)
                </p>
                {validationResults && (
                  <div className="mt-2 p-2 bg-green-100 rounded-lg">
                    <p className="text-xs text-green-700 font-medium">
                      📊 Accuracy: {validationResults.summary?.average_mape?.toFixed(1)}% MAPE • 
                      Quality: {validationResults.summary?.validation_quality}
                    </p>
                  </div>
                )}
                {!modelValidated && predictionDataLoaded && (
                  <p className="text-xs text-blue-600 mt-1 flex items-center">
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Validation provides accuracy insights but can be skipped to train directly
                  </p>
                )}
              </div>
            </div>

            {predictionDataLoaded && !modelValidated && (
              <button
                onClick={handleModelValidation}
                disabled={isValidating}
                className={`px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-all duration-200 font-medium ${
                  isValidating ? 'opacity-50 cursor-not-allowed transform scale-95' : 'hover:transform hover:scale-105'
                }`}
              >
                {isValidating ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                    Validating...
                  </>
                ) : (
                  <>
                    <BarChart3 className="w-4 h-4 mr-2 inline" />
                    Validate Model
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Stage 4: Model Training (Can bypass validation) */}
        <div className={`p-4 rounded-lg border transition-all duration-300 ${
          modelTrained 
            ? 'bg-green-50 border-green-200 shadow-sm' 
            : predictionDataLoaded 
              ? 'bg-blue-50 border-blue-200 hover:border-blue-300' 
              : 'bg-gray-50 border-gray-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Brain className={`w-5 h-5 mr-3 ${
                modelTrained 
                  ? 'text-green-600' 
                  : predictionDataLoaded 
                    ? 'text-blue-600' 
                    : 'text-gray-400'
              }`} />
              <div>
                <h3 className={`font-medium ${
                  modelTrained 
                    ? 'text-green-900' 
                    : predictionDataLoaded 
                      ? 'text-blue-900' 
                      : 'text-gray-500'
                }`}>
                  Step 4: Train AI Model {modelTrained && '✓'}
                </h3>
                <p className={`text-sm ${
                  modelTrained 
                    ? 'text-green-700' 
                    : predictionDataLoaded 
                      ? 'text-blue-700' 
                      : 'text-gray-500'
                }`}>
                  Train the ensemble forecasting model (Random Forest + XGBoost + LightGBM)
                </p>
                {!modelValidated && predictionDataLoaded && !modelTrained && (
                  <p className="text-xs text-yellow-600 mt-1 flex items-center">
                    <Zap className="w-3 h-3 mr-1" />
                    You can train directly or validate first for accuracy insights
                  </p>
                )}
              </div>
            </div>

            {predictionDataLoaded && !modelTrained && (
              <div className="flex items-center space-x-3">
                {/* Show validation status badge if validation was completed */}
                {modelValidated && validationResults && (
                  <div className="text-xs text-green-600 bg-green-100 px-3 py-2 rounded-lg border border-green-200">
                    ✅ Validated: {validationResults.summary?.average_mape?.toFixed(1)}% MAPE
                  </div>
                )}
                
                <button
                  onClick={handleModelTraining}
                  disabled={isTraining}
                  className={`px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-all duration-200 font-medium ${
                    isTraining ? 'opacity-50 cursor-not-allowed transform scale-95' : 'hover:transform hover:scale-105'
                  }`}
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-4 h-4 mr-2 inline animate-spin" />
                      Training...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2 inline" />
                      {modelValidated ? 'Train Model' : 'Train Model (Skip Validation)'}
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions Panel for Power Users */}
        {predictionDataLoaded && !modelTrained && (
          <div className="mt-4 p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-medium text-gray-900 flex items-center">
                  <Zap className="w-4 h-4 mr-2 text-yellow-500" />
                  Quick Actions
                </h4>
                <p className="text-sm text-gray-600">For experienced users who want to skip steps</p>
              </div>
              <div className="flex items-center space-x-3">
                {!modelValidated && (
                  <button
                    onClick={handleModelValidation}
                    disabled={isValidating}
                    className="px-3 py-2 bg-orange-100 text-orange-700 rounded-md text-sm font-medium hover:bg-orange-200 transition-colors disabled:opacity-50"
                  >
                    {isValidating ? (
                      <>
                        <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                        Validating...
                      </>
                    ) : (
                      'Quick Validate'
                    )}
                  </button>
                )}
                <button
                  onClick={handleModelTraining}
                  disabled={isTraining}
                  className="px-3 py-2 bg-purple-100 text-purple-700 rounded-md text-sm font-medium hover:bg-purple-200 transition-colors disabled:opacity-50"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-3 h-3 mr-1 inline animate-spin" />
                      Training...
                    </>
                  ) : (
                    'Train Directly'
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>


      {/* Add after the existing Stage 4 in DataUploadSection: */}

      {/* Stage 5: Advanced Optimization (Optional) */}
      {modelTrained && (
        <AdvancedOptimizationPanel
          modelStatus={modelStatus}
          onOptimizationComplete={(type, result) => {
            console.log(`Optimization ${type} completed:`, result);
            setOptimizationHistory(prev => [...prev, { type, result, timestamp: new Date() }]);
          }}
          brandConfig={brandConfig}
          isModelTrained={modelTrained}
        />
      )}

      {/* Full Screen Loading Overlays with Enhanced Design */}
      {isLoadingTraining && (
        <div className="fixed inset-0 bg-black bg-opacity-20 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-8 text-center border border-gray-200 max-w-md mx-4 transform animate-pulse">
            <div className="relative mb-6">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Database className="w-10 h-10 text-blue-600" />
              </div>
              <RefreshCw className="w-6 h-6 animate-spin text-blue-600 absolute top-2 right-2" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Loading Training Data
            </h3>
            <p className="text-sm text-gray-600 mb-2">
              Processing historical sales and inventory data...
            </p>
            <p className="text-xs text-gray-500 mb-6">
              🔍 Analyzing data structure, detecting features, and preparing for AI training
            </p>
            <div className="flex justify-center mb-4">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
            <div className="text-xs text-blue-600 font-medium bg-blue-50 px-3 py-2 rounded-full">
              ⏱️ This may take 30-60 seconds depending on data size...
            </div>
          </div>
        </div>
      )}

      {isLoadingPrediction && (
        <div className="fixed inset-0 bg-black bg-opacity-20 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-8 text-center border border-gray-200 max-w-md mx-4 transform animate-pulse">
            <div className="relative mb-6">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Package className="w-10 h-10 text-blue-600" />
              </div>
              <RefreshCw className="w-6 h-6 animate-spin text-blue-600 absolute top-2 right-2" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Loading Prediction Data
            </h3>
            <p className="text-sm text-gray-600 mb-2">
              Processing 2025 new products for forecasting...
            </p>
            <p className="text-xs text-gray-500 mb-6">
              ✅ Validating product attributes and checking feature compatibility
            </p>
            <div className="flex justify-center mb-4">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
            <div className="text-xs text-blue-600 font-medium bg-blue-50 px-3 py-2 rounded-full">
              🚀 Almost ready for AI model training...
            </div>
          </div>
        </div>
      )}

      {isValidating && (
        <div className="fixed inset-0 bg-black bg-opacity-20 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-8 text-center border border-gray-200 max-w-md mx-4 transform animate-pulse">
            <div className="relative mb-6">
              <div className="w-20 h-20 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-10 h-10 text-orange-600" />
              </div>
              <RefreshCw className="w-6 h-6 animate-spin text-orange-600 absolute top-2 right-2" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Validating Model Accuracy
            </h3>
            <p className="text-sm text-gray-600 mb-2">
              Testing AI model on historical data splits...
            </p>
            <p className="text-xs text-gray-500 mb-6">
              📊 Running time-series cross-validation to measure prediction accuracy
            </p>
            <div className="flex justify-center mb-4">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-orange-600 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-orange-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-orange-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
            <div className="text-xs text-orange-600 font-medium bg-orange-50 px-3 py-2 rounded-full">
              ⏳ This may take 1-3 minutes for comprehensive validation...
            </div>
          </div>
        </div>
      )}

      {isTraining && (
        <div className="fixed inset-0 bg-black bg-opacity-20 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-8 text-center border border-gray-200 max-w-md mx-4 transform animate-pulse">
            <div className="relative mb-6">
              <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="w-10 h-10 text-purple-600" />
              </div>
              <RefreshCw className="w-6 h-6 animate-spin text-purple-600 absolute top-2 right-2" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">
              Training AI Ensemble Model
            </h3>
            <p className="text-sm text-gray-600 mb-2">
              Training Random Forest, XGBoost, and LightGBM models...
            </p>
            <p className="text-xs text-gray-500 mb-6">
              🤖 Creating ensemble weights and optimizing for demand forecasting accuracy
            </p>
            <div className="flex justify-center mb-4">
              <div className="flex space-x-1">
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce"></div>
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-3 h-3 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
            <div className="text-xs text-purple-600 font-medium bg-purple-50 px-3 py-2 rounded-full">
              🎯 Training ensemble models - this may take 2-5 minutes...
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function AnalyticsDashboard({ salesData, onShowCategories, brandConfig, modelStatus }) {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (salesData.length > 0) {
      // For new products, show category distribution instead of sales history
      const categoryData = {};
      
      salesData.forEach(product => {
        const category = product.category || 'Unknown';
        if (!categoryData[category]) {
          categoryData[category] = { category, count: 0, predicted_demand: 0 };
        }
        categoryData[category].count += 1;
        categoryData[category].predicted_demand += product.predictedDemand || 0;
      });

      const topCategories = Object.values(categoryData)
        .sort((a, b) => b.count - a.count)
        .slice(0, 10);

      setChartData(topCategories);
    }
  }, [salesData]);

  const maxCount = Math.max(...chartData.map(item => item.count), 1);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">New Products Dashboard</h2>
          <p className="text-gray-600">2025 Product Categories for Demand Forecasting</p>
          {brandConfig.available_features && (
            <p className="text-sm text-purple-600 mt-1">
              Brand Features: {brandConfig.available_features.join(', ')}
            </p>
          )}
        </div>
        <button
          onClick={onShowCategories}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center"
        >
          <Grid3X3 className="w-4 h-4 mr-2" />
          View Categories
        </button>
      </div>

      {/* Model Validation Results Card */}
      {brandConfig.validation_results && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <BarChart3 className="w-5 h-5 mr-2 text-green-600" />
            Model Validation Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {brandConfig.validation_results.summary?.average_mape?.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Average MAPE</div>
              <div className="text-xs text-gray-500 mt-1">Lower is better</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {brandConfig.validation_results.summary?.accuracy_within_20_percent?.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">Within 20%</div>
              <div className="text-xs text-gray-500 mt-1">Predictions close to actual</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {brandConfig.validation_results.summary?.validation_quality || 'GOOD'}
              </div>
              <div className="text-sm text-gray-600">Quality Rating</div>
              <div className="text-xs text-gray-500 mt-1">Overall assessment</div>
            </div>
          </div>
          
          {/* Validation Details */}
          {brandConfig.validation_results.validation_results && (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Validation Details</h4>
              <div className="text-xs text-gray-600 space-y-1">
                <div>Validation Splits: {brandConfig.validation_results.n_splits}</div>
                <div>Products Tested: {brandConfig.validation_results.validation_results.reduce((sum, r) => sum + r.n_products, 0)}</div>
                <div>
                  Confidence Level: 
                  <span className={`ml-1 px-2 py-1 rounded text-xs font-medium ${
                    brandConfig.validation_results.summary?.validation_quality === 'EXCELLENT' ? 'bg-green-100 text-green-800' :
                    brandConfig.validation_results.summary?.validation_quality === 'GOOD' ? 'bg-blue-100 text-blue-800' :
                    brandConfig.validation_results.summary?.validation_quality === 'FAIR' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {brandConfig.validation_results.summary?.validation_quality}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Advanced Model Metrics Card */}
      {brandConfig.optimization_report && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2 text-purple-600" />
            Advanced Model Optimization
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {Object.keys(brandConfig.optimization_report.current_weights || {}).length}
              </div>
              <div className="text-sm text-gray-600">Ensemble Models</div>
              <div className="text-xs text-gray-500 mt-1">Dynamic weights</div>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {brandConfig.optimization_report.learning_rate?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-sm text-gray-600">Learning Rate</div>
              <div className="text-xs text-gray-500 mt-1">Adaptive optimization</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {brandConfig.optimization_report.category_models || 0}
              </div>
              <div className="text-sm text-gray-600">Category Models</div>
              <div className="text-xs text-gray-500 mt-1">Specialized predictions</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {brandConfig.optimization_report.model_versions_saved || 0}
              </div>
              <div className="text-sm text-gray-600">Model Versions</div>
              <div className="text-xs text-gray-500 mt-1">Version control</div>
            </div>
          </div>
        </div>
      )}

      {/* Advanced Optimization Controls */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Settings className="w-5 h-5 mr-2 text-purple-600" />
          Advanced Optimization Controls
          <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-700 text-xs rounded-full">
            Pro Features
          </span>
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Hyperparameter Optimization */}
          <button
            onClick={async () => {
              try {
                const result = await apiService.optimizeHyperparameters('optuna', 20);
                alert(`✅ Hyperparameter optimization completed!\n${result.message}`);
              } catch (error) {
                alert(`❌ Error: ${error.message}`);
              }
            }}
            className="p-4 bg-blue-50 hover:bg-blue-100 rounded-lg border border-blue-200 transition-colors"
          >
            <Sliders className="w-6 h-6 text-blue-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-blue-900">Optimize Parameters</div>
            <div className="text-xs text-blue-700 mt-1">Bayesian optimization</div>
          </button>

          {/* Dynamic Weight Updates */}
          <button
            onClick={async () => {
              try {
                const result = await apiService.updateEnsembleWeights();
                alert(`✅ Ensemble weights updated!\n${result.message}`);
              } catch (error) {
                alert(`❌ Error: ${error.message}`);
              }
            }}
            className="p-4 bg-green-50 hover:bg-green-100 rounded-lg border border-green-200 transition-colors"
          >
            <BarChart2 className="w-6 h-6 text-green-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-green-900">Update Weights</div>
            <div className="text-xs text-green-700 mt-1">Dynamic ensemble</div>
          </button>

          {/* Category Models */}
          <button
            onClick={async () => {
              try {
                const result = await apiService.optimizeCategoryModels();
                alert(`✅ Category models optimized!\n${result.message}`);
              } catch (error) {
                alert(`❌ Error: ${error.message}`);
              }
            }}
            className="p-4 bg-orange-50 hover:bg-orange-100 rounded-lg border border-orange-200 transition-colors"
          >
            <Layers className="w-6 h-6 text-orange-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-orange-900">Category Models</div>
            <div className="text-xs text-orange-700 mt-1">Specialized training</div>
          </button>

          {/* Model Drift Monitor */}
          <button
            onClick={async () => {
              try {
                const result = await apiService.monitorModelDrift();
                const status = result.drift_report?.retrain_recommended ? 'Retrain Recommended' : 'Model Stable';
                alert(`✅ Drift monitoring completed!\nStatus: ${status}`);
              } catch (error) {
                alert(`❌ Error: ${error.message}`);
              }
            }}
            className="p-4 bg-red-50 hover:bg-red-100 rounded-lg border border-red-200 transition-colors"
          >
            <Activity className="w-6 h-6 text-red-600 mx-auto mb-2" />
            <div className="text-sm font-medium text-red-900">Monitor Drift</div>
            <div className="text-xs text-red-700 mt-1">Performance check</div>
          </button>
        </div>

        {/* Model Versioning */}
        <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
          <h4 className="font-medium text-gray-900 mb-3 flex items-center">
            <GitBranch className="w-4 h-4 mr-2 text-purple-600" />
            Model Versioning
          </h4>
          <div className="flex items-center space-x-3">
            <button
              onClick={async () => {
                try {
                  const versionName = `optimized_${new Date().toISOString().split('T')[0]}`;
                  const result = await apiService.saveModelVersion(versionName);
                  alert(`✅ Model saved as: ${versionName}`);
                } catch (error) {
                  alert(`❌ Error: ${error.message}`);
                }
              }}
              className="px-4 py-2 bg-purple-600 text-white rounded-md text-sm font-medium hover:bg-purple-700 transition-colors"
            >
              <History className="w-4 h-4 mr-1 inline" />
              Save Current Model
            </button>
            
            <button
              onClick={async () => {
                try {
                  const result = await apiService.getOptimizationReport();
                  console.log('Optimization Report:', result);
                  alert(`✅ Optimization report loaded! Check console for details.`);
                } catch (error) {
                  alert(`❌ Error: ${error.message}`);
                }
              }}
              className="px-4 py-2 bg-gray-600 text-white rounded-md text-sm font-medium hover:bg-gray-700 transition-colors"
            >
              <BarChart3 className="w-4 h-4 mr-1 inline" />
              Get Report
            </button>
          </div>
        </div>
      </div>

      {/* Product Categories Distribution */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Product Categories Distribution</h3>
        <div className="space-y-4">
          {chartData.map((item, index) => (
            <div key={item.category} className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="font-medium text-gray-900">{item.category}</span>
                <span className="text-sm text-gray-500">{item.count} products</span>
              </div>
              
              <div className="flex h-8 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="bg-purple-500 flex items-center justify-center text-xs text-white font-medium"
                  style={{ width: `${(item.count / maxCount) * 100}%` }}
                  title={`${item.count} products in ${item.category}`}
                >
                  {item.count > 5 ? item.count : ''}
                </div>
              </div>
              
              <div className="flex justify-between text-xs text-gray-500">
                <span>Products: {item.count}</span>
                <span>Est. Demand: {item.predicted_demand}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-blue-600">{salesData.length.toLocaleString()}</div>
          <div className="text-sm text-gray-600">New Products</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-green-600">{chartData.length}</div>
          <div className="text-sm text-gray-600">Categories</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className="text-3xl font-bold text-purple-600">
            {chartData.length > 0 ? Math.round(chartData.reduce((sum, item) => sum + item.count, 0) / chartData.length) : 0}
          </div>
          <div className="text-sm text-gray-600">Avg Products/Category</div>
        </div>
        <div className="bg-white rounded-lg shadow p-6 text-center">
          <div className={`text-3xl font-bold ${
            modelStatus === 'READY' ? 'text-green-600' : 
            modelStatus === 'VALIDATED' ? 'text-blue-600' : 
            'text-orange-600'
          }`}>
            {modelStatus === 'READY' ? 'Ready' : 
             modelStatus === 'VALIDATED' ? 'Validated' : 
             modelStatus === 'TRAINING_DATA_LOADED' ? 'Data Loaded' :
             modelStatus === 'PREDICTION_DATA_LOADED' ? 'Data Ready' :
             'Training'}
          </div>
          <div className="text-sm text-gray-600">AI Model Status</div>
          {brandConfig.validation_results && (
            <div className="text-xs text-gray-500 mt-1">
              {brandConfig.validation_results.summary?.average_mape?.toFixed(1)}% MAPE
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Categories Grid Component (modified for new products)
function CategoriesGrid({ salesData, onSelectCategory, onBackToAnalytics, brandConfig }) {
  const [categories, setCategories] = useState([]);

  useEffect(() => {
    if (salesData.length > 0) {
      const categoryStats = salesData.reduce((acc, product) => {
        const category = product.category || 'Unknown';
        if (!acc[category]) {
          acc[category] = {
            name: category,
            totalProducts: 0,
            uniqueFeatures: new Set(),
            avgDemand: 0,
            totalDemand: 0
          };
        }
        acc[category].totalProducts += 1;
        acc[category].totalDemand += product.predictedDemand || 0;
        
        // Add unique features
        if (product.attributes) {
          Object.values(product.attributes).forEach(attr => {
            if (attr) acc[category].uniqueFeatures.add(attr);
          });
        }
        
        return acc;
      }, {});

      const categoriesArray = Object.values(categoryStats).map(cat => ({
        ...cat,
        uniqueFeatures: cat.uniqueFeatures.size,
        avgDemand: cat.totalProducts > 0 ? Math.round(cat.totalDemand / cat.totalProducts) : 0
      })).sort((a, b) => b.totalProducts - a.totalProducts);

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
            Back to Dashboard
          </button>
          <h2 className="text-2xl font-bold text-gray-900">New Product Categories</h2>
          <p className="text-gray-600">{categories.length} categories for 2025 forecasting</p>
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
                  <span>Products:</span>
                  <span className="font-medium">{category.totalProducts}</span>
                </div>
                <div className="flex justify-between">
                  <span>Avg Demand:</span>
                  <span className="font-medium">{category.avgDemand}</span>
                </div>
                <div className="flex justify-between">
                  <span>Variations:</span>
                  <span className="font-medium">{category.uniqueFeatures}</span>
                </div>
              </div>
              
              <div className="mt-4 px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-xs font-medium">
                Click to Forecast
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Split Screen View Component - keeping the original structure but with new data flow
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
              {selectedProducts.length > 0 && ` • ${selectedProducts.length} selected for forecasting`}
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
                      Generate AI Forecast
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
                <h3 className="text-lg font-medium text-gray-900 mb-2">Select New Products</h3>
                <p className="text-gray-600">
                  Choose one or more 2025 products to generate AI demand forecasts
                </p>
                <div className="mt-4 text-sm text-gray-500">
                  <p>💡 Tip: Select multiple products for batch forecasting</p>
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
              Generating AI Demand Forecast
            </h3>
            
            <p className="text-sm text-gray-600 mb-2">
              Processing {selectedProducts.length} new product{selectedProducts.length > 1 ? 's' : ''}
            </p>
            
            <p className="text-xs text-gray-500 mb-6">
              AI is analyzing product attributes, category patterns, and mapping to historical data for optimal demand prediction
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

// Product Level Card Component - keeping original structure
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
            <span className="text-gray-600">New Product:</span>
            <span className="font-medium">Yes</span>
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
              {selectedSKUs.length} of {product.skus.length} SKUs selected for forecasting
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
                    <p className="text-sm font-medium">New SKU</p>
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

// Enhanced Product Card Component with Multi-Select - keeping original structure
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
          <span className="text-sm text-gray-600">Status:</span>
          <span className="text-lg font-semibold text-blue-600">New Product</span>
        </div>

        {showPredictions && forecast && (
          <div className="border-t pt-3 space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">AI Forecast:</span>
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
              <span className="text-sm font-medium text-green-800">Forecast Ready</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Multi Distribution Panel Component - updated for new products
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
        exportType: 'NEW_PRODUCT_DEMAND_FORECAST',
        shop: shop,
        selectedProducts: selectedProducts.length,
        totalAllocated: totalAllocated,
        products: selectedProducts.map(product => ({
          productCode: product.productCode,
          productName: product.name,
          forecast: forecasts[product.productCode],
          distributions: distributions[product.productCode]?.data || []
        })),
        modelVersion: 'AI_Demand_Forecasting_v2.0'
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `new_product_forecast_${new Date().toISOString().split('T')[0]}.json`;
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
            New Product Demand Forecast
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
            <p className="text-gray-500 mb-4">No demand forecasts generated yet</p>
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
                  Generate AI Forecast for All Products
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
            New Product Demand Plan
          </h3>
          <div className="text-sm text-gray-500">
            Total: {totalAllocated} units across {selectedProducts.length} new SKU's
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between items-center mb-6 p-4 bg-gray-50 rounded-lg border">
          <div className="text-sm text-gray-600">
            <Brain className="w-4 h-4 inline mr-1" />
            AI-powered demand forecasting for new products
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
                  Export Forecast to POS
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

          {/* Products Summary */}
          <div className="space-y-4">
            <h5 className="font-medium text-gray-700">Demand Forecast Summary</h5>
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
                        <div className="text-xs text-gray-500">units forecasted</div>
                      </div>
                    </div>
                    
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

// Add this new function to handle incremental learning:
const handleIncrementalLearning = async (newSalesData) => {
  try {
    setIsLoading(true);
    const result = await apiService.updateWithNewData(newSalesData, 100);
    
    // Update model status if retrain was triggered
    if (result.incremental_data_size >= 100) {
      setModelStatus('UPDATING');
      // Trigger UI updates
      setBrandConfig(prev => ({ 
        ...prev, 
        last_update: new Date().toISOString(),
        incremental_samples: result.new_records_processed
      }));
    }
    
    return result;
  } catch (error) {
    setError(`Incremental learning failed: ${error.message}`);
    throw error;
  } finally {
    setIsLoading(false);
  }
};

// Main Component
function InitialStockDistribution() {
  // View State Management
  const [currentView, setCurrentView] = useState('upload');
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [currentStage, setCurrentStage] = useState('upload'); // upload, training, prediction, ready
  
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
  const [brandConfig, setBrandConfig] = useState({});

  // Add these state variables to the main component:
  const [optimizationStatus, setOptimizationStatus] = useState({
    hyperparameters: 'not_run',
    weights: 'not_run',
    category_models: 'not_run',
    drift_monitoring: 'not_run'
  });
  const [modelVersions, setModelVersions] = useState([]);
  const [currentModelVersion, setCurrentModelVersion] = useState(null);
  const [optimizationHistory, setOptimizationHistory] = useState([]);
  
  
  const handleStageComplete = async (stage, data) => {
    try {
      setError(null);
      setIsLoading(true);
      
      if (stage === 'training') {
        setBrandConfig(data.brand_config);
        setModelStatus('TRAINING_DATA_LOADED');
        console.log('Training data loaded:', data);
      } else if (stage === 'prediction') {
        const productsData = data.sample_products || [];
        setProducts(productsData);
        setIsDataLoaded(true);
        setModelStatus('PREDICTION_DATA_LOADED');
        console.log('Prediction data loaded:', data);
      } else if (stage === 'validation') {
        setModelStatus('VALIDATED');
        setBrandConfig(prev => ({ ...prev, validation_results: data }));
        console.log('Model validated:', data);
      } else if (stage === 'training_complete') {
        setModelStatus('READY');
        setCurrentView('analytics');
        setCurrentStage('ready');
        
        // Load optimization report
        try {
          const optimizationReport = await apiService.getOptimizationReport();
          setBrandConfig(prev => ({ ...prev, optimization_report: optimizationReport.report }));
        } catch (error) {
          console.warn('Could not load optimization report:', error);
        }
        
        console.log('Model trained successfully:', data);
      }
    } catch (error) {
      setError(`Stage completion failed: ${error.message}`);
      setModelStatus('ERROR');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDataLoad = async (csvData) => {
    try {
      setIsLoading(true);
      setError(null);
      setModelStatus('PROCESSING');

      const { products: processedProducts } = await apiService.processCSVData(csvData);
      
      const productsWithQuantities = processedProducts.map(product => ({
        ...product,
        totalQuantity: Math.max(50, product.historicalSales * 2)
      }));

      setProducts(productsWithQuantities);
      setSalesData([]);
      setIsDataLoaded(true);
      setModelStatus('READY');
      setCurrentView('analytics');

      console.log(`Loaded ${processedProducts.length} products`);

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
      setIsGeneratingDistribution(true);
      console.log('✅ Loading state set to true');
      setError(null);

      // Use new API for batch predictions
      const result = await apiService.generatePredictions(productCodes);
      
      // Convert API response to frontend format
      result.predictions.forEach(prediction => {
        setForecasts(prev => ({ ...prev, [prediction.product_code]: {
          predictedDemand: prediction.predicted_demand,
          confidence: prediction.confidence_score,
          riskLevel: prediction.risk_level,
          reasoning: `AI prediction based on product attributes and category patterns`
        }}));

        // Generate distribution for each product
        const distribution = [{
          shopId: 1,
          productCode: prediction.product_code,
          variation: {
            size: prediction.size,
            color: prediction.color,
            sizeCode: prediction.attributes.size_code,
            colorCode: prediction.attributes.color_code
          },
          allocatedQuantity: prediction.predicted_demand,
          reasoning: `Initial allocation for new product. ${prediction.confidence_score}% confidence.`
        }];

        setDistributions(prev => ({
          ...prev,
          [prediction.product_code]: {
            data: distribution,
            status: 'COMPLETE',
            forecast: {
              predictedDemand: prediction.predicted_demand,
              confidence: prediction.confidence_score,
              riskLevel: prediction.risk_level
            }
          }
        }));
      });

      console.log(`✅ Generated predictions for ${result.predictions.length} products`);

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

    // Generate forecast if not already generated (for new products, use AI prediction)
    if (!forecasts[product.productCode] && modelStatus === 'READY') {
      try {
        const forecast = await apiService.generateForecast(product.productCode, product);
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
                message="Processing Your Data" 
                subMessage="Setting up AI model and processing datasets..."
              />
            )}

            {!isLoading && !isDataLoaded && !error && (
              <div className="text-center py-12">
                <Package className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Welcome to AI Demand Forecasting</h3>
                <p className="text-gray-600 mb-4">
                  Upload your datasets to begin AI-powered demand forecasting for new products without sales history.
                </p>
                <div className="text-sm text-gray-500 mt-4 space-y-2">
                  <p className="mb-2"><strong>Three-Stage Process:</strong></p>
                  <div className="bg-gray-100 rounded p-3 text-left inline-block space-y-1">
                    <div>1. <strong>Training Data:</strong> Historical sales (2022-2024) + optional inventory data</div>
                    <div>2. <strong>Prediction Data:</strong> New products (2025) for demand forecasting</div>
                    <div>3. <strong>AI Training:</strong> Model learns patterns to predict new product demand</div>
                  </div>
                  <p className="text-xs mt-2">Supports multiple brands with flexible feature sets</p>
                </div>
              </div>
            )}
          </div>
        );

      case 'analytics':
        return (
          <div className="flex-1 overflow-y-auto p-6">
            <AnalyticsDashboard 
              salesData={salesData.length > 0 ? salesData : products} 
              onShowCategories={handleShowCategories}
              brandConfig={brandConfig}
              modelStatus={modelStatus}
            />
          </div>
        );

      case 'categories':
        return (
          <div className="flex-1 overflow-y-auto p-6">
            <CategoriesGrid 
              salesData={salesData.length > 0 ? salesData : products}
              onSelectCategory={handleSelectCategory}
              onBackToAnalytics={handleBackToAnalytics}
              brandConfig={brandConfig}
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
              <h2 className="text-2xl font-bold text-gray-800">AI Demand Forecasting</h2>
              <div className="ml-4 text-sm text-gray-500">
                {currentView === 'analytics' && `New Products Forecasting • ${Object.keys(brandConfig).length > 0 ? 'Multi-Brand Support' : 'Ready'}`}
                {currentView === 'categories' && `Product Categories • Brand Features: ${brandConfig.available_features?.length || 0}`}
                {currentView === 'split' && `${selectedCategory} • AI Predictions`}
                {currentView === 'upload' && `Multi-Stage Setup • Training → Prediction → Forecasting`}
              </div>
            </div>

            <ConnectionStatus isDataLoaded={isDataLoaded} modelStatus={modelStatus} />
          </div>

          {/* Multi-Stage Upload Section - Only show during setup */}
          {currentView === 'upload' && (
            <DataUploadSection 
              onDataLoad={handleDataLoad} 
              isLoading={isLoading} 
              currentStage={currentStage}
              onStageComplete={handleStageComplete}
            />
          )}
        </header>

        {/* Main Content */}
        {renderContent()}
      </div>
    </div>
  );
}

export default InitialStockDistribution;