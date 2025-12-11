import React, { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  Database,
  Upload,
  CheckCircle,
  AlertCircle,
  RefreshCw,
  Settings,
  BarChart3,
  Calendar,
  Target,
  Zap,
  Save,
  Play,
  Pause,
  RotateCcw,
  FileSpreadsheet,
  TrendingUp,
  Award,
  Clock,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Check,
  X,
  Layers,
  GitBranch,
  Activity
} from 'lucide-react';

// API Service for Model Training
const trainingApiService = {
  // Check current model state
  getModelState: async () => {
    const response = await fetch('http://localhost:5000/api/model-state');
    if (!response.ok) throw new Error('Failed to get model state');
    return await response.json();
  },

  // Quick check if model is ready
  quickCheck: async () => {
    const response = await fetch('http://localhost:5000/api/quick-check');
    if (!response.ok) throw new Error('Failed to check model status');
    return await response.json();
  },

  // Load training data
  loadTrainingData: async (salesFile, inventoryFile = null, predictionHorizon = 365) => {
    const formData = new FormData();
    formData.append('sales_file', salesFile);
    formData.append('prediction_horizon', predictionHorizon);
    if (inventoryFile) {
      formData.append('inventory_file', inventoryFile);
    }
    
    const response = await fetch('http://localhost:5000/api/load-training-data', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error('Failed to load training data');
    return await response.json();
  },

  // Load prediction data
  loadPredictionData: async (productsFile) => {
    const formData = new FormData();
    formData.append('products_file', productsFile);
    
    const response = await fetch('http://localhost:5000/api/load-prediction-data', {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) throw new Error('Failed to load prediction data');
    return await response.json();
  },

  // Train model
  trainModel: async (config = {}) => {
    const response = await fetch('http://localhost:5000/api/train-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    
    if (!response.ok) throw new Error('Model training failed');
    return await response.json();
  },

  // Validate model
  validateModel: async (nSplits = 3) => {
    const response = await fetch('http://localhost:5000/api/validate-model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n_splits: nSplits })
    });
    
    if (!response.ok) throw new Error('Model validation failed');
    return await response.json();
  },

  // Save current state
  saveCurrentState: async () => {
    const response = await fetch('http://localhost:5000/api/save-current-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})  // Send empty JSON object to avoid parsing errors
    });
    
    if (!response.ok) throw new Error('Failed to save model state');
    return await response.json();
  },

  // Restore saved model
  restoreModel: async () => {
    const response = await fetch('http://localhost:5000/api/restore-model', {
      method: 'POST'
    });
    
    if (!response.ok) throw new Error('Failed to restore model');
    return await response.json();
  },

  // Reset model state
  resetModelState: async (keepData = false) => {
    const response = await fetch('http://localhost:5000/api/reset-model-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep_data: keepData })
    });
    
    if (!response.ok) throw new Error('Failed to reset model state');
    return await response.json();
  },

  // Get all saved models
  getAllSavedModels: async () => {
    const response = await fetch('http://localhost:5000/api/all-saved-models');
    if (!response.ok) throw new Error('Failed to get saved models');
    return await response.json();
  },

  // Hyperparameter optimization
  optimizeHyperparameters: async (method = 'optuna', nTrials = 50) => {
    const response = await fetch('http://localhost:5000/api/optimize-hyperparameters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method, n_trials: nTrials })
    });
    if (!response.ok) throw new Error('Hyperparameter optimization failed');
    return await response.json();
  },

  // Update ensemble weights
  updateEnsembleWeights: async () => {
    const response = await fetch('http://localhost:5000/api/update-ensemble-weights', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error('Weight update failed');
    return await response.json();
  },

  // Monitor model drift
  monitorModelDrift: async () => {
    const response = await fetch('http://localhost:5000/api/monitor-model-drift', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    if (!response.ok) throw new Error('Drift monitoring failed');
    return await response.json();
  }
};

// Step indicator component
function StepIndicator({ steps, currentStep, completedSteps }) {
  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, index) => {
        const isCompleted = completedSteps.includes(step.id);
        const isCurrent = currentStep === step.id;
        
        return (
          <React.Fragment key={step.id}>
            <div className="flex flex-col items-center">
              <div className={`
                w-12 h-12 rounded-full flex items-center justify-center
                ${isCompleted ? 'bg-green-500 text-white' : 
                  isCurrent ? 'bg-blue-500 text-white' : 
                  'bg-gray-200 text-gray-500'}
                transition-all duration-300
              `}>
                {isCompleted ? <Check className="w-6 h-6" /> : step.icon}
              </div>
              <span className={`mt-2 text-sm font-medium ${
                isCompleted ? 'text-green-600' : 
                isCurrent ? 'text-blue-600' : 
                'text-gray-500'
              }`}>
                {step.label}
              </span>
            </div>
            {index < steps.length - 1 && (
              <div className={`flex-1 h-1 mx-4 rounded ${
                completedSteps.includes(steps[index + 1].id) || 
                (isCompleted && currentStep === steps[index + 1].id) 
                  ? 'bg-green-500' : 'bg-gray-200'
              }`} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}

// File upload component
function FileUploadCard({ 
  title, 
  description, 
  accept, 
  onFileSelect, 
  file, 
  isLoading,
  isUploaded,
  uploadResult,
  icon: Icon
}) {
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      onFileSelect(droppedFile);
    }
  }, [onFileSelect]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  return (
    <div 
      className={`
        border-2 border-dashed rounded-lg p-6 text-center transition-all
        ${isUploaded ? 'border-green-400 bg-green-50' : 
          file ? 'border-blue-400 bg-blue-50' : 
          'border-gray-300 hover:border-blue-400 bg-white'}
      `}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
    >
      <div className="flex flex-col items-center">
        {isLoading ? (
          <RefreshCw className="w-12 h-12 text-blue-500 animate-spin mb-4" />
        ) : isUploaded ? (
          <CheckCircle className="w-12 h-12 text-green-500 mb-4" />
        ) : (
          <Icon className="w-12 h-12 text-gray-400 mb-4" />
        )}
        
        <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
        <p className="text-sm text-gray-600 mb-4">{description}</p>
        
        {file && (
          <div className="mb-4 px-4 py-2 bg-white rounded-lg border">
            <span className="text-sm font-medium text-gray-700">{file.name}</span>
          </div>
        )}
        
        {uploadResult && (
          <div className="mb-4 text-left w-full">
            <div className="text-sm text-green-700 bg-green-100 rounded-lg p-3">
              {uploadResult.message}
              {uploadResult.sales_records && (
                <div className="mt-2 text-xs">
                  <div>📊 {uploadResult.sales_records?.toLocaleString() || uploadResult.brand_config?.sales_records?.toLocaleString()} records</div>
                  {uploadResult.brand_config?.unique_skus && (
                    <div>🏷️ {uploadResult.brand_config.unique_skus.toLocaleString()} unique SKUs</div>
                  )}
                  {uploadResult.products_count && (
                    <div>📦 {uploadResult.products_count.toLocaleString()} products</div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        
        <label className={`
          px-6 py-2 rounded-lg cursor-pointer transition-colors
          ${isUploaded ? 'bg-green-100 text-green-700 hover:bg-green-200' : 
            'bg-blue-600 text-white hover:bg-blue-700'}
        `}>
          <input
            type="file"
            accept={accept}
            className="hidden"
            onChange={(e) => onFileSelect(e.target.files[0])}
            disabled={isLoading}
          />
          {isUploaded ? 'Replace File' : 'Select File'}
        </label>
      </div>
    </div>
  );
}

// Training status card
function TrainingStatusCard({ modelState, onRefresh }) {
  if (!modelState) return null;

  const statusColors = {
    'READY': 'green',
    'TRAINED': 'green',
    'VALIDATED': 'blue',
    'TRAINING_DATA_LOADED': 'yellow',
    'SAVED_MODEL_AVAILABLE': 'purple',
    'NOT_INITIALIZED': 'gray'
  };

  const color = statusColors[modelState.status] || 'gray';

  return (
    <div className={`bg-${color}-50 border border-${color}-200 rounded-lg p-6`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold flex items-center">
          <Activity className={`w-5 h-5 mr-2 text-${color}-600`} />
          Model Status
        </h3>
        <button
          onClick={onRefresh}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <RefreshCw className="w-4 h-4 text-gray-600" />
        </button>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-white rounded-lg">
          <div className={`text-2xl font-bold text-${color}-600`}>
            {modelState.status?.replace(/_/g, ' ') || 'Unknown'}
          </div>
          <div className="text-sm text-gray-600">Current Status</div>
        </div>
        
        <div className="text-center p-3 bg-white rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {modelState.models_available?.length || 0}
          </div>
          <div className="text-sm text-gray-600">Models Trained</div>
        </div>
        
        <div className="text-center p-3 bg-white rounded-lg">
          <div className="text-2xl font-bold text-purple-600">
            {modelState.training_data_info?.sales_records?.toLocaleString() || 'N/A'}
          </div>
          <div className="text-sm text-gray-600">Training Records</div>
        </div>
        
        <div className="text-center p-3 bg-white rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {modelState.prediction_data_info?.products_count?.toLocaleString() || 'N/A'}
          </div>
          <div className="text-sm text-gray-600">Products to Predict</div>
        </div>
      </div>

      {modelState.persisted_state?.last_trained_at && (
        <div className="mt-4 text-sm text-gray-600 flex items-center">
          <Clock className="w-4 h-4 mr-2" />
          Last trained: {new Date(modelState.persisted_state.last_trained_at).toLocaleString()}
        </div>
      )}
    </div>
  );
}

// Validation results component
function ValidationResults({ results }) {
  if (!results || !results.validation_results) return null;

  const summary = results.summary;
  const qualityColors = {
    'EXCELLENT': 'green',
    'GOOD': 'blue',
    'FAIR': 'yellow',
    'NEEDS_IMPROVEMENT': 'red'
  };

  const color = qualityColors[summary?.validation_quality] || 'gray';

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <Award className={`w-5 h-5 mr-2 text-${color}-600`} />
        Validation Results
      </h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-4 bg-blue-50 rounded-lg">
          <div className="text-2xl font-bold text-blue-600">
            {summary?.average_mape?.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Average MAPE</div>
          <div className="text-xs text-gray-500">Lower is better</div>
        </div>
        
        <div className="text-center p-4 bg-green-50 rounded-lg">
          <div className="text-2xl font-bold text-green-600">
            {summary?.accuracy_within_20_percent?.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Within 20%</div>
          <div className="text-xs text-gray-500">Accurate predictions</div>
        </div>
        
        <div className="text-center p-4 bg-purple-50 rounded-lg">
          <div className="text-2xl font-bold text-purple-600">
            {summary?.accuracy_within_50_percent?.toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Within 50%</div>
          <div className="text-xs text-gray-500">Reasonable predictions</div>
        </div>
        
        <div className={`text-center p-4 bg-${color}-50 rounded-lg`}>
          <div className={`text-2xl font-bold text-${color}-600`}>
            {summary?.validation_quality || 'N/A'}
          </div>
          <div className="text-sm text-gray-600">Quality Rating</div>
          <div className="text-xs text-gray-500">Overall assessment</div>
        </div>
      </div>

      {/* Individual split results */}
      <div className="space-y-2">
        <h4 className="font-medium text-gray-700">Split Results</h4>
        {results.validation_results.map((result, idx) => (
          <div key={idx} className="flex justify-between items-center p-3 bg-gray-50 rounded">
            <span className="font-medium">Split {result.split}</span>
            <div className="flex space-x-4 text-sm">
              <span>MAE: {result.mae?.toFixed(2)}</span>
              <span>MAPE: {result.mape?.toFixed(1)}%</span>
              <span>Products: {result.n_products}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Main Model Training Component
export default function ModelTraining() {
  // State management
  const [currentStep, setCurrentStep] = useState('check');
  const [completedSteps, setCompletedSteps] = useState([]);
  const [modelState, setModelState] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // File states
  const [salesFile, setSalesFile] = useState(null);
  const [inventoryFile, setInventoryFile] = useState(null);
  const [predictionFile, setPredictionFile] = useState(null);
  
  // Upload results
  const [trainingDataResult, setTrainingDataResult] = useState(null);
  const [predictionDataResult, setPredictionDataResult] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [validationResult, setValidationResult] = useState(null);

  // Training config
  const [predictionHorizon, setPredictionHorizon] = useState(365);
  const [validationSplits, setValidationSplits] = useState(3);

  // Steps definition
  const steps = [
    { id: 'check', label: 'Check Status', icon: <Activity className="w-6 h-6" /> },
    { id: 'training-data', label: 'Training Data', icon: <Database className="w-6 h-6" /> },
    { id: 'prediction-data', label: 'Prediction Data', icon: <FileSpreadsheet className="w-6 h-6" /> },
    { id: 'train', label: 'Train Model', icon: <Brain className="w-6 h-6" /> },
    { id: 'validate', label: 'Validate', icon: <BarChart3 className="w-6 h-6" /> },
    { id: 'save', label: 'Save', icon: <Save className="w-6 h-6" /> }
  ];

  // Load initial state
  useEffect(() => {
    checkModelState();
  }, []);

  const checkModelState = async () => {
    try {
      setIsLoading(true);
      const state = await trainingApiService.getModelState();
      setModelState(state);
      
      // Determine completed steps based on state
      const completed = [];
      completed.push('check');
      
      if (state.training_data_info || state.in_memory?.training_data_loaded) {
        completed.push('training-data');
      }
      if (state.prediction_data_info || state.in_memory?.prediction_data_loaded) {
        completed.push('prediction-data');
      }
      if (state.status === 'TRAINED' || state.status === 'VALIDATED' || state.status === 'READY') {
        completed.push('train');
      }
      if (state.persisted_state?.validation_results?.length > 0) {
        completed.push('validate');
      }
      if (state.persisted_state?.model_version) {
        completed.push('save');
      }
      
      setCompletedSteps(completed);
      
      // Set current step
      if (state.status === 'READY' || state.status === 'SAVED_MODEL_AVAILABLE') {
        setCurrentStep('save');
      } else if (!completed.includes('training-data')) {
        setCurrentStep('training-data');
      } else if (!completed.includes('prediction-data')) {
        setCurrentStep('prediction-data');
      } else if (!completed.includes('train')) {
        setCurrentStep('train');
      }
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadTrainingData = async () => {
    if (!salesFile) {
      setError('Please select a sales file');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.loadTrainingData(
        salesFile, 
        inventoryFile,
        predictionHorizon
      );
      
      setTrainingDataResult(result);
      setCompletedSteps(prev => [...prev.filter(s => s !== 'training-data'), 'training-data']);
      setCurrentStep('prediction-data');
      await checkModelState();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadPredictionData = async () => {
    if (!predictionFile) {
      setError('Please select a prediction products file');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.loadPredictionData(predictionFile);
      
      setPredictionDataResult(result);
      setCompletedSteps(prev => [...prev.filter(s => s !== 'prediction-data'), 'prediction-data']);
      setCurrentStep('train');
      await checkModelState();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrainModel = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.trainModel({
        validation_split: 0.2
      });
      
      setTrainingResult(result);
      setCompletedSteps(prev => [...prev.filter(s => s !== 'train'), 'train']);
      setCurrentStep('validate');
      await checkModelState();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleValidateModel = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.validateModel(validationSplits);
      
      setValidationResult(result);
      setCompletedSteps(prev => [...prev.filter(s => s !== 'validate'), 'validate']);
      setCurrentStep('save');
      await checkModelState();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveModel = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.saveCurrentState();
      
      setCompletedSteps(prev => [...prev.filter(s => s !== 'save'), 'save']);
      await checkModelState();
      
      alert(`✅ Model saved successfully!\nVersion: ${result.model_version}`);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestoreModel = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const result = await trainingApiService.restoreModel();
      
      await checkModelState();
      
      alert(`✅ ${result.message}`);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetModel = async () => {
    if (!confirm('Are you sure you want to reset the model state? This will clear all training data and models from memory.')) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      await trainingApiService.resetModelState(false);
      
      // Reset local state
      setSalesFile(null);
      setInventoryFile(null);
      setPredictionFile(null);
      setTrainingDataResult(null);
      setPredictionDataResult(null);
      setTrainingResult(null);
      setValidationResult(null);
      setCompletedSteps(['check']);
      setCurrentStep('training-data');
      
      await checkModelState();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
            <Brain className="w-8 h-8 mr-3 text-purple-600" />
            AI Model Training Center
          </h1>
          <p className="text-gray-600 mt-2">
            Train and manage your demand forecasting model. This is a one-time setup that persists across sessions.
          </p>
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center">
            <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Model Status Card */}
        <TrainingStatusCard modelState={modelState} onRefresh={checkModelState} />

        {/* Quick Actions */}
        {modelState?.status === 'SAVED_MODEL_AVAILABLE' && (
          <div className="mt-6 bg-purple-50 border border-purple-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-purple-900 mb-3 flex items-center">
              <GitBranch className="w-5 h-5 mr-2" />
              Saved Model Available
            </h3>
            <p className="text-purple-700 mb-4">
              A previously trained model was found. You can restore it instead of retraining.
            </p>
            <div className="flex space-x-4">
              <button
                onClick={handleRestoreModel}
                disabled={isLoading}
                className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors flex items-center"
              >
                <RotateCcw className="w-5 h-5 mr-2" />
                Restore Saved Model
              </button>
              <button
                onClick={() => setCurrentStep('training-data')}
                className="px-6 py-3 bg-white text-purple-700 border border-purple-300 rounded-lg font-medium hover:bg-purple-50 transition-colors"
              >
                Train New Model
              </button>
            </div>
          </div>
        )}

        {/* Step Indicator */}
        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <StepIndicator 
            steps={steps} 
            currentStep={currentStep} 
            completedSteps={completedSteps} 
          />

          {/* Step Content */}
          <div className="mt-8">
            {/* Training Data Step */}
            {currentStep === 'training-data' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">Step 1: Upload Training Data</h2>
                <p className="text-gray-600">
                  Upload your historical sales data (required) and inventory data (optional) to train the forecasting model.
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <FileUploadCard
                    title="Sales Data (Required)"
                    description="Historical sales CSV with Product Code, Sale Date, Category, etc."
                    accept=".csv"
                    onFileSelect={setSalesFile}
                    file={salesFile}
                    isLoading={isLoading}
                    isUploaded={!!trainingDataResult}
                    uploadResult={trainingDataResult}
                    icon={Database}
                  />
                  
                  <FileUploadCard
                    title="Inventory Data (Optional)"
                    description="Inventory movements CSV for enhanced predictions"
                    accept=".csv"
                    onFileSelect={setInventoryFile}
                    file={inventoryFile}
                    isLoading={false}
                    isUploaded={false}
                    icon={Layers}
                  />
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Prediction Horizon (days)
                  </label>
                  <input
                    type="number"
                    value={predictionHorizon}
                    onChange={(e) => setPredictionHorizon(parseInt(e.target.value) || 365)}
                    className="w-32 px-3 py-2 border border-gray-300 rounded-lg"
                    min="30"
                    max="730"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    How far into the future to predict (default: 365 days / 1 year)
                  </p>
                </div>

                <div className="flex justify-end space-x-4">
                  <button
                    onClick={handleLoadTrainingData}
                    disabled={!salesFile || isLoading}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
                  >
                    {isLoading ? (
                      <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                    ) : (
                      <Upload className="w-5 h-5 mr-2" />
                    )}
                    Load Training Data
                  </button>
                </div>
              </div>
            )}

            {/* Prediction Data Step */}
            {currentStep === 'prediction-data' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">Step 2: Upload Prediction Data</h2>
                <p className="text-gray-600">
                  Upload the new products CSV that you want to generate demand forecasts for.
                </p>

                <FileUploadCard
                  title="New Products Data"
                  description="Products CSV with Product Code, Product Name, Category, Size, Color, etc."
                  accept=".csv"
                  onFileSelect={setPredictionFile}
                  file={predictionFile}
                  isLoading={isLoading}
                  isUploaded={!!predictionDataResult}
                  uploadResult={predictionDataResult}
                  icon={FileSpreadsheet}
                />

                <div className="flex justify-between">
                  <button
                    onClick={() => setCurrentStep('training-data')}
                    className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                  >
                    Back
                  </button>
                  <button
                    onClick={handleLoadPredictionData}
                    disabled={!predictionFile || isLoading}
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center"
                  >
                    {isLoading ? (
                      <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                    ) : (
                      <Upload className="w-5 h-5 mr-2" />
                    )}
                    Load Prediction Data
                  </button>
                </div>
              </div>
            )}

            {/* Train Model Step */}
            {currentStep === 'train' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">Step 3: Train the Model</h2>
                <p className="text-gray-600">
                  Train the ensemble forecasting model using your historical sales data. This may take a few minutes.
                </p>

                {trainingResult && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-green-800 mb-3 flex items-center">
                      <CheckCircle className="w-5 h-5 mr-2" />
                      Training Complete!
                    </h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-blue-600">
                          {trainingResult.training_samples?.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Training Samples</div>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-purple-600">
                          {trainingResult.feature_count}
                        </div>
                        <div className="text-sm text-gray-600">Features</div>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-green-600">
                          {trainingResult.models_trained?.length}
                        </div>
                        <div className="text-sm text-gray-600">Models</div>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-orange-600">
                          {trainingResult.target_range?.mean?.toFixed(1)}
                        </div>
                        <div className="text-sm text-gray-600">Avg Target</div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="flex justify-between">
                  <button
                    onClick={() => setCurrentStep('prediction-data')}
                    className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                  >
                    Back
                  </button>
                  <div className="flex space-x-4">
                    {!trainingResult && (
                      <button
                        onClick={handleTrainModel}
                        disabled={isLoading}
                        className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors disabled:opacity-50 flex items-center"
                      >
                        {isLoading ? (
                          <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        ) : (
                          <Brain className="w-5 h-5 mr-2" />
                        )}
                        Train Model
                      </button>
                    )}
                    {trainingResult && (
                      <button
                        onClick={() => setCurrentStep('validate')}
                        className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center"
                      >
                        Continue to Validation
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Validate Step */}
            {currentStep === 'validate' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">Step 4: Validate the Model</h2>
                <p className="text-gray-600">
                  Run time-series cross-validation to assess model accuracy. This step is optional but recommended.
                </p>

                <div className="bg-gray-50 rounded-lg p-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Validation Splits
                  </label>
                  <input
                    type="number"
                    value={validationSplits}
                    onChange={(e) => setValidationSplits(parseInt(e.target.value) || 3)}
                    className="w-32 px-3 py-2 border border-gray-300 rounded-lg"
                    min="2"
                    max="5"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Number of time-based splits for cross-validation (default: 3)
                  </p>
                </div>

                <ValidationResults results={validationResult} />

                <div className="flex justify-between">
                  <button
                    onClick={() => setCurrentStep('train')}
                    className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                  >
                    Back
                  </button>
                  <div className="flex space-x-4">
                    <button
                      onClick={() => setCurrentStep('save')}
                      className="px-6 py-3 bg-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-300 transition-colors"
                    >
                      Skip Validation
                    </button>
                    {!validationResult && (
                      <button
                        onClick={handleValidateModel}
                        disabled={isLoading}
                        className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors disabled:opacity-50 flex items-center"
                      >
                        {isLoading ? (
                          <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                        ) : (
                          <BarChart3 className="w-5 h-5 mr-2" />
                        )}
                        Run Validation
                      </button>
                    )}
                    {validationResult && (
                      <button
                        onClick={() => setCurrentStep('save')}
                        className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors flex items-center"
                      >
                        Continue to Save
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Save Step */}
            {currentStep === 'save' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-gray-800">Step 5: Save Model</h2>
                <p className="text-gray-600">
                  Save your trained model for future use. You can now use the AI Stock Distribution screen to generate predictions without retraining.
                </p>

                <div className="bg-green-50 border border-green-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-green-800 mb-3 flex items-center">
                    <CheckCircle className="w-5 h-5 mr-2" />
                    Model Ready!
                  </h3>
                  <p className="text-green-700 mb-4">
                    Your model is trained and ready to generate predictions. Click "Save Model" to persist it for future sessions.
                  </p>
                  
                  <div className="flex space-x-4">
                    <button
                      onClick={handleSaveModel}
                      disabled={isLoading}
                      className="px-6 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center"
                    >
                      {isLoading ? (
                        <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                      ) : (
                        <Save className="w-5 h-5 mr-2" />
                      )}
                      Save Model
                    </button>
                  </div>
                </div>

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-blue-800 mb-2">Next Steps</h3>
                  <ul className="text-blue-700 space-y-2">
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Go to "AI Stock Distribution" to generate demand forecasts
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Select categories and generate predictions
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Export results for inventory planning
                    </li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Reset Button */}
        {completedSteps.length > 1 && (
          <div className="mt-6 flex justify-end">
            <button
              onClick={handleResetModel}
              disabled={isLoading}
              className="px-4 py-2 text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors flex items-center"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset & Start Over
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

