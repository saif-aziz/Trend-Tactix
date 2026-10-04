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
  Activity,
  Download,
  FileText,
  Info,
  Lightbulb,
  PieChart,
  BarChart2,
  TrendingDown,
  Star,
  Shield,
  Cpu,
  BookOpen,
  ExternalLink
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
  },

  // Get detailed validation report
  getValidationReport: async () => {
    const response = await fetch('http://localhost:5000/api/validation-report');
    if (!response.ok) throw new Error('Failed to get validation report');
    return await response.json();
  }
};

// Step indicator component
function StepIndicator({ steps, currentStep, completedSteps, onStepClick }) {
  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, index) => {
        const isCompleted = completedSteps.includes(step.id);
        const isCurrent = currentStep === step.id;
        const isClickable = isCompleted || isCurrent || 
          (index > 0 && completedSteps.includes(steps[index - 1].id));
        
        return (
          <React.Fragment key={step.id}>
            <div 
              className={`flex flex-col items-center ${isClickable ? 'cursor-pointer' : ''}`}
              onClick={() => isClickable && onStepClick && onStepClick(step.id)}
            >
              <div className={`
                w-12 h-12 rounded-full flex items-center justify-center
                ${isCompleted ? 'bg-green-500 text-white' : 
                  isCurrent ? 'bg-blue-500 text-white' : 
                  'bg-gray-200 text-gray-500'}
                transition-all duration-300
                ${isClickable ? 'hover:ring-4 hover:ring-offset-2 hover:ring-blue-200' : ''}
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

// Comprehensive Validation Results Panel Component
function ValidationResultsPanel({ results, onGetDetailedReport }) {
  const [showDetailedReport, setShowDetailedReport] = useState(false);
  const [detailedReport, setDetailedReport] = useState(null);
  const [loadingReport, setLoadingReport] = useState(false);
  const [activeTab, setActiveTab] = useState('summary');

  // Get color based on quality
  const getQualityColor = (quality) => {
    const colors = {
      'EXCELLENT': { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-700', badge: 'bg-green-100 text-green-800' },
      'GOOD': { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', badge: 'bg-blue-100 text-blue-800' },
      'FAIR': { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', badge: 'bg-yellow-100 text-yellow-800' },
      'NEEDS_IMPROVEMENT': { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-800' },
      'POOR': { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', badge: 'bg-red-100 text-red-800' }
    };
    return colors[quality] || { bg: 'bg-gray-50', border: 'border-gray-200', text: 'text-gray-700', badge: 'bg-gray-100 text-gray-800' };
  };

  const loadDetailedReport = async () => {
    setLoadingReport(true);
    try {
      const report = await trainingApiService.getValidationReport();
      setDetailedReport(report);
      setShowDetailedReport(true);
    } catch (error) {
      console.error('Failed to load detailed report:', error);
    } finally {
      setLoadingReport(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!detailedReport) return;
    
    // Generate HTML content for PDF
    const content = generatePDFContent(detailedReport);
    
    // Create a new window and print
    const printWindow = window.open('', '_blank');
    printWindow.document.write(content);
    printWindow.document.close();
    printWindow.focus();
    
    setTimeout(() => {
      printWindow.print();
    }, 500);
  };

  const generatePDFContent = (report) => {
    const metrics = report.validation_metrics?.summary || {};
    const quality = report.validation_metrics?.quality_label || 'N/A';
    
    return `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Model Validation Report - Trend Tactix</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }
          h1 { color: #4F46E5; border-bottom: 2px solid #4F46E5; padding-bottom: 10px; }
          h2 { color: #374151; margin-top: 30px; }
          .header { display: flex; justify-content: space-between; align-items: center; }
          .date { color: #6B7280; }
          .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
          .metric-card { background: #F3F4F6; padding: 15px; border-radius: 8px; text-align: center; }
          .metric-value { font-size: 24px; font-weight: bold; color: #4F46E5; }
          .metric-label { font-size: 12px; color: #6B7280; }
          .quality-badge { display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; }
          .quality-EXCELLENT { background: #D1FAE5; color: #065F46; }
          .quality-GOOD { background: #DBEAFE; color: #1E40AF; }
          .quality-FAIR { background: #FEF3C7; color: #92400E; }
          .quality-NEEDS_IMPROVEMENT, .quality-POOR { background: #FEE2E2; color: #991B1B; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { border: 1px solid #E5E7EB; padding: 10px; text-align: left; }
          th { background: #F9FAFB; }
          .recommendation { background: #EEF2FF; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4F46E5; }
          .recommendation.HIGH { border-left-color: #EF4444; }
          .recommendation.MEDIUM { border-left-color: #F59E0B; }
          .recommendation.LOW { border-left-color: #10B981; }
          .model-card { background: #F9FAFB; padding: 15px; border-radius: 8px; margin: 10px 0; }
          .interpretation { background: #FEF3C7; padding: 20px; border-radius: 8px; margin: 20px 0; }
          @media print { body { padding: 20px; } }
        </style>
      </head>
      <body>
        <div class="header">
          <h1>🧠 Model Validation Report</h1>
          <div class="date">Generated: ${new Date(report.generated_at).toLocaleString()}</div>
        </div>
        
        <h2>📊 Performance Summary</h2>
        <div style="text-align: center; margin: 20px 0;">
          <span class="quality-badge quality-${quality}">${quality.replace('_', ' ')}</span>
        </div>
        
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-value">${metrics.average_mape?.toFixed(1) || 'N/A'}%</div>
            <div class="metric-label">Average MAPE</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${metrics.accuracy_within_20_percent?.toFixed(1) || 'N/A'}%</div>
            <div class="metric-label">Within 20% Accuracy</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${metrics.r2_estimate?.toFixed(3) || 'N/A'}</div>
            <div class="metric-label">R² Score</div>
          </div>
          <div class="metric-card">
            <div class="metric-value">${metrics.consistency_score?.toFixed(1) || 'N/A'}</div>
            <div class="metric-label">Consistency Score</div>
          </div>
        </div>
        
        <div class="interpretation">
          <strong>📝 Interpretation:</strong> ${report.validation_metrics?.interpretation || 'No interpretation available.'}
          <br><br>
          <strong>Fit Status:</strong> ${report.validation_metrics?.fit_status || 'N/A'}
        </div>
        
        <h2>📂 Dataset Information</h2>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>Total Records</td><td>${report.dataset_info?.total_records?.toLocaleString() || 'N/A'}</td></tr>
          <tr><td>Unique Products</td><td>${report.dataset_info?.unique_products?.toLocaleString() || 'N/A'}</td></tr>
          <tr><td>Categories</td><td>${report.dataset_info?.unique_categories || 'N/A'}</td></tr>
          <tr><td>Features Used</td><td>${report.dataset_info?.features_used || 'N/A'}</td></tr>
          <tr><td>Date Range</td><td>${report.dataset_info?.date_range?.start || 'N/A'} to ${report.dataset_info?.date_range?.end || 'N/A'}</td></tr>
        </table>
        
        <h2>🤖 Models Trained</h2>
        ${report.models?.map(model => `
          <div class="model-card">
            <h3>${model.name}</h3>
            <p>${model.description}</p>
            <p><strong>Best For:</strong> ${model.best_for}</p>
            ${report.ensemble_weights?.[model.key] ? `<p><strong>Ensemble Weight:</strong> ${(report.ensemble_weights[model.key] * 100).toFixed(1)}%</p>` : ''}
          </div>
        `).join('') || '<p>No models available</p>'}
        
        <h2>💡 Recommendations</h2>
        ${report.recommendations?.map(rec => `
          <div class="recommendation ${rec.priority}">
            <strong>[${rec.priority}] ${rec.category}</strong>
            <p>${rec.suggestion}</p>
            <p><em>Expected Impact: ${rec.impact}</em></p>
          </div>
        `).join('') || '<p>No recommendations available</p>'}
        
        <h2>📈 Cross-Validation Splits</h2>
        <table>
          <tr><th>Split</th><th>MAE</th><th>MAPE</th><th>RMSE</th><th>Within 20%</th><th>Products</th></tr>
          ${report.validation_metrics?.splits?.map(split => `
            <tr>
              <td>Split ${split.split}</td>
              <td>${split.mae?.toFixed(2)}</td>
              <td>${split.mape?.toFixed(1)}%</td>
              <td>${split.rmse?.toFixed(2)}</td>
              <td>${split.within_20_pct?.toFixed(1)}%</td>
              <td>${split.n_products}</td>
            </tr>
          `).join('') || '<tr><td colspan="6">No validation data</td></tr>'}
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #E5E7EB; text-align: center; color: #9CA3AF;">
          Generated by Trend Tactix AI Model Training System
        </div>
      </body>
      </html>
    `;
  };

  if (!results || !results.validation_results) return null;

  const summary = results.summary;
  const qualityStyle = getQualityColor(summary?.validation_quality);

  return (
    <div className="space-y-6 mt-6">
      {/* Summary Card */}
      <div className={`${qualityStyle.bg} ${qualityStyle.border} border rounded-xl p-6`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold flex items-center text-gray-900">
            <Award className="w-6 h-6 mr-2 text-purple-600" />
            Validation Results Summary
          </h3>
          <span className={`px-4 py-2 rounded-full text-sm font-bold ${qualityStyle.badge}`}>
            {summary?.validation_quality?.replace('_', ' ') || 'N/A'}
          </span>
        </div>
        
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-3xl font-bold text-blue-600">
              {summary?.average_mape?.toFixed(1) || 'N/A'}%
            </div>
            <div className="text-sm font-medium text-gray-700">MAPE</div>
            <div className="text-xs text-gray-500">Mean Absolute % Error</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-3xl font-bold text-green-600">
              {summary?.accuracy_within_20_percent?.toFixed(0) || 'N/A'}%
            </div>
            <div className="text-sm font-medium text-gray-700">High Accuracy</div>
            <div className="text-xs text-gray-500">Within 20% of actual</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-3xl font-bold text-purple-600">
              {summary?.accuracy_within_50_percent?.toFixed(0) || 'N/A'}%
            </div>
            <div className="text-sm font-medium text-gray-700">Reasonable</div>
            <div className="text-xs text-gray-500">Within 50% of actual</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-3xl font-bold text-orange-600">
              {summary?.average_mae?.toFixed(1) || 'N/A'}
            </div>
            <div className="text-sm font-medium text-gray-700">MAE</div>
            <div className="text-xs text-gray-500">Mean Absolute Error</div>
          </div>
          
          <div className="bg-white rounded-lg p-4 text-center shadow-sm">
            <div className="text-3xl font-bold text-indigo-600">
              {results.n_splits || 0}
            </div>
            <div className="text-sm font-medium text-gray-700">CV Splits</div>
            <div className="text-xs text-gray-500">Cross-validation folds</div>
          </div>
        </div>

        {/* Quick Interpretation */}
        <div className="bg-white rounded-lg p-4 border border-gray-200">
          <div className="flex items-start">
            <Lightbulb className="w-5 h-5 text-yellow-500 mr-3 mt-0.5 flex-shrink-0" />
            <div>
              <h4 className="font-semibold text-gray-800 mb-1">Quick Interpretation</h4>
              <p className="text-gray-600 text-sm">
                {summary?.validation_quality === 'EXCELLENT' && 
                  'Excellent! Your model predictions are highly reliable. Forecasts should closely match actual demand.'}
                {summary?.validation_quality === 'GOOD' && 
                  'Good performance! Predictions are reliable for most products with minor variations expected.'}
                {summary?.validation_quality === 'FAIR' && 
                  'Fair performance. Predictions provide useful guidance but verify for high-value decisions.'}
                {summary?.validation_quality === 'NEEDS_IMPROVEMENT' && 
                  'Model needs improvement. Consider adding more data or cleaning outliers for better accuracy.'}
                {!summary?.validation_quality && 'Run validation to see model performance insights.'}
              </p>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3 mt-4">
          <button
            onClick={loadDetailedReport}
            disabled={loadingReport}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 transition-colors flex items-center"
          >
            {loadingReport ? (
              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <FileText className="w-4 h-4 mr-2" />
            )}
            View Detailed Report
          </button>
          
          {detailedReport && (
            <button
              onClick={handleDownloadPDF}
              className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 transition-colors flex items-center"
            >
              <Download className="w-4 h-4 mr-2" />
              Download PDF Report
            </button>
          )}
        </div>
      </div>

      {/* Cross-Validation Splits */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h4 className="font-semibold text-gray-800 mb-4 flex items-center">
          <BarChart2 className="w-5 h-5 mr-2 text-blue-600" />
          Cross-Validation Split Results
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50">
                <th className="px-4 py-3 text-left font-semibold text-gray-600">Split</th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">MAE</th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">MAPE</th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">Within 20%</th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">Within 50%</th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">Products</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {results.validation_results.map((result, idx) => (
                <tr key={idx} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-medium text-gray-800">Split {result.split}</td>
                  <td className="px-4 py-3 text-right text-gray-600">{result.mae?.toFixed(2)}</td>
                  <td className="px-4 py-3 text-right">
                    <span className={`font-medium ${result.mape < 25 ? 'text-green-600' : result.mape < 40 ? 'text-yellow-600' : 'text-red-600'}`}>
                      {result.mape?.toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right text-gray-600">{result.within_20_pct?.toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right text-gray-600">{result.within_50_pct?.toFixed(1)}%</td>
                  <td className="px-4 py-3 text-right text-gray-600">{result.n_products?.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Report Modal */}
      {showDetailedReport && detailedReport && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 overflow-y-auto">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col my-auto">
            {/* Header */}
            <div className="p-6 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 flex items-center">
                  <BookOpen className="w-6 h-6 mr-3 text-indigo-600" />
                  Detailed Validation Report
                </h2>
                <p className="text-sm text-gray-500 mt-1">
                  Generated: {new Date(detailedReport.generated_at).toLocaleString()}
                </p>
              </div>
              <button
                onClick={() => setShowDetailedReport(false)}
                className="p-2 hover:bg-gray-100 rounded-lg"
              >
                <X className="w-6 h-6 text-gray-500" />
              </button>
            </div>

            {/* Tabs */}
            <div className="border-b border-gray-200 flex-shrink-0">
              <div className="flex space-x-1 px-6">
                {['summary', 'models', 'features', 'recommendations'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab
                        ? 'border-indigo-600 text-indigo-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700'
                    }`}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Content */}
            <div className="p-6 overflow-y-auto flex-1">
              {/* Summary Tab */}
              {activeTab === 'summary' && (
                <div className="space-y-6">
                  {/* Performance Overview */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 text-center">
                      <div className="text-3xl font-bold text-blue-700">
                        {detailedReport.validation_metrics?.summary?.average_mape?.toFixed(1)}%
                      </div>
                      <div className="text-sm font-medium text-blue-800">Average MAPE</div>
                      <div className="text-xs text-blue-600 mt-1">Lower is better</div>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-4 text-center">
                      <div className="text-3xl font-bold text-green-700">
                        {detailedReport.validation_metrics?.summary?.r2_estimate?.toFixed(3)}
                      </div>
                      <div className="text-sm font-medium text-green-800">R² Score</div>
                      <div className="text-xs text-green-600 mt-1">Higher is better</div>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-4 text-center">
                      <div className="text-3xl font-bold text-purple-700">
                        {detailedReport.validation_metrics?.summary?.consistency_score?.toFixed(0)}
                      </div>
                      <div className="text-sm font-medium text-purple-800">Consistency</div>
                      <div className="text-xs text-purple-600 mt-1">Cross-validation stability</div>
                    </div>
                    <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-4 text-center">
                      <div className="text-3xl font-bold text-orange-700">
                        {detailedReport.validation_metrics?.summary?.total_products_validated?.toLocaleString()}
                      </div>
                      <div className="text-sm font-medium text-orange-800">Products Validated</div>
                      <div className="text-xs text-orange-600 mt-1">Across all splits</div>
                    </div>
                  </div>

                  {/* Interpretation */}
                  <div className={`rounded-xl p-5 ${getQualityColor(detailedReport.validation_metrics?.quality_label).bg}`}>
                    <div className="flex items-start">
                      <Info className="w-5 h-5 mr-3 mt-0.5 text-indigo-600 flex-shrink-0" />
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-2">Model Interpretation</h4>
                        <p className="text-gray-700">{detailedReport.validation_metrics?.interpretation}</p>
                        <div className="mt-3 inline-flex items-center px-3 py-1 bg-white rounded-full text-sm font-medium">
                          <Shield className="w-4 h-4 mr-2" />
                          Fit Status: {detailedReport.validation_metrics?.fit_status}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Dataset Info */}
                  <div className="bg-gray-50 rounded-xl p-5">
                    <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                      <Database className="w-5 h-5 mr-2 text-gray-600" />
                      Dataset Information
                    </h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="bg-white rounded-lg p-3">
                        <div className="text-xl font-bold text-gray-900">
                          {detailedReport.dataset_info?.total_records?.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Total Records</div>
                      </div>
                      <div className="bg-white rounded-lg p-3">
                        <div className="text-xl font-bold text-gray-900">
                          {detailedReport.dataset_info?.unique_products?.toLocaleString()}
                        </div>
                        <div className="text-sm text-gray-600">Unique Products</div>
                      </div>
                      <div className="bg-white rounded-lg p-3">
                        <div className="text-xl font-bold text-gray-900">
                          {detailedReport.dataset_info?.unique_categories}
                        </div>
                        <div className="text-sm text-gray-600">Categories</div>
                      </div>
                      <div className="bg-white rounded-lg p-3">
                        <div className="text-xl font-bold text-gray-900">
                          {detailedReport.dataset_info?.features_used}
                        </div>
                        <div className="text-sm text-gray-600">Features Used</div>
                      </div>
                      <div className="bg-white rounded-lg p-3 col-span-2">
                        <div className="text-xl font-bold text-gray-900">
                          {detailedReport.dataset_info?.date_range?.start} to {detailedReport.dataset_info?.date_range?.end}
                        </div>
                        <div className="text-sm text-gray-600">Date Range ({detailedReport.dataset_info?.date_range?.days} days)</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Models Tab */}
              {activeTab === 'models' && (
                <div className="space-y-6">
                  <p className="text-gray-600">
                    These are the machine learning models trained in the ensemble. Each model contributes to the final prediction based on its assigned weight.
                  </p>
                  
                  {detailedReport.models?.map((model, idx) => (
                    <div key={idx} className="bg-gray-50 rounded-xl p-5 border border-gray-200">
                      <div className="flex items-start justify-between mb-3">
                        <div className="flex items-center">
                          <Cpu className="w-6 h-6 mr-3 text-indigo-600" />
                          <div>
                            <h4 className="font-bold text-lg text-gray-900">{model.name}</h4>
                            <span className="text-sm text-gray-500">Model Key: {model.key}</span>
                          </div>
                        </div>
                        {detailedReport.ensemble_weights?.[model.key] && (
                          <div className="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm font-medium">
                            Weight: {(detailedReport.ensemble_weights[model.key] * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                      
                      <p className="text-gray-700 mb-4">{model.description}</p>
                      
                      <div className="grid md:grid-cols-2 gap-4 mb-4">
                        <div>
                          <h5 className="font-medium text-green-700 mb-2 flex items-center">
                            <CheckCircle className="w-4 h-4 mr-1" /> Pros
                          </h5>
                          <ul className="space-y-1">
                            {model.pros?.map((pro, i) => (
                              <li key={i} className="text-sm text-gray-600 flex items-start">
                                <span className="text-green-500 mr-2">•</span>
                                {pro}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <h5 className="font-medium text-red-700 mb-2 flex items-center">
                            <AlertCircle className="w-4 h-4 mr-1" /> Cons
                          </h5>
                          <ul className="space-y-1">
                            {model.cons?.map((con, i) => (
                              <li key={i} className="text-sm text-gray-600 flex items-start">
                                <span className="text-red-500 mr-2">•</span>
                                {con}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                      
                      <div className="bg-white rounded-lg p-3 border border-gray-200">
                        <span className="font-medium text-gray-700">Best For: </span>
                        <span className="text-gray-600">{model.best_for}</span>
                      </div>
                      
                      {/* Top Features */}
                      {model.top_features && model.top_features.length > 0 && (
                        <div className="mt-4">
                          <h5 className="font-medium text-gray-700 mb-2">Top Features by Importance</h5>
                          <div className="flex flex-wrap gap-2">
                            {model.top_features.slice(0, 8).map((feat, i) => (
                              <span key={i} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">
                                {feat.feature}: {(feat.importance * 100).toFixed(1)}%
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}

                  {/* Ensemble Explanation */}
                  <div className="bg-indigo-50 rounded-xl p-5 border border-indigo-200">
                    <h4 className="font-semibold text-indigo-900 mb-2 flex items-center">
                      <Layers className="w-5 h-5 mr-2" />
                      How Ensemble Works
                    </h4>
                    <p className="text-indigo-800 text-sm">
                      The ensemble combines predictions from multiple models weighted by their individual performance. 
                      This approach typically outperforms any single model by leveraging the strengths of each while 
                      minimizing their weaknesses. The final prediction is a weighted average of all model predictions.
                    </p>
                  </div>
                </div>
              )}

              {/* Features Tab */}
              {activeTab === 'features' && (
                <div className="space-y-6">
                  <p className="text-gray-600">
                    Features are the input variables used by the model to make predictions. More relevant features typically lead to better predictions.
                  </p>
                  
                  <div className="bg-gray-50 rounded-xl p-5">
                    <h4 className="font-semibold text-gray-900 mb-4">Features Used ({detailedReport.dataset_info?.features_used} total)</h4>
                    <div className="flex flex-wrap gap-2">
                      {detailedReport.dataset_info?.feature_list?.map((feature, idx) => (
                        <span
                          key={idx}
                          className="px-3 py-1 bg-white border border-gray-200 rounded-lg text-sm text-gray-700"
                        >
                          {feature}
                        </span>
                      ))}
                      {detailedReport.dataset_info?.features_used > 30 && (
                        <span className="px-3 py-1 bg-gray-200 text-gray-600 rounded-lg text-sm">
                          +{detailedReport.dataset_info.features_used - 30} more
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="bg-blue-50 rounded-xl p-5 border border-blue-200">
                    <h4 className="font-semibold text-blue-900 mb-2">Feature Engineering Applied</h4>
                    <ul className="space-y-2 text-blue-800 text-sm">
                      <li className="flex items-start"><Check className="w-4 h-4 mr-2 mt-0.5" /> Seasonal features (month, quarter, season indicators)</li>
                      <li className="flex items-start"><Check className="w-4 h-4 mr-2 mt-0.5" /> Category and product aggregations</li>
                      <li className="flex items-start"><Check className="w-4 h-4 mr-2 mt-0.5" /> Historical sales patterns and trends</li>
                      <li className="flex items-start"><Check className="w-4 h-4 mr-2 mt-0.5" /> Size and color encoding</li>
                      <li className="flex items-start"><Check className="w-4 h-4 mr-2 mt-0.5" /> Price and discount features</li>
                    </ul>
                  </div>
                </div>
              )}

              {/* Recommendations Tab */}
              {activeTab === 'recommendations' && (
                <div className="space-y-4">
                  <p className="text-gray-600 mb-6">
                    Based on the validation results, here are recommendations to improve model accuracy:
                  </p>
                  
                  {detailedReport.recommendations?.map((rec, idx) => (
                    <div
                      key={idx}
                      className={`rounded-xl p-5 border-l-4 ${
                        rec.priority === 'HIGH' ? 'bg-red-50 border-red-500' :
                        rec.priority === 'MEDIUM' ? 'bg-yellow-50 border-yellow-500' :
                        'bg-green-50 border-green-500'
                      }`}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex items-center mb-2">
                            <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                              rec.priority === 'HIGH' ? 'bg-red-200 text-red-800' :
                              rec.priority === 'MEDIUM' ? 'bg-yellow-200 text-yellow-800' :
                              'bg-green-200 text-green-800'
                            }`}>
                              {rec.priority}
                            </span>
                            <span className="ml-2 text-sm font-medium text-gray-600">{rec.category}</span>
                          </div>
                          <p className="font-medium text-gray-900 mb-1">{rec.suggestion}</p>
                          <p className="text-sm text-gray-600">
                            <TrendingUp className="w-4 h-4 inline mr-1" />
                            Expected Impact: {rec.impact}
                          </p>
                        </div>
                        <Lightbulb className={`w-6 h-6 flex-shrink-0 ${
                          rec.priority === 'HIGH' ? 'text-red-500' :
                          rec.priority === 'MEDIUM' ? 'text-yellow-500' :
                          'text-green-500'
                        }`} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-6 border-t border-gray-200 flex justify-between items-center flex-shrink-0 bg-gray-50">
              <button
                onClick={() => setShowDetailedReport(false)}
                className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Close
              </button>
              <button
                onClick={handleDownloadPDF}
                className="px-6 py-2 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 flex items-center"
              >
                <Download className="w-4 h-4 mr-2" />
                Download Full Report (PDF)
              </button>
            </div>
          </div>
        </div>
      )}
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
            onStepClick={(stepId) => setCurrentStep(stepId)}
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

                <ValidationResultsPanel results={validationResult} />

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

                {/* Validation Reminder if not done */}
                {!completedSteps.includes('validate') && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-amber-800 mb-3 flex items-center">
                      <BarChart3 className="w-5 h-5 mr-2" />
                      Model Validation Recommended
                    </h3>
                    <p className="text-amber-700 mb-4">
                      You haven't validated your model yet. Validation helps you understand how accurate your predictions will be.
                    </p>
                    <button
                      onClick={() => setCurrentStep('validate')}
                      className="px-6 py-3 bg-amber-600 text-white rounded-lg font-medium hover:bg-amber-700 transition-colors flex items-center"
                    >
                      <BarChart3 className="w-5 h-5 mr-2" />
                      Run Validation Now
                    </button>
                  </div>
                )}

                {/* Show validation summary if completed */}
                {validationResult && completedSteps.includes('validate') && (
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold text-blue-800 mb-3 flex items-center">
                      <Award className="w-5 h-5 mr-2" />
                      Validation Complete
                    </h3>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-blue-600">
                          {validationResult.summary?.average_mape?.toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">MAPE</div>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-green-600">
                          {validationResult.summary?.accuracy_within_20_percent?.toFixed(0)}%
                        </div>
                        <div className="text-sm text-gray-600">Within 20%</div>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg">
                        <div className="text-xl font-bold text-purple-600">
                          {validationResult.summary?.validation_quality}
                        </div>
                        <div className="text-sm text-gray-600">Quality</div>
                      </div>
                    </div>
                    <button
                      onClick={() => setCurrentStep('validate')}
                      className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center"
                    >
                      <ExternalLink className="w-4 h-4 mr-1" />
                      View Full Validation Report
                    </button>
                  </div>
                )}

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

                <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Next Steps</h3>
                  <ul className="text-gray-700 space-y-2">
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                      Go to "AI Stock Distribution" to generate demand forecasts
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                      Select categories and generate predictions
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
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

