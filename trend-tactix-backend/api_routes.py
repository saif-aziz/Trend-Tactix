from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import traceback

# Import our OPTIMIZED forecasting logic
# from optimized_forecaster import OptimizedInventoryForecaster
from advanced_model_optimization import AdvancedOptimizedForecaster

app = Flask(__name__)
CORS(app, 
     origins=['*'],  # Allow all origins
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

# Global variables
forecaster = None
training_sales_data = None
training_inventory_data = None
prediction_products_data = None
trained_model = None
brand_config = {}
validation_results = []

DATA_FOLDER = 'data'

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/api/debug-routes', methods=['GET'])
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'rule': str(rule)
        })
    return jsonify(routes)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        training_status = 'LOADED' if training_sales_data is not None else 'NOT_LOADED'
        model_status = 'TRAINED' if trained_model is not None else 'NOT_TRAINED'
        prediction_status = 'LOADED' if prediction_products_data is not None else 'NOT_LOADED'
        validation_status = 'COMPLETED' if validation_results else 'NOT_RUN'
        
        return jsonify({
            'status': 'healthy',
            'forecaster_type': 'AdvancedOptimizedForecaster',
            'training_data_status': training_status,
            'model_status': model_status,
            'prediction_data_status': prediction_status,
            'validation_status': validation_status,
            'brand_features': forecaster.brand_features if forecaster else [],
            'prediction_horizon_days': forecaster.prediction_horizon if forecaster else 90,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-training-data', methods=['POST'])
def load_training_data():
    """Load historical sales and inventory data for training"""
    global forecaster, training_sales_data, training_inventory_data, brand_config
    
    try:
        sales_file = None
        inventory_file = None
        
        # Get prediction horizon from request
        prediction_horizon = int(request.form.get('prediction_horizon', 90))
        model_type = request.form.get('model_type', 'ensemble')
        
        # Check for uploaded files
        if 'sales_file' in request.files:
            sales_file = request.files['sales_file']
            if sales_file.filename != '':
                sales_filepath = os.path.join(DATA_FOLDER, 'training_sales.csv')
                sales_file.save(sales_filepath)
            else:
                return jsonify({'error': 'No sales file provided'}), 400
        else:
            # Use existing sales dataset
            sales_filepath = os.path.join(DATA_FOLDER, 'Sale 20222024Training.csv')
            if not os.path.exists(sales_filepath):
                return jsonify({'error': 'Training sales dataset not found. Please upload a file.'}), 400

        # Handle optional inventory file
        inventory_filepath = None
        if 'inventory_file' in request.files:
            inventory_file = request.files['inventory_file']
            if inventory_file.filename != '':
                inventory_filepath = os.path.join(DATA_FOLDER, 'training_inventory.csv')
                inventory_file.save(inventory_filepath)
        else:
            # Check for existing inventory dataset
            inventory_filepath = os.path.join(DATA_FOLDER, 'chunk of inventory  Sheet1.csv')
            if not os.path.exists(inventory_filepath):
                inventory_filepath = None

        # Initialize OPTIMIZED forecaster
        # forecaster = OptimizedInventoryForecaster(
        #     model_type=model_type,
        #     prediction_horizon_days=prediction_horizon
        # )

        forecaster = AdvancedOptimizedForecaster(
        model_type=model_type,
        prediction_horizon_days=prediction_horizon
        )
        
        # Load training data
        print("Loading training data...")
        sales_df = pd.read_csv(sales_filepath)
        sales_df['Sale Date'] = pd.to_datetime(sales_df['Sale Date'])
        sales_df.columns = sales_df.columns.str.strip()
        
        training_sales_data = sales_df
        
        # Load inventory if available
        if inventory_filepath and os.path.exists(inventory_filepath):
            inventory_df = pd.read_csv(inventory_filepath)
            inventory_df['T_Date'] = pd.to_datetime(inventory_df['T_Date'])
            inventory_df.columns = inventory_df.columns.str.strip()
            training_inventory_data = inventory_df
        else:
            training_inventory_data = None
        
        # Auto-detect brand features
        forecaster.brand_features = forecaster._detect_brand_features(training_sales_data)
        
        print(f"✅ Sales data loaded: {len(training_sales_data):,} records")
        print(f"   📅 Date range: {training_sales_data['Sale Date'].min()} to {training_sales_data['Sale Date'].max()}")
        print(f"   🏷️ Unique SKUs: {training_sales_data['Product Code'].nunique():,}")
        print(f"   🎯 Detected features: {forecaster.brand_features}")
        
        if training_inventory_data is not None:
            print(f"✅ Inventory data loaded: {len(training_inventory_data):,} records")
        
        # Store brand configuration
        brand_config = {
            'available_features': forecaster.brand_features,
            'sales_records': len(training_sales_data),
            'inventory_records': len(training_inventory_data) if training_inventory_data is not None else 0,
            'date_range': {
                'start': training_sales_data['Sale Date'].min().isoformat(),
                'end': training_sales_data['Sale Date'].max().isoformat()
            },
            'unique_skus': training_sales_data['Product Code'].nunique(),
            'categories': training_sales_data['Category'].unique().tolist() if 'Category' in training_sales_data.columns else [],
            'prediction_horizon_days': prediction_horizon,
            'model_type': model_type
        }
        
        return jsonify({
            'message': 'Training data loaded successfully',
            'brand_config': brand_config,
            'has_inventory_data': training_inventory_data is not None,
            'forecaster_type': 'OptimizedInventoryForecaster'
        })
        
    except Exception as e:
        print(f"💥 ERROR in load_training_data: {str(e)}")
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Training data loading failed: {str(e)}'}), 500

@app.route('/api/load-prediction-data', methods=['POST'])
def load_prediction_data():
    """Load new products for prediction"""
    global prediction_products_data
    
    try:
        print("🚀 load_prediction_data endpoint called!")
        print("📁 Request files:", list(request.files.keys()))
        print("📝 Request form:", dict(request.form))
        print("🧠 Forecaster status:", forecaster is not None)
        
        if forecaster is None:
            print("❌ Forecaster is None - training data must be loaded first")
            return jsonify({'error': 'Training data must be loaded first'}), 400
            
        products_file = None
        
        # Check for uploaded file
        if 'products_file' in request.files:
            products_file = request.files['products_file']
            print(f"📦 Products file: {products_file.filename}")
            if products_file.filename != '':
                products_filepath = os.path.join(DATA_FOLDER, 'prediction_products.csv')
                products_file.save(products_filepath)
                print(f"💾 Saved products file to: {products_filepath}")
            else:
                print("❌ No products file provided")
                return jsonify({'error': 'No products file provided'}), 400
        else:
            print("❌ 'products_file' not in request.files")
            # Use existing products dataset
            products_filepath = os.path.join(DATA_FOLDER, 'Products2025Prediction Data.csv')
            print(f"🔍 Checking for existing file: {products_filepath}")
            if not os.path.exists(products_filepath):
                print("❌ Prediction products dataset not found")
                return jsonify({'error': 'Prediction products dataset not found. Please upload a file.'}), 400

        print("📖 Loading prediction data...")
        # Load prediction data
        products_df = pd.read_csv(products_filepath)
        products_df.columns = products_df.columns.str.strip()
        prediction_products_data = products_df
        
        # Check feature compatibility with training data
        prediction_features = forecaster._detect_brand_features(prediction_products_data)
        common_features = set(forecaster.brand_features) & set(prediction_features)
        missing_features = set(forecaster.brand_features) - set(prediction_features)
        new_features = set(prediction_features) - set(forecaster.brand_features)
        
        print(f"✅ Prediction data loaded: {len(prediction_products_data):,} new products")
        print(f"   🔗 Common features: {list(common_features)}")
        if missing_features:
            print(f"   ⚠️ Missing features (will be imputed): {list(missing_features)}")
        if new_features:
            print(f"   🆕 New features (will be ignored): {list(new_features)}")
        
        # Extract categories for frontend
        categories = prediction_products_data['Category'].unique().tolist() if 'Category' in prediction_products_data.columns else []
        
        print("🎉 Returning success response...")
        return jsonify({
            'message': 'Prediction data loaded successfully',
            'products_count': len(prediction_products_data),
            'categories': categories,
            'common_features': list(common_features),
            'missing_features': list(missing_features),
            'sample_products': extract_sample_products(prediction_products_data)
        })
        
    except Exception as e:
        print(f"💥 ERROR in load_prediction_data: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Prediction data loading failed: {str(e)}'}), 500

@app.route('/api/validate-model', methods=['POST'])
def validate_model():
    """NEW ENDPOINT: Run time-series cross-validation"""
    global validation_results
    
    try:
        if forecaster is None or training_sales_data is None:
            return jsonify({'error': 'Training data must be loaded first'}), 400
        
        data = request.get_json() or {}
        n_splits = data.get('n_splits', 3)
        
        print(f"🔍 Starting model validation with {n_splits} splits...")
        
        # Run time-series cross-validation
        validation_results = forecaster.time_series_cross_validate(
            sales_df=training_sales_data,
            inventory_df=training_inventory_data,
            n_splits=n_splits
        )
        
        if validation_results:
            # Calculate summary statistics and convert numpy types to Python types
            avg_mae = float(np.mean([r['mae'] for r in validation_results]))
            avg_mape = float(np.mean([r['mape'] for r in validation_results]))
            avg_within_20 = float(np.mean([r['within_20_pct'] for r in validation_results]))
            avg_within_50 = float(np.mean([r['within_50_pct'] for r in validation_results]))
            
            # Convert all numpy types in validation_results to Python types
            converted_results = []
            for result in validation_results:
                converted_result = {}
                for key, value in result.items():
                    if isinstance(value, (np.integer, np.int64, np.int32)):
                        converted_result[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64, np.float32)):
                        converted_result[key] = float(value)
                    else:
                        converted_result[key] = value
                converted_results.append(converted_result)
            
            return jsonify({
                'message': 'Model validation completed successfully',
                'validation_results': converted_results,
                'summary': {
                    'average_mae': round(avg_mae, 2),
                    'average_mape': round(avg_mape, 1),
                    'accuracy_within_20_percent': round(avg_within_20, 1),
                    'accuracy_within_50_percent': round(avg_within_50, 1),
                    'validation_quality': 'EXCELLENT' if avg_mape < 25 else 'GOOD' if avg_mape < 40 else 'FAIR' if avg_mape < 60 else 'NEEDS_IMPROVEMENT'
                },
                'n_splits': len(validation_results)
            })
        else:
            return jsonify({
                'message': 'Validation completed but no results generated',
                'validation_results': [],
                'summary': {'validation_quality': 'NO_DATA'}
            })
        
    except Exception as e:
        print(f"💥 ERROR in validate_model: {str(e)}")
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Model validation failed: {str(e)}'}), 500

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the optimized forecasting model - FIXED VERSION"""
    global trained_model
    
    try:
        print("🚀 train_model endpoint called!")
        
        if forecaster is None or training_sales_data is None:
            print("❌ Training data not loaded")
            return jsonify({'error': 'Training data must be loaded first'}), 400
        
        # Get training parameters
        data = request.get_json() or {}
        print("📝 Training parameters:", data)
        
        train_end_date = data.get('train_end_date')
        validation_split = data.get('validation_split', 0.2)
        
        if train_end_date:
            train_end_date = pd.to_datetime(train_end_date)
        
        # Create training features with REAL targets
        print("Creating training features with real targets...")
        training_features = forecaster.create_training_features_with_temporal_split(
            sales_df=training_sales_data, 
            inventory_df=training_inventory_data, 
            train_end_date=train_end_date,
            validation_split=validation_split
        )
        
        # Train ADVANCED ensemble model with proper feature preparation
        print("Training ADVANCED ensemble model...")
        model_params = data.get('model_params', {})
        
        # Use the new advanced training method that handles categorical data properly
        trained_model = forecaster.train_ensemble_model_advanced(
            training_features,
            model_params
        )
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(forecaster, 'models') and 'rf' in forecaster.models:
            try:
                rf_model = forecaster.models['rf']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': forecaster.feature_columns,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    feature_importance = importance_df.head(20)  # Top 20 features
            except Exception as e:
                print(f"Could not extract feature importance: {e}")
        
        print("✅ Model trained successfully!")
        
        return jsonify({
            'message': 'Advanced optimized ensemble model trained successfully',
            'training_samples': len(training_features),
            'feature_count': len(forecaster.feature_columns),
            'models_trained': list(forecaster.models.keys()),
            'ensemble_weights': forecaster.ensemble_weights,
            'target_range': {
                'min': int(training_features['target_demand'].min()),
                'max': int(training_features['target_demand'].max()),
                'mean': round(training_features['target_demand'].mean(), 1)
            },
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else [],
            'validation_available': len(validation_results) > 0
        })
        
    except Exception as e:
        print(f"💥 ERROR in train_model: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

@app.route('/api/generate-predictions', methods=['POST'])
def generate_predictions():
    """Generate demand predictions for new products using FIXED optimized system"""
    try:
        if trained_model is None:
            return jsonify({'error': 'Model must be trained first'}), 400
        
        if prediction_products_data is None:
            return jsonify({'error': 'Prediction data must be loaded first'}), 400
        
        data = request.get_json() or {}
        
        # Filter products if specific ones requested
        product_codes = data.get('product_codes', [])
        if product_codes:
            filtered_products = prediction_products_data[
                prediction_products_data['Product Code'].isin(product_codes)
            ].copy()
        else:
            filtered_products = prediction_products_data.copy()
        
        if len(filtered_products) == 0:
            return jsonify({'error': 'No products found for prediction'}), 400
        
        # Create enhanced prediction features
        print(f"Creating enhanced prediction features for {len(filtered_products)} products...")
        prediction_features = forecaster.create_enhanced_prediction_features(filtered_products)
        
        # Generate ADVANCED ensemble predictions with business rules
        print("Generating ADVANCED ensemble predictions...")
        predictions = forecaster.predict_demand_ensemble_advanced(prediction_features)
        
        # Merge with product details
        detailed_predictions = filtered_products.merge(predictions, on='Product Code', how='left')
        
        # Convert to list of dictionaries for JSON response
        predictions_list = []
        for _, row in detailed_predictions.iterrows():
            prediction_dict = {
                'product_code': row['Product Code'],
                'product_name': row.get('Product Name', ''),
                'category': row.get('Category', ''),
                'size': row.get('Size Name', ''),
                'color': row.get('Color Name', ''),
                'predicted_demand': int(row['predicted_demand']),
                'confidence_score': int(row['confidence_score']),
                'risk_level': row['risk_level'],
                'attributes': {
                    'gender': row.get('Gender', ''),
                    'season': row.get('Season', ''),
                    'size_code': row.get('Size Code', ''),
                    'color_code': row.get('Color Code', ''),
                    'line_item': row.get('LineItem', '')
                },
                'business_reasoning': get_prediction_reasoning(row)
            }
            predictions_list.append(prediction_dict)
        
        # Enhanced summary with validation metrics
        summary = {
            'total_products': len(predictions_list),
            'total_predicted_demand': int(predictions['predicted_demand'].sum()),
            'avg_confidence': int(predictions['confidence_score'].mean()),
            'risk_distribution': predictions['risk_level'].value_counts().to_dict(),
            'demand_distribution': {
                'low_demand_1_5': len(predictions[predictions['predicted_demand'] <= 5]),
                'medium_demand_6_15': len(predictions[(predictions['predicted_demand'] > 5) & (predictions['predicted_demand'] <= 15)]),
                'high_demand_16_plus': len(predictions[predictions['predicted_demand'] > 15])
            }
        }
        
        # Add validation insights if available
        if validation_results:
            avg_mape = np.mean([r['mape'] for r in validation_results])
            summary['validation_insights'] = {
                'historical_accuracy_mape': round(avg_mape, 1),
                'confidence_adjustment': 'HIGH' if avg_mape < 30 else 'MEDIUM' if avg_mape < 50 else 'LOW'
            }
        
        return jsonify({
            'message': f'Advanced predictions generated for {len(predictions_list)} products',
            'predictions': predictions_list,
            'summary': summary,
            'model_info': {
                'ensemble_models': list(forecaster.models.keys()),
                'prediction_horizon_days': forecaster.prediction_horizon,
                'features_used': len(forecaster.feature_columns),
                'optimization_features': ['categorical_encoding', 'business_rules', 'ensemble_weighting']
            }
        })
        
    except Exception as e:
        print(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction generation failed: {str(e)}'}), 500


@app.route('/api/products', methods=['GET'])
def get_prediction_products():
    """Get all products available for prediction"""
    try:
        if prediction_products_data is None:
            return jsonify({'error': 'No prediction data loaded'}), 400
            
        # Extract products with basic info
        products = extract_products_from_prediction_data(prediction_products_data)
        
        return jsonify({
            'products': products,
            'count': len(products)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shops', methods=['GET'])
def get_shops():
    """Get shop information"""
    try:
        shops = [{
            'id': 1,
            'name': '(S-12) Packages Mall Lahore',
            'location': 'Lahore',
            'tier': 'FLAGSHIP',
            'performance': {
                'sellThroughRate': 75,
                'returnRate': 8
            }
        }]
        
        return jsonify({'shops': shops})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-forecast', methods=['POST'])
def generate_forecast():
    """Generate forecast for a specific product (backward compatibility)"""
    try:
        if trained_model is None:
            return jsonify({'error': 'Model must be trained first'}), 400
            
        data = request.get_json()
        product_code = data.get('product_code')
        
        if not product_code:
            return jsonify({'error': 'Product code required'}), 400
        
        # Find product in prediction data
        if prediction_products_data is None:
            return jsonify({'error': 'Prediction data not loaded'}), 400
            
        product_data = prediction_products_data[
            prediction_products_data['Product Code'] == product_code
        ]
        
        if len(product_data) == 0:
            return jsonify({'error': 'Product not found in prediction dataset'}), 400
        
        # Generate prediction for single product
        single_product_df = product_data.iloc[[0]]
        prediction_features = forecaster.create_enhanced_prediction_features(single_product_df)
        predictions = forecaster.predict_demand_ensemble(prediction_features)
        
        if len(predictions) > 0:
            result = predictions.iloc[0]
            forecast = {
                'predictedDemand': int(result['predicted_demand']),
                'confidence': int(result['confidence_score']),
                'riskLevel': result['risk_level'],
                'reasoning': get_prediction_reasoning(product_data.iloc[0])
            }
        else:
            forecast = {
                'predictedDemand': 5,
                'confidence': 40,
                'riskLevel': 'MEDIUM',
                'reasoning': 'Fallback prediction - model error'
            }
        
        return jsonify({
            'product_code': product_code,
            'forecast': forecast
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-distribution', methods=['POST'])
def generate_distribution():
    """Generate smart distribution for a product"""
    try:
        if trained_model is None:
            return jsonify({'error': 'Model must be trained first'}), 400
            
        data = request.get_json()
        product_code = data.get('product_id')  # Frontend sends product_id
        
        if not product_code:
            return jsonify({'error': 'Product code required'}), 400
        
        # Find product in prediction data
        if prediction_products_data is None:
            return jsonify({'error': 'Prediction data not loaded'}), 400
            
        product_data = prediction_products_data[
            prediction_products_data['Product Code'] == product_code
        ]
        
        if len(product_data) == 0:
            return jsonify({'error': 'Product not found in prediction dataset'}), 400
        
        # Generate forecast first
        single_product_df = product_data.iloc[[0]]
        prediction_features = forecaster.create_enhanced_prediction_features(single_product_df)
        predictions = forecaster.predict_demand_ensemble(prediction_features)
        
        if len(predictions) > 0:
            result = predictions.iloc[0]
            forecast = {
                'predictedDemand': int(result['predicted_demand']),
                'confidence': int(result['confidence_score']),
                'riskLevel': result['risk_level'],
                'reasoning': get_prediction_reasoning(product_data.iloc[0])
            }
        else:
            forecast = {
                'predictedDemand': 5,
                'confidence': 40,
                'riskLevel': 'MEDIUM',
                'reasoning': 'Fallback prediction'
            }
        
        # Generate distribution based on product attributes
        distribution = generate_new_product_distribution(product_data.iloc[0], forecast)
        
        return jsonify({
            'product_code': product_code,
            'forecast': forecast,
            'distribution': distribution,
            'reasoning': f"AI-optimized distribution using ensemble model with {forecast['confidence']}% confidence"
        })
        
    except Exception as e:
        print(f"Distribution error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predictions', methods=['POST'])
def batch_predictions():
    """Generate predictions for multiple products"""
    try:
        if trained_model is None:
            return jsonify({'error': 'Model must be trained first'}), 400
            
        data = request.get_json()
        product_codes = data.get('product_codes', [])
        
        if not product_codes:
            return jsonify({'error': 'Product codes required'}), 400
        
        # Generate predictions for batch
        batch_result = generate_predictions()
        if isinstance(batch_result, tuple):  # Error response
            return batch_result
            
        # Filter results for requested products
        all_predictions = batch_result.get_json()['predictions']
        filtered_predictions = [
            pred for pred in all_predictions 
            if pred['product_code'] in product_codes
        ]
        
        return jsonify({
            'message': f'Batch predictions generated for {len(filtered_predictions)} products',
            'predictions': filtered_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-predictions', methods=['POST'])
def export_predictions():
    """Export predictions in POS format"""
    try:
        data = request.get_json()
        product_codes = data.get('product_codes', [])
        
        if not product_codes:
            return jsonify({'error': 'Product codes required'}), 400
        
        # Get predictions for requested products
        predictions_response = generate_predictions()
        if isinstance(predictions_response, tuple):  # Error response
            return predictions_response
            
        all_predictions = predictions_response.get_json()['predictions']
        selected_predictions = [
            pred for pred in all_predictions 
            if pred['product_code'] in product_codes
        ]
        
        # Create enhanced POS export format
        pos_data = {
            'export_date': datetime.now().isoformat(),
            'export_type': 'OPTIMIZED_NEW_PRODUCT_DEMAND_FORECAST',
            'shop': '(S-12) Packages Mall Lahore',
            'total_products': len(selected_predictions),
            'total_predicted_demand': sum(pred['predicted_demand'] for pred in selected_predictions),
            'model_version': f'OptimizedForecasting_v3.0_ensemble',
            'prediction_horizon_days': forecaster.prediction_horizon,
            'ensemble_models': list(forecaster.models.keys()),
            'brand_features': forecaster.brand_features,
            'validation_accuracy': {
                'mape': round(np.mean([r['mape'] for r in validation_results]), 1) if validation_results else 'Not Available'
            },
            'predictions': selected_predictions
        }
        
        return jsonify({
            'message': 'Enhanced export ready',
            'pos_data': pos_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about the trained model"""
    try:
        if forecaster is None:
            return jsonify({'error': 'No forecaster initialized'}), 400
        
        if trained_model is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        
        # Get feature importance from Random Forest if available
        feature_importance = None
        if hasattr(forecaster, 'models') and 'rf' in forecaster.models:
            try:
                rf_model = forecaster.models['rf']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': forecaster.feature_columns,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    feature_importance = importance_df.head(15)  # Top 15 features
            except Exception as e:
                print(f"Could not extract feature importance: {e}")
        
        model_info = {
            'forecaster_type': 'AdvancedOptimizedForecaster',
            'ensemble_models': list(forecaster.models.keys()),
            'ensemble_weights': forecaster.ensemble_weights,
            'prediction_horizon_days': forecaster.prediction_horizon,
            'feature_count': len(forecaster.feature_columns),
            'brand_features': forecaster.brand_features,
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else [],
            'training_config': brand_config,
            'validation_results': {
                'completed': len(validation_results) > 0,
                'average_mape': round(np.mean([r['mape'] for r in validation_results]), 1) if validation_results else None,
                'accuracy_within_20_percent': round(np.mean([r['within_20_pct'] for r in validation_results]), 1) if validation_results else None
            }
        }
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def extract_sample_products(products_df, sample_size=None):
    """Extract sample products for frontend preview"""
    try:
        # Return ALL products instead of just a sample
        if sample_size is None:
            sample_df = products_df  # Return all products
        else:
            sample_df = products_df.head(sample_size)
            
        products = []
        
        for _, row in sample_df.iterrows():
            product = {
                'id': row['Product Code'],
                'productCode': row['Product Code'],
                'name': row.get('Product Name', ''),
                'category': row.get('Category', ''),
                'attributes': {
                    'size': row.get('Size Name', ''),
                    'color': row.get('Color Name', ''),
                    'gender': row.get('Gender', ''),
                    'season': row.get('Season', ''),
                    'line_item': row.get('LineItem', '')
                }
            }
            products.append(product)
        
        return products
        
    except Exception as e:
        print(f"Sample extraction error: {e}")
        return []

def extract_products_from_prediction_data(products_df):
    """Extract products from prediction dataset for frontend"""
    try:
        products = []
        
        for _, row in products_df.iterrows():
            product = {
                'id': row['Product Code'],
                'productCode': row['Product Code'],
                'name': row.get('Product Name', ''),
                'category': row.get('Category', ''),
                'gender': row.get('Gender', ''),
                'season': row.get('Season', ''),
                'attributes': {
                    'size': row.get('Size Name', ''),
                    'color': row.get('Color Name', ''),
                    'gender': row.get('Gender', ''),
                    'season': row.get('Season', ''),
                    'size_code': row.get('Size Code', ''),
                    'color_code': row.get('Color Code', ''),
                    'line_item': row.get('LineItem', '')
                },
                'historicalSales': 0,  # New products have no sales history
                'totalQuantity': 0     # Will be determined by prediction
            }
            products.append(product)
        
        return products
        
    except Exception as e:
        print(f"Product extraction error: {e}")
        return []

def get_prediction_reasoning(product_row):
    """Generate reasoning for prediction based on product attributes"""
    try:
        category = product_row.get('Category', 'Unknown')
        gender = product_row.get('Gender', 'Unknown')
        season = product_row.get('Season', 'Unknown')
        size = product_row.get('Size Name', 'Unknown')
        
        reasoning_parts = []
        
        # Category-based reasoning
        if 'Under Garments' in str(category):
            reasoning_parts.append("High-frequency purchase category")
        elif 'Pant' in str(category) or 'Top' in str(category):
            reasoning_parts.append("Core apparel category with steady demand")
        elif 'Belt' in str(category) or 'Accessories' in str(category):
            reasoning_parts.append("Accessory category with selective demand")
        else:
            reasoning_parts.append("Standard fashion category")
        
        # Gender-based reasoning
        if gender == 'Female':
            reasoning_parts.append("female segment typically shows higher engagement")
        elif gender == 'Male':
            reasoning_parts.append("male segment with focused purchasing patterns")
        
        # Seasonal reasoning
        current_month = datetime.now().month
        if season == 'Open Season':
            reasoning_parts.append("year-round relevance")
        elif season == 'Winter' and current_month in [10, 11, 12, 1, 2]:
            reasoning_parts.append("in-season winter demand boost")
        elif season == 'Summer' and current_month in [4, 5, 6, 7, 8]:
            reasoning_parts.append("in-season summer demand boost")
        elif season not in ['Open Season', 'Unknown']:
            reasoning_parts.append("seasonal timing considered")
        
        # Size reasoning
        if any(size_pattern in str(size) for size_pattern in ['5-6Y', '7-8Y', '9-10Y', '11-12Y', '13-14Y']):
            reasoning_parts.append("popular kids size range")
        
        if reasoning_parts:
            return f"AI prediction based on ensemble model considering: {', '.join(reasoning_parts)}"
        else:
            return "AI prediction using optimized ensemble model with business rules"
            
    except Exception as e:
        return "AI prediction using optimized forecasting model"

def generate_new_product_distribution(product_row, forecast):
    """Generate distribution for new product based on attributes and forecast"""
    try:
        # Enhanced distribution logic for new products
        base_quantity = forecast['predictedDemand']
        confidence = forecast['confidence']
        risk_level = forecast['riskLevel']
        
        # Adjust quantity based on confidence and risk
        if risk_level == 'LOW' and confidence >= 75:
            # High confidence - use full prediction
            allocated_quantity = base_quantity
        elif risk_level == 'MEDIUM':
            # Medium confidence - slightly conservative
            allocated_quantity = max(1, int(base_quantity * 0.9))
        else:
            # High risk - conservative approach
            allocated_quantity = max(1, int(base_quantity * 0.7))
        
        distribution = [{
            'shopId': 1,
            'productCode': product_row['Product Code'],
            'variation': {
                'size': product_row.get('Size Name', 'OS'),
                'color': product_row.get('Color Name', 'Default'),
                'sizeCode': product_row.get('Size Code', 'OS'),
                'colorCode': product_row.get('Color Code', 'DEF')
            },
            'allocatedQuantity': allocated_quantity,
            'reasoning': (
                f"Optimized allocation: {allocated_quantity} units. "
                f"Base prediction: {base_quantity}, Confidence: {confidence}%, Risk: {risk_level}. "
                f"Ensemble model with business rules applied."
            )
        }]
        
        return distribution
        
    except Exception as e:
        print(f"Distribution error: {e}")
        return [{
            'shopId': 1,
            'productCode': product_row.get('Product Code', 'Unknown'),
            'variation': {
                'size': 'OS',
                'color': 'Default',
                'sizeCode': 'OS',
                'colorCode': 'DEF'
            },
            'allocatedQuantity': 5,
            'reasoning': 'Fallback allocation due to processing error'
        }]
    
    # Add these routes to your existing api_routes.py file

@app.route('/api/optimize-hyperparameters', methods=['POST'])
def optimize_hyperparameters():
    """Optimize model hyperparameters - FIXED VERSION"""
    try:
        if forecaster is None or training_sales_data is None:
            return jsonify({'error': 'Training data must be loaded first'}), 400
        
        data = request.get_json() or {}
        optimization_method = data.get('method', 'optuna')  # optuna, grid_search, random_search
        n_trials = data.get('n_trials', 50)
        
        print(f"🔧 Starting FIXED hyperparameter optimization using {optimization_method}")
        
        # Create training features
        training_features = forecaster.create_training_features_with_temporal_split(
            training_sales_data, training_inventory_data
        )
        
        print(f"   📊 Training features shape: {training_features.shape}")
        
        # CRITICAL FIX: Prepare data for ML properly
        X, feature_columns = forecaster._prepare_features_for_ml(training_features)
        y = training_features['target_demand']
        
        print(f"   📊 Prepared features: {X.shape}")
        print(f"   📊 Feature columns: {len(feature_columns)}")
        print(f"   📊 Target shape: {y.shape}")
        
        # FIXED: Check if all features are numeric properly
        numeric_check = True
        non_numeric_cols = []
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                numeric_check = False
                non_numeric_cols.append(col)
        
        print(f"   🎯 All features numeric: {numeric_check}")
        
        if not numeric_check:
            print(f"   ⚠️ Non-numeric columns found: {non_numeric_cols}")
            # Try to convert remaining non-numeric columns
            for col in non_numeric_cols:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                    print(f"   🔧 Converted {col} to numeric")
                except:
                    print(f"   ❌ Could not convert {col} to numeric")
        
        # Final check
        final_check = all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns)
        print(f"   ✅ Final numeric check: {final_check}")
        
        if not final_check:
            return jsonify({'error': 'Some features could not be converted to numeric format'}), 400
        
        # Check for NaN values
        if X.isnull().any().any():
            print("   🔧 Filling NaN values with 0")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("   🔧 Removing rows with NaN targets")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
        
        print(f"   📈 Final data shapes: X={X.shape}, y={y.shape}")
        
        # Optimize hyperparameters with properly prepared data
        best_params = forecaster.optimize_hyperparameters(
            X, y, method=optimization_method, n_trials=n_trials
        )
        
        # Store best parameters
        forecaster.best_params = best_params
        
        return jsonify({
            'message': f'Hyperparameter optimization completed using {optimization_method}',
            'best_parameters': best_params,
            'optimization_method': optimization_method,
            'n_trials': n_trials,
            'feature_preparation': 'categorical_data_properly_encoded',
            'data_info': {
                'samples': len(X),
                'features': len(feature_columns),
                'target_range': [float(y.min()), float(y.max())],
                'all_numeric': final_check
            }
        })
        
    except Exception as e:
        print(f"💥 ERROR in optimize_hyperparameters: {str(e)}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Hyperparameter optimization failed: {str(e)}'}), 500

@app.route('/api/optimize-category-models', methods=['POST'])
def optimize_category_models():
    """Optimize category-specific models - FIXED VERSION"""
    try:
        if forecaster is None or training_sales_data is None:
            return jsonify({'error': 'Training data must be loaded first'}), 400
        
        print("🎯 Starting FIXED category-specific model optimization...")
        
        # Create training features
        training_features = forecaster.create_training_features_with_temporal_split(
            training_sales_data, training_inventory_data
        )
        
        print(f"   📊 Training features shape: {training_features.shape}")
        print(f"   📦 Categories available: {training_features.get('Category_first', pd.Series()).nunique()}")
        
        # Optimize category-specific models with FIXED preprocessing
        category_performance = forecaster.optimize_category_specific_models(training_features)
        
        return jsonify({
            'message': 'Category-specific models optimized successfully with proper categorical encoding',
            'category_performance': category_performance,
            'categories_optimized': len(category_performance),
            'categories_skipped': len([cat for cat in training_features.get('Category_first', pd.Series()).unique() 
                                    if cat not in category_performance]),
            'fix_applied': 'categorical_data_preprocessing_fixed'
        })
        
    except Exception as e:
        print(f"💥 ERROR in optimize_category_models: {str(e)}")
        return jsonify({'error': f'Category optimization failed: {str(e)}'}), 500

@app.route('/api/update-ensemble-weights', methods=['POST'])
def update_ensemble_weights():
    """Update ensemble weights dynamically - FIXED VERSION"""
    try:
        if forecaster is None or not hasattr(forecaster, 'models'):
            return jsonify({'error': 'Models must be trained first'}), 400
        
        # FIXED: Handle empty JSON body
        try:
            data = request.get_json() or {}
        except Exception as e:
            print(f"   📝 No JSON data provided, using defaults: {e}")
            data = {}
        
        # Get validation data (use recent period for weight updates)
        max_date = training_sales_data['Sale Date'].max()
        val_start = max_date - timedelta(days=90)
        
        val_sales = training_sales_data[training_sales_data['Sale Date'] >= val_start]
        
        if len(val_sales) == 0:
            return jsonify({'error': 'No recent validation data available'}), 400
        
        # Create validation features
        val_features = forecaster.create_training_features_with_temporal_split(
            val_sales, training_inventory_data, max_date
        )
        
        # CRITICAL FIX: Prepare features properly for ML with consistency
        X_val, _ = forecaster._prepare_features_for_ml(val_features)
        y_val = val_features['target_demand']
        
        print(f"   📊 Validation data shape: {X_val.shape}")
        print(f"   🎯 Feature consistency: {X_val.shape[1]} features")
        
        # Get individual model predictions
        model_predictions = {}
        for name, model in forecaster.models.items():
            try:
                model_predictions[name] = model.predict(X_val)
                print(f"   ✅ {name} predictions generated")
            except Exception as e:
                print(f"   ⚠️ {name} prediction failed: {e}")
        
        if not model_predictions:
            return jsonify({'error': 'No models could generate predictions for weight update'}), 400
        
        # Update weights dynamically
        new_weights = forecaster.update_ensemble_weights_dynamically(
            X_val, y_val, model_predictions
        )
        
        # Update learning rate based on performance
        forecaster.adapt_learning_rate(None)
        
        return jsonify({
            'message': 'Ensemble weights updated successfully with feature consistency',
            'new_weights': new_weights,
            'learning_rate': forecaster.learning_rate,
            'performance_history_length': len(forecaster.performance_history),
            'models_evaluated': list(model_predictions.keys()),
            'validation_samples': len(X_val),
            'feature_consistency': 'fixed'
        })
        
    except Exception as e:
        print(f"💥 ERROR in update_ensemble_weights: {str(e)}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Weight update failed: {str(e)}'}), 500



@app.route('/api/monitor-model-drift', methods=['POST'])
def monitor_model_drift():
    """Monitor model drift and performance degradation - FIXED VERSION"""
    try:
        if forecaster is None or not hasattr(forecaster, 'models'):
            return jsonify({'error': 'Models must be trained first'}), 400
        
        # Use recent data for drift monitoring
        max_date = training_sales_data['Sale Date'].max()
        recent_start = max_date - timedelta(days=30)  # Last 30 days
        
        recent_sales = training_sales_data[training_sales_data['Sale Date'] >= recent_start]
        
        if len(recent_sales) == 0:
            return jsonify({'error': 'No recent data available for drift monitoring'}), 400
        
        # Create features for recent data
        recent_features = forecaster.create_training_features_with_temporal_split(
            recent_sales, training_inventory_data, max_date
        )
        
        # CRITICAL FIX: Prepare features properly for ML with consistency
        X_recent, available_features = forecaster._prepare_features_for_ml(recent_features)
        y_recent = recent_features['target_demand']
        
        print(f"   📊 Recent data shape: {X_recent.shape}")
        print(f"   🎯 Available features: {len(available_features)}")
        print(f"   ✅ Feature consistency: {X_recent.shape[1]} features")
        
        # Generate predictions using the first available model
        predictions = None
        model_used = None
        for name, model in forecaster.models.items():
            try:
                predictions = model.predict(X_recent)
                model_used = name
                print(f"   ✅ Used {name} for drift monitoring")
                break
            except Exception as e:
                print(f"   ⚠️ {name} prediction failed: {e}")
                continue
        
        if predictions is None:
            return jsonify({'error': 'All models failed to generate predictions for drift monitoring'}), 400
        
        # Monitor drift
        drift_report = forecaster.monitor_model_drift(predictions, y_recent)
        
        return jsonify({
            'message': 'Model drift monitoring completed with feature consistency',
            'drift_report': drift_report,
            'samples_analyzed': len(recent_sales),
            'monitoring_period_days': 30,
            'features_used': len(available_features),
            'model_used_for_monitoring': model_used,
            'feature_consistency': 'fixed'
        })
        
    except Exception as e:
        print(f"💥 ERROR in monitor_model_drift: {str(e)}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Drift monitoring failed: {str(e)}'}), 500

@app.route('/api/save-model-version', methods=['POST'])
def save_model_version():
    """Save current model version"""
    try:
        if forecaster is None or not hasattr(forecaster, 'models'):
            return jsonify({'error': 'Models must be trained first'}), 400
        
        data = request.get_json() or {}
        version_name = data.get('version_name')
        
        # Save models with versioning
        model_dir = forecaster.save_models(version=version_name)
        
        return jsonify({
            'message': 'Model version saved successfully',
            'model_directory': model_dir,
            'version': version_name or 'auto-generated',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"💥 ERROR in save_model_version: {str(e)}")
        return jsonify({'error': f'Model saving failed: {str(e)}'}), 500

@app.route('/api/load-model-version', methods=['POST'])
def load_model_version():
    """Load specific model version"""
    try:
        data = request.get_json()
        version = data.get('version')
        
        if not version:
            return jsonify({'error': 'Version parameter is required'}), 400
        
        # Load model version
        metadata = forecaster.load_models(version)
        
        return jsonify({
            'message': f'Model version {version} loaded successfully',
            'metadata': metadata,
            'models_loaded': list(forecaster.models.keys()),
            'ensemble_weights': forecaster.ensemble_weights
        })
        
    except Exception as e:
        print(f"💥 ERROR in load_model_version: {str(e)}")
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/get-optimization-report', methods=['GET'])
def get_optimization_report():
    """Get comprehensive optimization report"""
    try:
        if forecaster is None:
            return jsonify({'error': 'Forecaster not initialized'}), 400
        
        # Generate optimization report
        report = forecaster.get_optimization_report()
        
        # Add additional metrics
        if hasattr(forecaster, 'models') and forecaster.models:
            report['models_trained'] = list(forecaster.models.keys())
        
        # Add training data info
        if training_sales_data is not None:
            report['training_data_size'] = len(training_sales_data)
            report['date_range'] = {
                'start': training_sales_data['Sale Date'].min().isoformat(),
                'end': training_sales_data['Sale Date'].max().isoformat()
            }
        
        return jsonify({
            'message': 'Optimization report generated',
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"💥 ERROR in get_optimization_report: {str(e)}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/update-with-new-data', methods=['POST'])
def update_with_new_data():
    """Update models with new sales data (online learning)"""
    try:
        if forecaster is None:
            return jsonify({'error': 'Forecaster not initialized'}), 400
        
        # Get new data from request
        data = request.get_json()
        new_sales_records = data.get('new_sales_data', [])
        retrain_threshold = data.get('retrain_threshold', 100)
        
        if not new_sales_records:
            return jsonify({'error': 'No new sales data provided'}), 400
        
        # Convert to DataFrame
        new_sales_df = pd.DataFrame(new_sales_records)
        
        # Ensure required columns exist
        required_cols = ['Product Code', 'Sale Date']
        missing_cols = [col for col in required_cols if col not in new_sales_df.columns]
        
        if missing_cols:
            return jsonify({'error': f'Missing required columns: {missing_cols}'}), 400
        
        # Convert Sale Date to datetime
        new_sales_df['Sale Date'] = pd.to_datetime(new_sales_df['Sale Date'])
        
        # Update models with new data
        forecaster.update_with_new_data(new_sales_df, retrain_threshold)
        
        return jsonify({
            'message': 'Models updated with new data successfully',
            'new_records_processed': len(new_sales_records),
            'retrain_threshold': retrain_threshold,
            'incremental_data_size': len(getattr(forecaster, 'incremental_data', []))
        })
        
    except Exception as e:
        print(f"💥 ERROR in update_with_new_data: {str(e)}")
        return jsonify({'error': f'Data update failed: {str(e)}'}), 500

@app.route('/api/setup-ab-testing', methods=['POST'])
def setup_ab_testing():
    """Setup A/B testing framework"""
    try:
        if forecaster is None:
            return jsonify({'error': 'Forecaster not initialized'}), 400
        
        data = request.get_json() or {}
        test_ratio = data.get('test_ratio', 0.2)
        
        # Setup A/B testing
        forecaster.setup_ab_testing(test_ratio)
        
        return jsonify({
            'message': 'A/B testing framework setup successfully',
            'test_ratio': test_ratio,
            'control_version': 'current',
            'test_version': 'latest'
        })
        
    except Exception as e:
        print(f"💥 ERROR in setup_ab_testing: {str(e)}")
        return jsonify({'error': f'A/B testing setup failed: {str(e)}'}), 500

@app.route('/api/get-available-model-versions', methods=['GET'])
def get_available_model_versions():
    """Get list of available model versions"""
    try:
        model_save_path = getattr(forecaster, 'model_save_path', 'models/')
        
        if not os.path.exists(model_save_path):
            return jsonify({
                'message': 'No model versions found',
                'versions': []
            })
        
        # Get all version directories
        versions = []
        for item in os.listdir(model_save_path):
            if item.startswith('version_') and os.path.isdir(os.path.join(model_save_path, item)):
                version_dir = os.path.join(model_save_path, item)
                metadata_path = os.path.join(version_dir, 'metadata.json')
                
                version_info = {'version': item.replace('version_', '')}
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        version_info.update({
                            'timestamp': metadata.get('timestamp'),
                            'performance': metadata.get('baseline_performance'),
                            'models': list(metadata.get('ensemble_weights', {}).keys())
                        })
                    except:
                        pass
                
                versions.append(version_info)
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'message': f'Found {len(versions)} model versions',
            'versions': versions
        })
        
    except Exception as e:
        print(f"💥 ERROR in get_available_model_versions: {str(e)}")
        return jsonify({'error': f'Version listing failed: {str(e)}'}), 500

@app.route('/api/compare-model-versions', methods=['POST'])
def compare_model_versions():
    """Compare performance between different model versions"""
    try:
        data = request.get_json()
        version1 = data.get('version1')
        version2 = data.get('version2')
        
        if not version1 or not version2:
            return jsonify({'error': 'Both version1 and version2 are required'}), 400
        
        # Load metadata for both versions
        model_save_path = getattr(forecaster, 'model_save_path', 'models/')
        
        def load_version_metadata(version):
            version_dir = os.path.join(model_save_path, f'version_{version}')
            metadata_path = os.path.join(version_dir, 'metadata.json')
            
            if not os.path.exists(metadata_path):
                raise ValueError(f'Version {version} not found')
            
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        metadata1 = load_version_metadata(version1)
        metadata2 = load_version_metadata(version2)
        
        # Compare key metrics
        comparison = {
            'version1': {
                'version': version1,
                'timestamp': metadata1.get('timestamp'),
                'performance': metadata1.get('baseline_performance'),
                'weights': metadata1.get('ensemble_weights')
            },
            'version2': {
                'version': version2,
                'timestamp': metadata2.get('timestamp'),
                'performance': metadata2.get('baseline_performance'),
                'weights': metadata2.get('ensemble_weights')
            }
        }
        
        # Calculate improvement metrics
        if (metadata1.get('baseline_performance') and metadata2.get('baseline_performance')):
            perf1 = metadata1['baseline_performance']
            perf2 = metadata2['baseline_performance']
            
            if 'mae' in perf1 and 'mae' in perf2:
                mae_improvement = (perf1['mae'] - perf2['mae']) / perf1['mae'] * 100
                comparison['mae_improvement_pct'] = mae_improvement
            
            if 'mape' in perf1 and 'mape' in perf2:
                mape_improvement = (perf1['mape'] - perf2['mape']) / perf1['mape'] * 100
                comparison['mape_improvement_pct'] = mape_improvement
        
        return jsonify({
            'message': 'Version comparison completed',
            'comparison': comparison
        })
        
    except Exception as e:
        print(f"💥 ERROR in compare_model_versions: {str(e)}")
        return jsonify({'error': f'Version comparison failed: {str(e)}'}), 500

# Add this helper function for incremental learning
def _create_features_from_sales(self, sales_df):
    """Create features from new sales data for incremental learning"""
    # This is a simplified version - you'd implement the full feature creation logic
    features_df = sales_df.groupby('Product Code').agg({
        'Sale Date': ['count', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    features_df.columns = ['Product Code', 'total_sales', 'first_sale', 'last_sale']
    
    # Create target (simplified)
    features_df['target_demand'] = features_df['total_sales']
    
    return features_df

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)