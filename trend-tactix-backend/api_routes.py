from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import traceback

# Import our forecasting logic
from sequential_inventory_forecaster import SequentialInventoryForecaster

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
        
        return jsonify({
            'status': 'healthy',
            'training_data_status': training_status,
            'model_status': model_status,
            'prediction_data_status': prediction_status,
            'brand_features': forecaster.brand_features if forecaster else [],
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

        # Initialize forecaster
        model_type = request.form.get('model_type', 'random_forest')
        forecaster = SequentialInventoryForecaster(model_type=model_type)
        
        # Load training data
        training_sales_data, training_inventory_data = forecaster.load_training_data(
            sales_filepath, inventory_filepath
        )
        
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
            'categories': training_sales_data['Category'].unique().tolist() if 'Category' in training_sales_data.columns else []
        }
        
        return jsonify({
            'message': 'Training data loaded successfully',
            'brand_config': brand_config,
            'has_inventory_data': training_inventory_data is not None
        })
        
    except Exception as e:
        return jsonify({'error': f'Training data loading failed: {str(e)}'}), 500

@app.route('/api/load-prediction-data', methods=['POST'])
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
        prediction_products_data = forecaster.load_prediction_data(products_filepath)
        
        print("✅ Prediction data loaded successfully!")
        
        # Extract categories for frontend
        categories = prediction_products_data['Category'].unique().tolist() if 'Category' in prediction_products_data.columns else []
        
        print("🎉 Returning success response...")
        return jsonify({
            'message': 'Prediction data loaded successfully',
            'products_count': len(prediction_products_data),
            'categories': categories,
            'sample_products': extract_sample_products(prediction_products_data)
        })
        
    except Exception as e:
        print(f"💥 ERROR in load_prediction_data: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Prediction data loading failed: {str(e)}'}), 500

@app.route('/api/generate-predictions', methods=['POST'])
def generate_predictions():
    """Generate demand predictions for new products"""
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
        
        # Create prediction features
        print(f"Creating prediction features for {len(filtered_products)} products...")
        prediction_features = forecaster.create_prediction_features(filtered_products)
        
        # Generate predictions
        print("Generating predictions...")
        predictions = forecaster.predict_demand(prediction_features)
        
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
                    'color_code': row.get('Color Code', '')
                }
            }
            predictions_list.append(prediction_dict)
        
        return jsonify({
            'message': f'Predictions generated for {len(predictions_list)} products',
            'predictions': predictions_list,
            'summary': {
                'total_products': len(predictions_list),
                'total_predicted_demand': int(predictions['predicted_demand'].sum()),
                'avg_confidence': int(predictions['confidence_score'].mean()),
                'risk_distribution': predictions['risk_level'].value_counts().to_dict()
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

@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Train the forecasting model on historical data"""
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
        if train_end_date:
            train_end_date = pd.to_datetime(train_end_date)
        
        # Create training features
        print("Creating training features...")
        training_features = forecaster.create_training_features(
            training_sales_data, 
            training_inventory_data, 
            train_end_date
        )
        
        # Train model
        print("Training model...")
        model_params = data.get('model_params', {})
        target_start_date = data.get('target_start_date')
        target_end_date = data.get('target_end_date')
        
        if target_start_date:
            target_start_date = pd.to_datetime(target_start_date)
        if target_end_date:
            target_end_date = pd.to_datetime(target_end_date)
        
        trained_model = forecaster.train_model(
            training_features,
            target_start_date,
            target_end_date,
            model_params
        )
        
        # Get feature importance
        feature_importance = forecaster.get_feature_importance()
        
        print("✅ Model trained successfully!")
        
        return jsonify({
            'message': 'Model trained successfully',
            'training_samples': len(training_features),
            'feature_count': len(forecaster.feature_columns),
            'model_type': forecaster.model_type,
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else []
        })
        
    except Exception as e:
        print(f"💥 ERROR in train_model: {str(e)}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        print(f"📚 Full traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'Model training failed: {str(e)}'}), 500

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
        
        # Get product features
        product_features = product_data.iloc[0].to_dict()
        
        # Generate single product forecast
        forecast = forecaster.quick_forecast_single_product(product_code, product_features)
        
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
        product_features = product_data.iloc[0].to_dict()
        forecast = forecaster.quick_forecast_single_product(product_code, product_features)
        
        # Generate distribution based on product attributes
        distribution = generate_new_product_distribution(product_data.iloc[0], forecast)
        
        return jsonify({
            'product_code': product_code,
            'forecast': forecast,
            'distribution': distribution,
            'reasoning': f"AI-optimized distribution for new product based on category and attribute patterns"
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
        
        # Create POS export format
        pos_data = {
            'export_date': datetime.now().isoformat(),
            'export_type': 'NEW_PRODUCT_DEMAND_FORECAST',
            'shop': '(S-12) Packages Mall Lahore',
            'total_products': len(selected_predictions),
            'total_predicted_demand': sum(pred['predicted_demand'] for pred in selected_predictions),
            'model_version': f'Sequential_Forecasting_v2.0_{forecaster.model_type}',
            'brand_features': forecaster.brand_features,
            'predictions': selected_predictions
        }
        
        return jsonify({
            'message': 'Export ready',
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
        
        feature_importance = forecaster.get_feature_importance()
        
        return jsonify({
            'model_type': forecaster.model_type,
            'feature_count': len(forecaster.feature_columns),
            'brand_features': forecaster.brand_features,
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else [],
            'training_config': brand_config
        })
        
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
                    'season': row.get('Season', '')
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
                    'season': row.get('Season', '')
                },
                'historicalSales': 0,  # New products have no sales history
                'totalQuantity': 0     # Will be determined by prediction
            }
            products.append(product)
        
        return products
        
    except Exception as e:
        print(f"Product extraction error: {e}")
        return []

def generate_new_product_distribution(product_row, forecast):
    """Generate distribution for new product based on attributes"""
    try:
        # For new products, create a single variation based on their attributes
        distribution = [{
            'shopId': 1,
            'productCode': product_row['Product Code'],
            'variation': {
                'size': product_row.get('Size Name', 'OS'),
                'color': product_row.get('Color Name', 'Default'),
                'sizeCode': product_row.get('Size Code', 'OS'),
                'colorCode': product_row.get('Color Code', 'DEF')
            },
            'allocatedQuantity': forecast['predictedDemand'],
            'reasoning': f"Initial allocation for new product. {forecast['confidence']}% confidence, {forecast['riskLevel']} risk."
        }]
        
        return distribution
        
    except Exception as e:
        print(f"Distribution error: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)