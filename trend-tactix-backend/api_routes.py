from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import traceback

# Import our forecasting logic
from sequential_inventory_forecaster import SequentialInventoryForecaster

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global variables
forecaster = None
sales_data = None
products_data = None
DATA_FOLDER = 'data'

# Ensure data folder exists
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = 'READY' if sales_data is not None else 'NOT_READY'
        return jsonify({
            'status': 'healthy',
            'model_status': model_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load and process sales data"""
    global forecaster, sales_data, products_data
    
    try:
        # Check if file uploaded or use existing
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Save uploaded file
                filepath = os.path.join(DATA_FOLDER, 'uploaded_sales_data.csv')
                file.save(filepath)
            else:
                return jsonify({'error': 'No file provided'}), 400
        else:
            # Use existing dataset
            filepath = os.path.join(DATA_FOLDER, 'SALEDATAROLLOVER_20222025_output.csv')
            if not os.path.exists(filepath):
                return jsonify({'error': 'Dataset not found. Please upload a file.'}), 400

        # Initialize forecaster
        forecaster = SequentialInventoryForecaster(model_type='xgboost')
        
        # Load and prepare data
        sales_data = forecaster.load_and_prepare_data(filepath)
        
        # Extract unique products
        products_data = extract_products_from_sales(sales_data)
        
        return jsonify({
            'message': 'Data loaded successfully',
            'records_count': len(sales_data),
            'products_count': len(products_data),
            'date_range': {
                'start': sales_data['Sale Date'].min().isoformat(),
                'end': sales_data['Sale Date'].max().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Data loading failed: {str(e)}'}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products with basic info"""
    global products_data
    
    try:
        if products_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        return jsonify({
            'products': products_data,
            'count': len(products_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/shops', methods=['GET'])
def get_shops():
    """Get shop information"""
    try:
        # Since you have single shop - Packages Mall Lahore
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
    """Generate forecast for a specific product"""
    global forecaster, sales_data
    
    try:
        if not sales_data is not None:
            return jsonify({'error': 'No data loaded'}), 400
            
        data = request.get_json()
        product_code = data.get('product_code')
        
        if not product_code:
            return jsonify({'error': 'Product code required'}), 400
        
        # Generate forecast using our sequential logic
        forecast = generate_product_forecast(sales_data, product_code)
        
        return jsonify({
            'product_code': product_code,
            'forecast': forecast
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-distribution', methods=['POST'])
def generate_distribution():
    """Generate smart distribution for a product"""
    global sales_data
    
    try:
        if sales_data is None:
            return jsonify({'error': 'No data loaded'}), 400
            
        data = request.get_json()
        product_code = data.get('product_id')  # Frontend sends product_id
        
        if not product_code:
            return jsonify({'error': 'Product code required'}), 400
        
        # Generate forecast first
        forecast = generate_product_forecast(sales_data, product_code)
        
        # Generate distribution
        distribution = generate_smart_distribution(sales_data, product_code, forecast)
        
        return jsonify({
            'product_code': product_code,
            'forecast': forecast,
            'distribution': distribution,
            'reasoning': f"AI-optimized distribution based on {len(distribution)} variations"
        })
        
    except Exception as e:
        print(f"Distribution error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/distribution/<product_code>', methods=['GET'])
def get_distribution(product_code):
    """Get existing distribution for a product"""
    try:
        # In a real app, you'd store distributions in a database
        # For now, regenerate on demand
        return jsonify({
            'distribution': None,
            'status': 'NOT_GENERATED'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-distribution', methods=['POST'])
def export_distribution():
    """Export distribution in POS format"""
    try:
        data = request.get_json()
        product_code = data.get('product_id')
        
        if not product_code:
            return jsonify({'error': 'Product code required'}), 400
        
        # Generate fresh distribution
        forecast = generate_product_forecast(sales_data, product_code)
        distribution = generate_smart_distribution(sales_data, product_code, forecast)
        
        # Create POS format
        pos_data = {
            'export_date': datetime.now().isoformat(),
            'product_code': product_code,
            'shop': '(S-12) Packages Mall Lahore',
            'forecast_summary': forecast,
            'allocations': distribution,
            'total_units': sum(d['allocatedQuantity'] for d in distribution),
            'model_version': 'Sequential_Forecasting_v1.0'
        }
        
        return jsonify({
            'message': 'Export ready',
            'pos_data': pos_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def extract_products_from_sales(sales_df):
    """Extract unique products from sales data"""
    try:
        products = []
        
        # Group by product code to get unique products
        product_groups = sales_df.groupby('Product Code').agg({
            'Product Name': 'first',
            'Category': 'first',
            'Gender': 'first',
            'Season': 'first',
            'Size Name': 'first',
            'Size Code': 'first',
            'Color Name': 'first',
            'Color Code': 'first',
            'Sale Date': 'count'  # Count sales as historical sales
        }).reset_index()
        
        for _, row in product_groups.iterrows():
            product = {
                'id': row['Product Code'],
                'productCode': row['Product Code'],
                'name': row['Product Name'],
                'category': row['Category'],
                'gender': row['Gender'],
                'season': row['Season'],
                'historicalSales': int(row['Sale Date']),
                'attributes': {
                    'size': row['Size Name'],
                    'color': row['Color Name'],
                    'gender': row['Gender']
                },
                'totalQuantity': max(50, int(row['Sale Date']) * 2)  # Estimate
            }
            products.append(product)
        
        return products
        
    except Exception as e:
        print(f"Product extraction error: {e}")
        return []

def generate_product_forecast(sales_df, product_code):
    """Generate forecast for a specific product"""
    try:
        product_sales = sales_df[sales_df['Product Code'] == product_code]
        
        if len(product_sales) == 0:
            return {
                'predictedDemand': 0,
                'confidence': 0,
                'riskLevel': 'HIGH',
                'reasoning': 'No historical sales data available'
            }
        
        # Calculate basic metrics
        total_sales = len(product_sales)
        
        # Calculate date range and velocity
        dates = pd.to_datetime(product_sales['Sale Date'])
        date_range_days = (dates.max() - dates.min()).days
        months_span = max(1, date_range_days / 30)
        velocity = total_sales / months_span
        
        # Predict for next 3 months
        predicted_demand = max(1, int(velocity * 3))
        
        # Calculate confidence based on sales consistency
        monthly_sales = product_sales.groupby(dates.dt.to_period('M')).size()
        if len(monthly_sales) > 1:
            cv = monthly_sales.std() / monthly_sales.mean()
            confidence = max(10, min(95, 80 - (cv * 30)))
        else:
            confidence = 50 if total_sales >= 3 else 25
        
        # Risk assessment
        if confidence >= 70 and total_sales >= 10:
            risk_level = 'LOW'
        elif confidence >= 40 and total_sales >= 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        return {
            'predictedDemand': predicted_demand,
            'confidence': int(confidence),
            'riskLevel': risk_level,
            'reasoning': f'Based on {total_sales} sales over {months_span:.1f} months. Velocity: {velocity:.1f} sales/month'
        }
        
    except Exception as e:
        print(f"Forecast error: {e}")
        return {
            'predictedDemand': 1,
            'confidence': 0,
            'riskLevel': 'HIGH',
            'reasoning': f'Error in calculation: {str(e)}'
        }

def generate_smart_distribution(sales_df, product_code, forecast):
    """Generate smart distribution across variations"""
    try:
        product_sales = sales_df[sales_df['Product Code'] == product_code]
        
        if len(product_sales) == 0:
            return []
        
        # Group by size and color to get variations
        variations = product_sales.groupby(['Size Code', 'Color Code']).agg({
            'Size Name': 'first',
            'Color Name': 'first',
            'Sale Date': 'count'
        }).reset_index()
        
        variations.columns = ['sizeCode', 'colorCode', 'sizeName', 'colorName', 'salesCount']
        
        total_variation_sales = variations['salesCount'].sum()
        predicted_demand = forecast['predictedDemand']
        
        distributions = []
        
        for _, variation in variations.iterrows():
            # Calculate proportional allocation
            if total_variation_sales > 0:
                proportion = variation['salesCount'] / total_variation_sales
            else:
                proportion = 1 / len(variations)
            
            allocated_qty = max(1, int(predicted_demand * proportion))
            
            distribution = {
                'shopId': 1,
                'productCode': product_code,
                'variation': {
                    'size': variation['sizeName'],
                    'color': variation['colorName'],
                    'sizeCode': variation['sizeCode'],
                    'colorCode': variation['colorCode']
                },
                'allocatedQuantity': allocated_qty,
                'reasoning': f'{proportion*100:.1f}% allocation based on {variation["salesCount"]} historical sales'
            }
            distributions.append(distribution)
        
        return distributions
        
    except Exception as e:
        print(f"Distribution error: {e}")
        return []

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)