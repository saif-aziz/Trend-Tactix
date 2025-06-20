# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# import json
# from datetime import datetime, timedelta
# import logging

# # Import your forecasting system
# from business_forecasting_system import BusinessForecastingSystem

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React frontend

# # Enable debug logging
# logging.basicConfig(level=logging.DEBUG)

# # Initialize the forecasting system
# forecaster = BusinessForecastingSystem()

# # Global variables to store data
# enhanced_data = None
# forecasts_cache = {}
# shop_allocations = {}

# # Add debugging middleware
# @app.before_request
# def log_request_info():
#     print(f"🔍 Request: {request.method} {request.path}")
#     if request.is_json:
#         print(f"   Body: {request.get_json()}")

# @app.after_request
# def log_response_info(response):
#     print(f"📤 Response: {response.status_code}")
#     return response

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     print("✅ Health check requested")
#     return jsonify({
#         'status': 'healthy',
#         'timestamp': datetime.now().isoformat(),
#         'model_status': 'READY' if enhanced_data is not None else 'NOT_READY'
#     })

# @app.route('/api/load-data', methods=['POST'])
# def load_historical_data():
#     """Load historical sales data for training"""
#     global enhanced_data
    
#     try:
#         print("🔄 Loading historical data...")
        
#         # Always use working sample data for now
#         enhanced_data = create_working_sample_data()
        
#         # Train the forecasting models
#         print("🧠 Training forecasting models...")
#         category_analysis = forecaster.analyze_business_patterns(enhanced_data)
#         risk_analysis = forecaster.create_risk_segments(enhanced_data)
        
#         print(f"✅ Data loaded successfully: {len(enhanced_data)} records")
        
#         return jsonify({
#             'status': 'success',
#             'message': 'Data loaded and models trained successfully',
#             'data_summary': {
#                 'total_records': len(enhanced_data),
#                 'years': sorted(enhanced_data['Year'].unique().tolist()),
#                 'products': enhanced_data['Product Code'].nunique(),
#                 'categories': enhanced_data['Category'].nunique()
#             },
#             'model_status': 'TRAINED'
#         })
        
#     except Exception as e:
#         print(f"❌ Error loading data: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/products', methods=['GET'])
# def get_products():
#     """Get products for distribution with forecasts"""
#     global enhanced_data, forecasts_cache
    
#     try:
#         print("📦 Getting products...")
#         if enhanced_data is None:
#             print("⚠️ No data loaded - auto-loading sample data")
#             enhanced_data = create_working_sample_data()
#             forecaster.analyze_business_patterns(enhanced_data)
#             forecaster.create_risk_segments(enhanced_data)
        
#         # Generate forecasts for 2025
#         print("🔮 Generating forecasts...")
#         forecasts_2025 = forecaster.generate_business_forecasts(enhanced_data, 2025)
#         forecasts_cache = forecasts_2025
        
#         # Convert to frontend format
#         products = []
        
#         # Get latest year data
#         latest_year = enhanced_data['Year'].max()
#         latest_products = enhanced_data[enhanced_data['Year'] == latest_year]
        
#         # Take top 20 products
#         sample_products = latest_products.head(20)
        
#         for idx, (_, row) in enumerate(sample_products.iterrows()):
#             product_code = row['Product Code']
#             forecast_data = forecasts_2025.get(product_code, {})
            
#             # Generate product details
#             product_details = {
#                 'id': idx + 1,
#                 'productCode': product_code,
#                 'name': row.get('Product Name_x', f'Product {idx+1}'),
#                 'category': map_category(row['Category']),
#                 'subCategory': map_subcategory(row['Category']),
#                 'season': 'SS25',
#                 'attributes': {
#                     'gender': map_gender(row.get('Gender', 'Male')),
#                     'age_group': '18-35',
#                     'price_range': np.random.choice(['MID', 'PREMIUM']),
#                     'style': np.random.choice(['CASUAL', 'TRENDY']),
#                     'color_family': row.get('Color Name', 'BLUE').upper(),
#                     'material': 'COTTON'
#                 },
#                 'totalQuantity': np.random.randint(500, 2000),
#                 'variations': create_variations()
#             }
            
#             # Add forecast information
#             product_details['predictedPerformance'] = {
#                 'expectedSellThrough': min(95, max(60, forecast_data.get('forecast', 50) * 2)),
#                 'riskLevel': forecast_data.get('product_risk', 'MEDIUM'),
#                 'confidence': min(95, max(70, 85 + np.random.randint(-10, 10)))
#             }
            
#             products.append(product_details)
        
#         print(f"✅ Returning {len(products)} products")
        
#         return jsonify({
#             'products': products,
#             'total_count': len(products),
#             'season': 'SS25',
#             'forecast_year': 2025
#         })
        
#     except Exception as e:
#         print(f"❌ Error getting products: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/shops', methods=['GET'])
# def get_shops():
#     """Get shop information"""
#     try:
#         print("🏪 Getting shops data...")
        
#         shops = [
#             {
#                 'id': 'shop-main',
#                 'name': 'Packages Mall Lahore',
#                 'location': 'Lahore',
#                 'tier': 'FLAGSHIP',
#                 'characteristics': {
#                     'footfall': 'HIGH',
#                     'demographics': ['18-35', '25-40'],
#                     'preferences': ['TRENDY', 'PREMIUM'],
#                     'seasons_strong': ['SUMMER', 'WINTER']
#                 },
#                 'performance': {
#                     'sellThroughRate': 78,
#                     'averageBasket': 4500,
#                     'returnRate': 12
#                 }
#             },
#             {
#                 'id': 'shop-expansion1',
#                 'name': 'DHA Mall Lahore',
#                 'location': 'DHA Lahore',
#                 'tier': 'REGULAR',
#                 'characteristics': {
#                     'footfall': 'MEDIUM',
#                     'demographics': ['20-45'],
#                     'preferences': ['CASUAL', 'MID'],
#                     'seasons_strong': ['WINTER', 'SPRING']
#                 },
#                 'performance': {
#                     'sellThroughRate': 72,
#                     'averageBasket': 3200,
#                     'returnRate': 8
#                 }
#             },
#             {
#                 'id': 'shop-expansion2',
#                 'name': 'Gulberg Outlet',
#                 'location': 'Gulberg Lahore',
#                 'tier': 'OUTLET',
#                 'characteristics': {
#                     'footfall': 'MEDIUM',
#                     'demographics': ['25-45'],
#                     'preferences': ['VALUE', 'CLASSIC'],
#                     'seasons_strong': ['SPRING', 'AUTUMN']
#                 },
#                 'performance': {
#                     'sellThroughRate': 65,
#                     'averageBasket': 2800,
#                     'returnRate': 5
#                 }
#             }
#         ]
        
#         print(f"✅ Returning {len(shops)} shops")
#         return jsonify({'shops': shops})
        
#     except Exception as e:
#         print(f"❌ Error getting shops: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/generate-distribution', methods=['POST'])
# def generate_distribution():
#     """Generate AI-powered stock distribution"""
#     global forecasts_cache, shop_allocations
    
#     try:
#         data = request.get_json()
#         product_id = data.get('product_id')
        
#         print(f"🎲 Generating distribution for product: {product_id}")
        
#         if not product_id:
#             return jsonify({'error': 'Product ID is required'}), 400
        
#         # Get product forecast or create default
#         product_forecast = forecasts_cache.get(product_id, {
#             'forecast': 100,
#             'confidence': 'MEDIUM',
#             'category': 'TOPS',
#             'category_status': '🔄 Stable Core',
#             'product_risk': 'MEDIUM'
#         })
        
#         # Generate distribution
#         distribution = create_smart_distribution(product_id, product_forecast)
#         shop_allocations[product_id] = distribution
        
#         print(f"✅ Generated distribution with {len(distribution)} allocations")
        
#         return jsonify({
#             'status': 'success',
#             'distribution': distribution,
#             'reasoning': {
#                 'overall_strategy': 'Optimized distribution for multi-shop expansion',
#                 'confidence_level': 'HIGH',
#                 'key_factors': [
#                     'Historical performance at main shop',
#                     'Expansion shop demographics',
#                     'Category preferences by location'
#                 ]
#             }
#         })
        
#     except Exception as e:
#         print(f"❌ Error generating distribution: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/distribution/<product_id>', methods=['GET'])
# def get_distribution(product_id):
#     """Get existing distribution for a product"""
#     try:
#         print(f"📋 Getting distribution for product: {product_id}")
#         if product_id in shop_allocations:
#             return jsonify({
#                 'distribution': shop_allocations[product_id],
#                 'status': 'COMPLETE'
#             })
#         else:
#             return jsonify({
#                 'distribution': None,
#                 'status': 'PENDING'
#             })
#     except Exception as e:
#         print(f"❌ Error getting distribution: {e}")
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/export-distribution', methods=['POST'])
# def export_distribution():
#     """Export distribution data for POS systems"""
#     try:
#         data = request.get_json()
#         product_id = data.get('product_id')
        
#         print(f"📤 Exporting distribution for product: {product_id}")
        
#         if product_id not in shop_allocations:
#             return jsonify({'error': 'No distribution found for this product'}), 404
        
#         # Convert to POS format
#         pos_data = []
#         for item in shop_allocations[product_id]:
#             pos_data.append({
#                 'shop_id': item['shopId'],
#                 'product_code': product_id,
#                 'size': item['variation']['size'],
#                 'color': item['variation']['color'],
#                 'quantity': item['allocatedQuantity'],
#                 'allocation_date': datetime.now().isoformat()
#             })
        
#         return jsonify({
#             'status': 'success',
#             'pos_data': pos_data,
#             'export_timestamp': datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         print(f"❌ Error exporting distribution: {e}")
#         return jsonify({'error': str(e)}), 500

# # Catch-all route for debugging 404s
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def catch_all(path):
#     print(f"🚫 404 - Route not found: {request.method} /{path}")
#     return jsonify({
#         'error': f'Route not found: {request.method} /{path}',
#         'available_routes': [
#             'GET /api/health',
#             'POST /api/load-data', 
#             'GET /api/products',
#             'GET /api/shops',
#             'POST /api/generate-distribution',
#             'GET /api/distribution/<product_id>',
#             'POST /api/export-distribution'
#         ]
#     }), 404

# # Helper Functions
# def create_working_sample_data():
#     """Create sample data that works with your business forecasting system"""
#     print("📊 Creating working sample data...")
#     sample_data = []
    
#     # Use your actual categories
#     categories = ['Boys Knit Top', 'Boys Woven', 'Girls Knit Top', 'Boys Pant', 
#                  'Girls Woven Top', 'Boys Polo', 'Girls Legging', 'Girls Eastern Wear']
    
#     for i in range(150):  # More products for better analysis
#         for year in [2022, 2023, 2024]:
#             sample_data.append({
#                 'Product Code': f'PRD{i:03d}',
#                 'Product Name_x': f'Product {i}',
#                 'Category': np.random.choice(categories),
#                 'Gender': np.random.choice(['Male', 'Female']),
#                 'Season': 'Flat Price Code',
#                 'Size Name': np.random.choice(['S', 'M', 'L', 'XL']),
#                 'Color Name': np.random.choice(['Blue', 'Red', 'Black', 'White']),
#                 'Shop': '(S-12) Packages Mall Lahore',
#                 'Year': year,
#                 'quantity_sold': max(1, int(np.random.normal(25, 15)))  # More realistic sales
#             })
    
#     df = pd.DataFrame(sample_data)
#     print(f"✅ Created sample data: {len(df)} records")
#     return df

# def map_category(category):
#     """Map your categories to frontend categories"""
#     mapping = {
#         'Boys Knit Top': 'TOPS',
#         'Boys Woven': 'TOPS',
#         'Girls Knit Top': 'TOPS',
#         'Boys Pant': 'BOTTOMS',
#         'Girls Woven Top': 'TOPS',
#         'Boys Polo': 'TOPS',
#         'Girls Legging': 'BOTTOMS',
#         'Girls Eastern Wear': 'OUTERWEAR'
#     }
#     return mapping.get(category, 'TOPS')

# def map_subcategory(category):
#     """Map to subcategories"""
#     mapping = {
#         'Boys Knit Top': 'T-SHIRT',
#         'Boys Woven': 'SHIRT',
#         'Girls Knit Top': 'T-SHIRT',
#         'Boys Pant': 'TROUSER',
#         'Girls Woven Top': 'BLOUSE',
#         'Boys Polo': 'POLO',
#         'Girls Legging': 'LEGGING',
#         'Girls Eastern Wear': 'DRESS'
#     }
#     return mapping.get(category, 'T-SHIRT')

# def map_gender(gender):
#     """Map gender for frontend"""
#     if gender == 'Female':
#         return 'WOMEN'
#     elif gender == 'Male':
#         return 'UNISEX'
#     else:
#         return 'UNISEX'

# def create_variations():
#     """Create product variations"""
#     variations = []
#     sizes = ['S', 'M', 'L']
#     colors = ['Blue', 'Black']
    
#     for size in sizes:
#         for color in colors:
#             variations.append({
#                 'size': size,
#                 'color': color,
#                 'quantity': np.random.randint(100, 300)
#             })
    
#     return variations

# def create_smart_distribution(product_id, forecast_data):
#     """Create realistic distribution across shops"""
#     # Base allocation percentages
#     allocations = {
#         'shop-main': 0.50,      # 50% to main performing shop
#         'shop-expansion1': 0.30, # 30% to regular expansion
#         'shop-expansion2': 0.20  # 20% to outlet
#     }
    
#     # Adjust based on product characteristics
#     category = forecast_data.get('category', 'TOPS')
#     if category == 'BOTTOMS':
#         allocations['shop-main'] = 0.45
#         allocations['shop-expansion1'] = 0.35
#         allocations['shop-expansion2'] = 0.20
    
#     # Create distribution
#     distributions = []
#     variations = create_variations()
    
#     for variation in variations:
#         for shop_id, percentage in allocations.items():
#             allocated_qty = max(1, int(variation['quantity'] * percentage))
            
#             distributions.append({
#                 'shopId': shop_id,
#                 'variation': variation,
#                 'allocatedQuantity': allocated_qty,
#                 'confidence': np.random.randint(80, 95),
#                 'reasoning': f"Optimized for {shop_id} performance and demographics"
#             })
    
#     return distributions

# if __name__ == '__main__':
#     print("🚀 Starting AI Stock Distribution Backend...")
#     print("🔧 Debug mode enabled - all requests will be logged")
    
#     # Initialize with sample data
#     print("📊 Initializing with sample data...")
#     enhanced_data = create_working_sample_data()
    
#     # Train the model
#     print("🧠 Training initial models...")
#     forecaster.analyze_business_patterns(enhanced_data)
#     forecaster.create_risk_segments(enhanced_data)
    
#     print("✅ Backend ready! Starting Flask server...")
#     print("📡 Available endpoints:")
#     print("   GET  /api/health")
#     print("   POST /api/load-data")
#     print("   GET  /api/products")
#     print("   GET  /api/shops")
#     print("   POST /api/generate-distribution")
#     print()
    
#     app.run(debug=True, host='0.0.0.0', port=5000)