from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

from advanced_model_optimization import AdvancedOptimizedForecaster

app = Flask(__name__)
CORS(app, 
     origins=['*'],
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

shop_allocations = {}

if __name__ == '__main__':
    print("🚀 Starting AI Stock Distribution Backend...")
    print("🔧 Debug mode enabled - all requests will be logged")
    
    # # Initialize with sample data
    # print("📊 Initializing with sample data...")
    # enhanced_data = create_working_sample_data()
    
    # # Train the model
    # print("🧠 Training initial models...")
    # forecaster.analyze_business_patterns(enhanced_data)
    # forecaster.create_risk_segments(enhanced_data)
    
    print("✅ Backend ready! Starting Flask server...")
    print("📡 Available endpoints:")
    print("   GET  /api/health")
    print("   POST /api/load-data")
    print("   GET  /api/products")
    print("   GET  /api/shops")
    print("   POST /api/generate-distribution")
    print("   GET  /api/distribution/<product_id>")
    print("   POST /api/export-distribution")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)