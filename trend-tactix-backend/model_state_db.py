# Model State Database - Persistent storage for trained model state
# This allows the system to remember trained models across server restarts

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

DATABASE_PATH = 'data/model_state.db'

def get_db_connection():
    """Get a database connection"""
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table for storing model training state
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE DEFAULT 'default',
            status TEXT DEFAULT 'NOT_TRAINED',
            model_version TEXT,
            model_path TEXT,
            training_data_path TEXT,
            prediction_data_path TEXT,
            brand_config TEXT,
            ensemble_weights TEXT,
            feature_columns TEXT,
            validation_results TEXT,
            training_samples INTEGER,
            feature_count INTEGER,
            models_trained TEXT,
            prediction_horizon INTEGER DEFAULT 365,
            last_trained_at TIMESTAMP,
            last_validated_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table for storing training data metadata (not the actual data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_data_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT DEFAULT 'default',
            sales_records INTEGER,
            inventory_records INTEGER,
            date_range_start TEXT,
            date_range_end TEXT,
            unique_skus INTEGER,
            categories TEXT,
            available_features TEXT,
            file_hash TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_name) REFERENCES model_state(model_name)
        )
    ''')
    
    # Table for storing prediction data metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_data_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT DEFAULT 'default',
            products_count INTEGER,
            categories TEXT,
            sample_products TEXT,
            common_features TEXT,
            missing_features TEXT,
            file_hash TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_name) REFERENCES model_state(model_name)
        )
    ''')
    
    # Table for prediction period configuration
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_period (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT DEFAULT 'default',
            start_date TEXT,
            end_date TEXT,
            prediction_type TEXT,
            total_days INTEGER,
            historical_analysis TEXT,
            set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (model_name) REFERENCES model_state(model_name)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

def save_model_state(
    model_name: str = 'default',
    status: str = 'NOT_TRAINED',
    model_version: str = None,
    model_path: str = None,
    training_data_path: str = None,
    prediction_data_path: str = None,
    brand_config: Dict = None,
    ensemble_weights: Dict = None,
    feature_columns: list = None,
    validation_results: list = None,
    training_samples: int = None,
    feature_count: int = None,
    models_trained: list = None,
    prediction_horizon: int = 365
):
    """Save or update the model state"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if model exists
    cursor.execute('SELECT id FROM model_state WHERE model_name = ?', (model_name,))
    exists = cursor.fetchone()
    
    now = datetime.now().isoformat()
    
    if exists:
        # Update existing record
        cursor.execute('''
            UPDATE model_state SET
                status = COALESCE(?, status),
                model_version = COALESCE(?, model_version),
                model_path = COALESCE(?, model_path),
                training_data_path = COALESCE(?, training_data_path),
                prediction_data_path = COALESCE(?, prediction_data_path),
                brand_config = COALESCE(?, brand_config),
                ensemble_weights = COALESCE(?, ensemble_weights),
                feature_columns = COALESCE(?, feature_columns),
                validation_results = COALESCE(?, validation_results),
                training_samples = COALESCE(?, training_samples),
                feature_count = COALESCE(?, feature_count),
                models_trained = COALESCE(?, models_trained),
                prediction_horizon = COALESCE(?, prediction_horizon),
                last_trained_at = CASE WHEN ? = 'TRAINED' THEN ? ELSE last_trained_at END,
                last_validated_at = CASE WHEN ? = 'VALIDATED' THEN ? ELSE last_validated_at END,
                updated_at = ?
            WHERE model_name = ?
        ''', (
            status,
            model_version,
            model_path,
            training_data_path,
            prediction_data_path,
            json.dumps(brand_config) if brand_config else None,
            json.dumps(ensemble_weights) if ensemble_weights else None,
            json.dumps(feature_columns) if feature_columns else None,
            json.dumps(validation_results) if validation_results else None,
            training_samples,
            feature_count,
            json.dumps(models_trained) if models_trained else None,
            prediction_horizon,
            status, now,
            status, now,
            now,
            model_name
        ))
    else:
        # Insert new record
        cursor.execute('''
            INSERT INTO model_state (
                model_name, status, model_version, model_path,
                training_data_path, prediction_data_path,
                brand_config, ensemble_weights, feature_columns,
                validation_results, training_samples, feature_count,
                models_trained, prediction_horizon,
                last_trained_at, last_validated_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name,
            status,
            model_version,
            model_path,
            training_data_path,
            prediction_data_path,
            json.dumps(brand_config) if brand_config else None,
            json.dumps(ensemble_weights) if ensemble_weights else None,
            json.dumps(feature_columns) if feature_columns else None,
            json.dumps(validation_results) if validation_results else None,
            training_samples,
            feature_count,
            json.dumps(models_trained) if models_trained else None,
            prediction_horizon,
            now if status == 'TRAINED' else None,
            now if status == 'VALIDATED' else None,
            now
        ))
    
    conn.commit()
    conn.close()
    print(f"✅ Model state saved: {model_name} - {status}")

def get_model_state(model_name: str = 'default') -> Optional[Dict[str, Any]]:
    """Get the current model state"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM model_state WHERE model_name = ?', (model_name,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        state = dict(row)
        # Parse JSON fields
        for json_field in ['brand_config', 'ensemble_weights', 'feature_columns', 
                          'validation_results', 'models_trained']:
            if state.get(json_field):
                try:
                    state[json_field] = json.loads(state[json_field])
                except:
                    pass
        return state
    
    return None

def save_training_data_info(
    model_name: str = 'default',
    sales_records: int = 0,
    inventory_records: int = 0,
    date_range_start: str = None,
    date_range_end: str = None,
    unique_skus: int = 0,
    categories: list = None,
    available_features: list = None,
    file_hash: str = None
):
    """Save training data metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete existing record for this model
    cursor.execute('DELETE FROM training_data_info WHERE model_name = ?', (model_name,))
    
    # Insert new record
    cursor.execute('''
        INSERT INTO training_data_info (
            model_name, sales_records, inventory_records,
            date_range_start, date_range_end, unique_skus,
            categories, available_features, file_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_name,
        sales_records,
        inventory_records,
        date_range_start,
        date_range_end,
        unique_skus,
        json.dumps(categories) if categories else None,
        json.dumps(available_features) if available_features else None,
        file_hash
    ))
    
    conn.commit()
    conn.close()

def get_training_data_info(model_name: str = 'default') -> Optional[Dict[str, Any]]:
    """Get training data metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM training_data_info WHERE model_name = ? ORDER BY uploaded_at DESC LIMIT 1', (model_name,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        info = dict(row)
        for json_field in ['categories', 'available_features']:
            if info.get(json_field):
                try:
                    info[json_field] = json.loads(info[json_field])
                except:
                    pass
        return info
    
    return None

def save_prediction_data_info(
    model_name: str = 'default',
    products_count: int = 0,
    categories: list = None,
    sample_products: list = None,
    common_features: list = None,
    missing_features: list = None,
    file_hash: str = None
):
    """Save prediction data metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete existing record for this model
    cursor.execute('DELETE FROM prediction_data_info WHERE model_name = ?', (model_name,))
    
    # Insert new record
    cursor.execute('''
        INSERT INTO prediction_data_info (
            model_name, products_count, categories,
            sample_products, common_features, missing_features, file_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        model_name,
        products_count,
        json.dumps(categories) if categories else None,
        json.dumps(sample_products) if sample_products else None,
        json.dumps(common_features) if common_features else None,
        json.dumps(missing_features) if missing_features else None,
        file_hash
    ))
    
    conn.commit()
    conn.close()

def get_prediction_data_info(model_name: str = 'default') -> Optional[Dict[str, Any]]:
    """Get prediction data metadata"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM prediction_data_info WHERE model_name = ? ORDER BY uploaded_at DESC LIMIT 1', (model_name,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        info = dict(row)
        for json_field in ['categories', 'sample_products', 'common_features', 'missing_features']:
            if info.get(json_field):
                try:
                    info[json_field] = json.loads(info[json_field])
                except:
                    pass
        return info
    
    return None

def save_prediction_period(
    model_name: str = 'default',
    start_date: str = None,
    end_date: str = None,
    prediction_type: str = 'custom',
    total_days: int = 0,
    historical_analysis: Dict = None
):
    """Save prediction period configuration"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete existing record for this model
    cursor.execute('DELETE FROM prediction_period WHERE model_name = ?', (model_name,))
    
    # Insert new record
    cursor.execute('''
        INSERT INTO prediction_period (
            model_name, start_date, end_date,
            prediction_type, total_days, historical_analysis
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        model_name,
        start_date,
        end_date,
        prediction_type,
        total_days,
        json.dumps(historical_analysis) if historical_analysis else None
    ))
    
    conn.commit()
    conn.close()

def get_prediction_period(model_name: str = 'default') -> Optional[Dict[str, Any]]:
    """Get prediction period configuration"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM prediction_period WHERE model_name = ? ORDER BY set_at DESC LIMIT 1', (model_name,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        info = dict(row)
        if info.get('historical_analysis'):
            try:
                info['historical_analysis'] = json.loads(info['historical_analysis'])
            except:
                pass
        return info
    
    return None

def reset_model_state(model_name: str = 'default'):
    """Reset the model state (for retraining)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM model_state WHERE model_name = ?', (model_name,))
    cursor.execute('DELETE FROM training_data_info WHERE model_name = ?', (model_name,))
    cursor.execute('DELETE FROM prediction_data_info WHERE model_name = ?', (model_name,))
    cursor.execute('DELETE FROM prediction_period WHERE model_name = ?', (model_name,))
    
    conn.commit()
    conn.close()
    print(f"✅ Model state reset: {model_name}")

def get_all_models() -> list:
    """Get all saved models"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT model_name, status, model_version, last_trained_at, updated_at FROM model_state ORDER BY updated_at DESC')
    rows = cursor.fetchall()
    
    conn.close()
    
    return [dict(row) for row in rows]

# Initialize database on module import
init_database()

