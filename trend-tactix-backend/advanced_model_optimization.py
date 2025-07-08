# Advanced Model Optimization System - FIXED VERSION
# This system implements dynamic weight updates, hyperparameter optimization,
# and continuous learning capabilities for better forecasting accuracy

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Additional imports for optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# CHANGE TO:
from optimized_forecaster import OptimizedInventoryForecaster

class AdvancedOptimizedForecaster(OptimizedInventoryForecaster):
    """
    Advanced forecasting system with dynamic weight updates and parameter optimization
    """
    
    def __init__(self, model_type='ensemble', prediction_horizon_days=90):
        super().__init__(model_type, prediction_horizon_days)
        
        # Advanced optimization attributes
        self.ensemble_weights = {}
        self.dynamic_weights = {}
        self.performance_history = []
        self.best_params = {}
        self.learning_rate = 0.1
        self.weight_decay = 0.95
        
        # Model persistence
        self.model_save_path = "models/"
        os.makedirs(self.model_save_path, exist_ok=True)
    
    # ========================
    # DATA PREPROCESSING - FIXED
    # ========================
    
    def _fix_categorical_columns(self, df):
        """Fix categorical columns by converting them to object type first"""
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype.name == 'category':
                df_copy[col] = df_copy[col].astype('object')
        return df_copy
    
    def _prepare_features_for_ml(self, features_df):
        """
        CRITICAL FIX: Properly prepare features for machine learning with consistent feature set
        """
        print(f"📋 Preparing features for ML: {features_df.shape}")
        
        # Make a copy to avoid modifying original
        df = features_df.copy()
        
        # Fix categorical columns
        df = self._fix_categorical_columns(df)
        
        # Identify columns to exclude from feature matrix
        exclude_cols = [
            'Product Code', 'target_demand',
            'first_sale_date', 'last_sale_date',
            'first_inventory_date', 'last_inventory_date'
        ]
        
        # Add any _first columns that aren't encoded
        for col in df.columns:
            if col.endswith('_first') and not col.endswith('_encoded'):
                exclude_cols.append(col)
        
        # Get potential feature columns
        potential_feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Process each column to ensure it's ML-ready
        processed_features = pd.DataFrame()
        processed_features['Product Code'] = df['Product Code']  # Keep for reference
        
        for col in potential_feature_cols:
            if col in df.columns:
                series = df[col]
                
                # Handle different data types
                if series.dtype == 'object' or series.dtype.name == 'category':
                    # Categorical data - encode it
                    print(f"   🔤 Encoding categorical column: {col}")
                    
                    # Create or use existing label encoder
                    encoder_key = f"{col}_encoder"
                    if encoder_key not in self.label_encoders:
                        self.label_encoders[encoder_key] = LabelEncoder()
                        # Fit encoder
                        unique_values = series.dropna().astype(str).unique()
                        if len(unique_values) > 0:
                            self.label_encoders[encoder_key].fit(unique_values)
                    
                    # Transform data
                    try:
                        encoded_values = series.astype(str).map(
                            lambda x: self.label_encoders[encoder_key].transform([x])[0] 
                            if x in self.label_encoders[encoder_key].classes_ else -1
                        )
                        processed_features[f"{col}_encoded"] = encoded_values
                    except Exception as e:
                        print(f"   ⚠️ Warning: Could not encode {col}: {e}")
                        # Create dummy numeric column
                        processed_features[f"{col}_encoded"] = 0
                        
                elif pd.api.types.is_datetime64_any_dtype(series):
                    # DateTime data - convert to numeric
                    print(f"   📅 Converting datetime column: {col}")
                    reference_date = datetime.now()
                    try:
                        processed_features[f"{col}_days"] = (reference_date - pd.to_datetime(series)).dt.days
                    except:
                        processed_features[f"{col}_days"] = 0
                        
                elif pd.api.types.is_numeric_dtype(series):
                    # Numeric data - use as is, but handle NaN
                    print(f"   🔢 Using numeric column: {col}")
                    processed_features[col] = pd.to_numeric(series, errors='coerce').fillna(0)
                    
                else:
                    # Unknown type - try to convert to numeric
                    print(f"   ❓ Converting unknown type column: {col}")
                    try:
                        processed_features[col] = pd.to_numeric(series, errors='coerce').fillna(0)
                    except:
                        processed_features[col] = 0
        
        # CRITICAL FIX: Ensure consistent seasonal features
        expected_seasonal_features = ['season_1_sales', 'season_2_sales', 'season_3_sales', 'season_4_sales']
        for seasonal_feature in expected_seasonal_features:
            if seasonal_feature not in processed_features.columns:
                print(f"   🔧 Adding missing seasonal feature: {seasonal_feature}")
                processed_features[seasonal_feature] = 0
        
        # Remove Product Code from features (keep only for reference)
        feature_columns = [col for col in processed_features.columns if col != 'Product Code']
        X = processed_features[feature_columns].copy()
        
        # CRITICAL FIX: If we have stored feature_columns from training, ensure consistency
        if hasattr(self, 'feature_columns') and self.feature_columns:
            print(f"   🔄 Ensuring consistency with training features...")
            training_features = self.feature_columns
            current_features = X.columns.tolist()
            
            # Add missing features with zeros
            for feature in training_features:
                if feature not in current_features:
                    print(f"   ➕ Adding missing training feature: {feature}")
                    X[feature] = 0
            
            # Remove extra features not in training
            extra_features = [f for f in current_features if f not in training_features]
            if extra_features:
                print(f"   ➖ Removing extra features: {extra_features}")
                X = X.drop(columns=extra_features)
            
            # Reorder columns to match training order
            X = X[training_features]
            feature_columns = training_features
        
        # Final safety check - ensure all data is numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                print(f"   🔧 Final conversion for {col}")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Store feature columns for later use (only if not already stored)
        if not hasattr(self, 'feature_columns') or not self.feature_columns:
            self.feature_columns = feature_columns
        
        print(f"✅ Features prepared: {X.shape}, all numeric: {X.dtypes.apply(pd.api.types.is_numeric_dtype).all()}")
        
        return X, feature_columns
    
    # ========================
    # HYPERPARAMETER OPTIMIZATION - FIXED
    # ========================
    
    def optimize_hyperparameters(self, X_train, y_train, method='optuna', n_trials=100):
        """
        Optimize hyperparameters using various methods - FIXED VERSION
        """
        print(f"\n🔧 Starting hyperparameter optimization using {method}")
        
        # FIXED: Proper data validation
        print(f"   📊 Input data shapes: X={X_train.shape}, y={y_train.shape}")
        
        # Check if X_train is properly prepared
        if isinstance(X_train, pd.DataFrame):
            # Check each column individually
            non_numeric_cols = []
            for col in X_train.columns:
                if not pd.api.types.is_numeric_dtype(X_train[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                print(f"⚠️ Warning: Non-numeric columns found: {non_numeric_cols}")
                raise ValueError(f"X_train contains non-numeric columns: {non_numeric_cols}. Use _prepare_features_for_ml first.")
            
            print("✅ All columns are numeric")
        else:
            print(f"   📊 X_train type: {type(X_train)}")
        
        # Check for NaN values
        if isinstance(X_train, pd.DataFrame):
            nan_cols = X_train.columns[X_train.isnull().any()].tolist()
            if nan_cols:
                print(f"⚠️ Warning: NaN values found in columns: {nan_cols}")
                X_train = X_train.fillna(0)
        
        if pd.isna(y_train).any():
            print("⚠️ Warning: NaN values found in target, removing those rows")
            valid_mask = ~pd.isna(y_train)
            X_train = X_train[valid_mask] if isinstance(X_train, pd.DataFrame) else X_train[valid_mask]
            y_train = y_train[valid_mask]
        
        print(f"   ✅ Final data ready: X={X_train.shape}, y={y_train.shape}")
        
        if method == 'optuna' and OPTUNA_AVAILABLE:
            return self._optimize_with_optuna(X_train, y_train, n_trials)
        elif method == 'grid_search':
            return self._optimize_with_grid_search(X_train, y_train)
        elif method == 'random_search':
            return self._optimize_with_random_search(X_train, y_train)
        else:
            print(f"⚠️ {method} not available, using default parameters")
            return self._get_default_parameters()
    
    def _optimize_with_optuna(self, X_train, y_train, n_trials=100):
        """Bayesian optimization using Optuna"""
        import optuna
        
        def objective(trial):
            # Define parameter spaces for each model
            params = {}
            
            # Random Forest parameters
            params['rf'] = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None])
            }
            
            # XGBoost parameters
            if XGBOOST_AVAILABLE:
                params['xgb'] = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 1.0)
                }
            
            # LightGBM parameters
            if LIGHTGBM_AVAILABLE:
                params['lgb'] = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.0, 1.0)
                }
            
            # Cross-validation score
            cv_score = self._cross_validate_params(X_train, y_train, params)
            return cv_score
        
        # Suppress Optuna logging for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        print(f"✅ Best CV score: {study.best_value:.4f}")
        print(f"🎯 Best parameters found after {n_trials} trials")
        
        return self._extract_best_params_from_optuna(study.best_params)
    
    def _cross_validate_params(self, X_train, y_train, params):
        """Cross-validate parameters and return average MAE - FIXED VERSION"""
        try:
            print(f"      🔄 Cross-validating with data shape: X={X_train.shape}, y={y_train.shape}")
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                print(f"         Fold {fold_idx + 1}/3...")
                
                if isinstance(X_train, pd.DataFrame):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                else:
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                
                # Train models with current parameters
                predictions = {}
                
                # Random Forest
                if 'rf' in params:
                    try:
                        rf = RandomForestRegressor(**params['rf'], random_state=42, n_jobs=-1)
                        rf.fit(X_tr, y_tr)
                        predictions['rf'] = rf.predict(X_val)
                    except Exception as e:
                        print(f"         RF training failed in fold {fold_idx + 1}: {e}")
                        continue
                
                # XGBoost
                if 'xgb' in params and XGBOOST_AVAILABLE:
                    try:
                        xgb_params = params['xgb'].copy()
                        xgb_params.update({'random_state': 42, 'verbosity': 0})
                        xgb_model = XGBRegressor(**xgb_params)
                        xgb_model.fit(X_tr, y_tr)
                        predictions['xgb'] = xgb_model.predict(X_val)
                    except Exception as e:
                        print(f"         XGB training failed in fold {fold_idx + 1}: {e}")
                        continue
                
                # LightGBM
                if 'lgb' in params and LIGHTGBM_AVAILABLE:
                    try:
                        lgb_params = params['lgb'].copy()
                        lgb_params.update({'random_state': 42, 'verbosity': -1})
                        lgb_model = LGBMRegressor(**lgb_params)
                        lgb_model.fit(X_tr, y_tr)
                        predictions['lgb'] = lgb_model.predict(X_val)
                    except Exception as e:
                        print(f"         LGB training failed in fold {fold_idx + 1}: {e}")
                        continue
                
                # Ensemble prediction
                if len(predictions) > 1:
                    ensemble_pred = np.mean(list(predictions.values()), axis=0)
                elif len(predictions) == 1:
                    ensemble_pred = list(predictions.values())[0]
                else:
                    print(f"         All models failed in fold {fold_idx + 1}")
                    return 999999  # High error for failed fold
                
                score = mean_absolute_error(y_val, ensemble_pred)
                scores.append(score)
                print(f"         Fold {fold_idx + 1} MAE: {score:.4f}")
            
            final_score = np.mean(scores) if scores else 999999
            print(f"      ✅ CV complete. Average MAE: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            print(f"      ❌ Cross-validation failed: {e}")
            return 999999
    
    def _extract_best_params_from_optuna(self, best_params):
        """Extract best parameters from Optuna study"""
        extracted_params = {}
        
        # Group parameters by model
        for key, value in best_params.items():
            if key.startswith('rf_'):
                if 'rf' not in extracted_params:
                    extracted_params['rf'] = {}
                param_name = key[3:]  # Remove 'rf_' prefix
                extracted_params['rf'][param_name] = value
            elif key.startswith('xgb_'):
                if 'xgb' not in extracted_params:
                    extracted_params['xgb'] = {}
                param_name = key[4:]  # Remove 'xgb_' prefix
                extracted_params['xgb'][param_name] = value
            elif key.startswith('lgb_'):
                if 'lgb' not in extracted_params:
                    extracted_params['lgb'] = {}
                param_name = key[4:]  # Remove 'lgb_' prefix
                extracted_params['lgb'][param_name] = value
        
        return extracted_params
    
    # ========================
    # CATEGORY-SPECIFIC OPTIMIZATION - FIXED
    # ========================
    
    def optimize_category_specific_models(self, features_df):
        """
        Create and optimize models for specific categories
        """
        print("\n🎯 Optimizing category-specific models...")
        
        # Fix categorical columns first
        features_df = self._fix_categorical_columns(features_df)
        
        self.category_models = {}
        category_performance = {}
        
        # Group by category
        if 'Category_first' not in features_df.columns:
            print("⚠️ Warning: Category_first column not found. Cannot optimize category-specific models.")
            return {}
        
        category_groups = features_df.groupby('Category_first')
        
        for category, group_data in category_groups:
            if len(group_data) >= 50:  # Minimum samples for category-specific model
                print(f"\n   📦 Training model for category: {category}")
                
                try:
                    # Prepare features for this category
                    X_cat, feature_cols = self._prepare_features_for_ml(group_data)
                    y_cat = group_data['target_demand']
                    
                    print(f"   📊 Category data shape: {X_cat.shape}")
                    
                    # Optimize parameters for this category
                    best_params = self.optimize_hyperparameters(
                        X_cat, y_cat, method='optuna', n_trials=20  # Reduced trials for speed
                    )
                    
                    # Train category-specific model
                    category_model = self._train_category_model(X_cat, y_cat, best_params)
                    
                    # Validate category model
                    cv_score = self._cross_validate_params(X_cat, y_cat, best_params)
                    
                    self.category_models[category] = {
                        'model': category_model,
                        'params': best_params,
                        'performance': cv_score,
                        'samples': len(group_data),
                        'feature_columns': feature_cols
                    }
                    
                    category_performance[category] = cv_score
                    print(f"   ✅ {category}: {cv_score:.4f} MAE ({len(group_data)} samples)")
                    
                except Exception as e:
                    print(f"   ❌ Error training {category}: {e}")
                    continue
        
        # Save category models
        try:
            self._save_category_models()
        except Exception as e:
            print(f"   ⚠️ Could not save category models: {e}")
        
        return category_performance
    
    def _train_category_model(self, X_cat, y_cat, best_params):
        """Train a single category model with best parameters"""
        category_models = {}
        
        # Train Random Forest
        if 'rf' in best_params:
            try:
                rf = RandomForestRegressor(**best_params['rf'], random_state=42, n_jobs=-1)
                rf.fit(X_cat, y_cat)
                category_models['rf'] = rf
            except Exception as e:
                print(f"      ⚠️ RF training failed: {e}")
        
        # Train XGBoost
        if 'xgb' in best_params and XGBOOST_AVAILABLE:
            try:
                xgb_model = XGBRegressor(**best_params['xgb'], random_state=42, verbosity=0)
                xgb_model.fit(X_cat, y_cat)
                category_models['xgb'] = xgb_model
            except Exception as e:
                print(f"      ⚠️ XGB training failed: {e}")
        
        # Train LightGBM
        if 'lgb' in best_params and LIGHTGBM_AVAILABLE:
            try:
                lgb_model = LGBMRegressor(**best_params['lgb'], random_state=42, verbosity=-1)
                lgb_model.fit(X_cat, y_cat)
                category_models['lgb'] = lgb_model
            except Exception as e:
                print(f"      ⚠️ LGB training failed: {e}")
        
        return category_models
    
    def _save_category_models(self):
        """Save category models to disk"""
        if not hasattr(self, 'category_models'):
            return
        
        category_model_dir = os.path.join(self.model_save_path, "category_models")
        os.makedirs(category_model_dir, exist_ok=True)
        
        for category, model_info in self.category_models.items():
            # Create safe filename
            safe_category = "".join(c for c in category if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_category = safe_category.replace(' ', '_')
            
            category_dir = os.path.join(category_model_dir, safe_category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Save models
            for model_name, model in model_info['model'].items():
                model_path = os.path.join(category_dir, f"{model_name}.joblib")
                joblib.dump(model, model_path)
            
            # Save metadata
            metadata = {
                'category': category,
                'params': model_info['params'],
                'performance': model_info['performance'],
                'samples': model_info['samples'],
                'feature_columns': model_info.get('feature_columns', [])
            }
            
            metadata_path = os.path.join(category_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"   💾 Category models saved to {category_model_dir}")
    
    # ========================
    # DYNAMIC WEIGHT UPDATES
    # ========================
    
    def update_ensemble_weights_dynamically(self, X_val, y_val, model_predictions):
        """
        Update ensemble weights based on recent performance
        Uses exponential moving average for smooth updates
        """
        print("\n⚖️ Updating ensemble weights dynamically...")
        
        # Calculate individual model performance
        model_errors = {}
        for name, predictions in model_predictions.items():
            mae = mean_absolute_error(y_val, predictions)
            model_errors[name] = mae
            print(f"   {name} MAE: {mae:.4f}")
        
        # Calculate inverse error weights (better models get higher weights)
        total_inverse_error = sum(1 / (error + 0.001) for error in model_errors.values())
        new_weights = {
            name: (1 / (error + 0.001)) / total_inverse_error 
            for name, error in model_errors.items()
        }
        
        # Update weights using exponential moving average
        if not self.dynamic_weights:
            self.dynamic_weights = new_weights
        else:
            for name in new_weights:
                if name in self.dynamic_weights:
                    # EMA: new_weight = α * new + (1-α) * old
                    self.dynamic_weights[name] = (
                        self.learning_rate * new_weights[name] + 
                        (1 - self.learning_rate) * self.dynamic_weights[name]
                    )
                else:
                    self.dynamic_weights[name] = new_weights[name]
        
        # Normalize weights
        total_weight = sum(self.dynamic_weights.values())
        self.dynamic_weights = {
            name: weight / total_weight 
            for name, weight in self.dynamic_weights.items()
        }
        
        print(f"   Updated weights: {self.dynamic_weights}")
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_errors': model_errors,
            'weights': self.dynamic_weights.copy()
        })
        
        return self.dynamic_weights
    
    def adapt_learning_rate(self, performance_trend):
        """
        Adapt learning rate based on performance trend
        """
        if len(self.performance_history) >= 3:
            recent_errors = [h['model_errors'] for h in self.performance_history[-3:]]
            
            # Calculate if we're improving
            avg_error_trend = []
            for i in range(1, len(recent_errors)):
                current_avg = np.mean(list(recent_errors[i].values()))
                previous_avg = np.mean(list(recent_errors[i-1].values()))
                avg_error_trend.append(current_avg - previous_avg)
            
            # Adjust learning rate
            if np.mean(avg_error_trend) < 0:  # Improving
                self.learning_rate = min(0.3, self.learning_rate * 1.1)
                print(f"   📈 Performance improving, increased learning rate to {self.learning_rate:.3f}")
            else:  # Getting worse
                self.learning_rate = max(0.01, self.learning_rate * 0.9)
                print(f"   📉 Performance declining, decreased learning rate to {self.learning_rate:.3f}")
    
    # ========================
    # ONLINE LEARNING & ADAPTATION
    # ========================
    
    def update_with_new_data(self, new_sales_data, retrain_threshold=100):
        """
        Update models with new sales data (online learning)
        """
        print(f"\n🔄 Updating models with {len(new_sales_data)} new data points...")
        
        # Store new data
        if not hasattr(self, 'incremental_data'):
            self.incremental_data = []
        
        self.incremental_data.extend(new_sales_data.to_dict('records'))
        
        # Check if we should retrain
        if len(self.incremental_data) >= retrain_threshold:
            print(f"   📚 Accumulated {len(self.incremental_data)} samples, triggering retrain...")
            
            # Convert to DataFrame and create features
            incremental_df = pd.DataFrame(self.incremental_data)
            new_features = self._create_features_from_sales(incremental_df)
            
            # Partial fit or retrain models
            self._incremental_fit(new_features)
            
            # Clear accumulated data
            self.incremental_data = []
            
            # Update performance metrics
            self._evaluate_incremental_performance(new_features)
    
    def _incremental_fit(self, new_features):
        """
        Incrementally update models with new data
        """
        # Prepare features for ML
        X_new, _ = self._prepare_features_for_ml(new_features)
        y_new = new_features['target_demand']
        
        # For models that support incremental learning
        for name, model in self.models.items():
            if hasattr(model, 'partial_fit'):
                # SGD-based models support partial_fit
                model.partial_fit(X_new, y_new)
                print(f"   ✅ Updated {name} incrementally")
            else:
                # For tree-based models, we need to retrain
                # But we can use warm_start for some efficiency
                if name == 'rf':
                    model.n_estimators += 10  # Add more trees
                    model.fit(X_new, y_new)
                    print(f"   🌳 Added trees to {name}")
    
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
    
    def _evaluate_incremental_performance(self, new_features):
        """Evaluate performance after incremental learning"""
        try:
            X_new, _ = self._prepare_features_for_ml(new_features)
            y_new = new_features['target_demand']
            
            # Get predictions and calculate metrics
            predictions = {}
            for name, model in self.models.items():
                try:
                    predictions[name] = model.predict(X_new)
                except:
                    continue
            
            if predictions:
                # Update weights based on new performance
                self.update_ensemble_weights_dynamically(X_new, y_new, predictions)
        except Exception as e:
            print(f"   ⚠️ Could not evaluate incremental performance: {e}")
    
    # ========================
    # PERFORMANCE MONITORING
    # ========================
    
    def monitor_model_drift(self, current_predictions, actual_values):
        """
        Monitor for model drift and trigger retraining if needed
        """
        print("\n📊 Monitoring model drift...")
        
        # Calculate current performance
        current_mae = mean_absolute_error(actual_values, current_predictions)
        current_mape = np.mean(np.abs((actual_values - current_predictions) / np.maximum(actual_values, 1))) * 100
        
        # Compare with baseline performance
        if hasattr(self, 'baseline_performance'):
            baseline_mae = self.baseline_performance.get('mae', current_mae)
            baseline_mape = self.baseline_performance.get('mape', current_mape)
            
            mae_degradation = (current_mae - baseline_mae) / max(baseline_mae, 0.001)
            mape_degradation = (current_mape - baseline_mape) / max(baseline_mape, 0.001)
            
            print(f"   Current MAE: {current_mae:.4f} (baseline: {baseline_mae:.4f})")
            print(f"   MAE degradation: {mae_degradation*100:.2f}%")
            print(f"   Current MAPE: {current_mape:.2f}% (baseline: {baseline_mape:.2f}%)")
            
            # Trigger retraining if performance degrades significantly
            retrain_recommended = mae_degradation > 0.15 or mape_degradation > 0.15
            
            if retrain_recommended:
                print(f"   🚨 Significant performance degradation detected!")
                print(f"   💡 Recommendation: Retrain models with recent data")
            else:
                print(f"   ✅ Model performance is stable")
                
            return {
                'retrain_recommended': retrain_recommended, 
                'degradation': {'mae': mae_degradation, 'mape': mape_degradation},
                'current_performance': {'mae': current_mae, 'mape': current_mape},
                'baseline_performance': {'mae': baseline_mae, 'mape': baseline_mape}
            }
        else:
            # Set baseline if not exists
            self.baseline_performance = {'mae': current_mae, 'mape': current_mape}
            print(f"   📏 Set baseline performance: MAE={current_mae:.4f}, MAPE={current_mape:.2f}%")
            
            return {
                'retrain_recommended': False, 
                'current_performance': {'mae': current_mae, 'mape': current_mape},
                'baseline_performance': {'mae': current_mae, 'mape': current_mape}
            }
    
    # ========================
    # MODEL PERSISTENCE & VERSIONING
    # ========================
    
    def save_models(self, version=None):
        """
        Save models with versioning
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = os.path.join(self.model_save_path, f"version_{version}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{name}_model.joblib")
            joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'ensemble_weights': self.ensemble_weights,
            'dynamic_weights': self.dynamic_weights,
            'feature_columns': self.feature_columns,
            'best_params': self.best_params,
            'performance_history': self.performance_history[-10:],  # Last 10 entries
            'baseline_performance': getattr(self, 'baseline_performance', None)
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save encoders
        encoders_path = os.path.join(model_dir, "encoders.joblib")
        joblib.dump(self.label_encoders, encoders_path)
        
        print(f"✅ Models saved to {model_dir}")
        return model_dir
    
    def load_models(self, version):
        """
        Load models from specific version
        """
        model_dir = os.path.join(self.model_save_path, f"version_{version}")
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Model version {version} not found")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load models
        self.models = {}
        for model_file in os.listdir(model_dir):
            if model_file.endswith('_model.joblib'):
                model_name = model_file.replace('_model.joblib', '')
                model_path = os.path.join(model_dir, model_file)
                self.models[model_name] = joblib.load(model_path)
        
        # Restore metadata
        self.ensemble_weights = metadata['ensemble_weights']
        self.dynamic_weights = metadata.get('dynamic_weights', {})
        self.feature_columns = metadata['feature_columns']
        self.best_params = metadata.get('best_params', {})
        self.performance_history = metadata.get('performance_history', [])
        self.baseline_performance = metadata.get('baseline_performance')
        
        # Load encoders
        encoders_path = os.path.join(model_dir, "encoders.joblib")
        if os.path.exists(encoders_path):
            self.label_encoders = joblib.load(encoders_path)
        
        print(f"✅ Models loaded from {model_dir}")
        return metadata
    
    # ========================
    # A/B TESTING FRAMEWORK
    # ========================
    
    def setup_ab_testing(self, test_ratio=0.2):
        """
        Setup A/B testing framework to compare model versions
        """
        self.ab_test_config = {
            'test_ratio': test_ratio,
            'control_version': 'current',
            'test_version': 'latest',
            'results': []
        }
        print(f"🧪 A/B testing setup: {test_ratio*100}% traffic to test model")
    
    def ab_test_predict(self, prediction_features_df):
        """
        Make predictions using A/B testing framework
        """
        import random
        
        # Determine which model to use for each prediction
        results = []
        
        for idx, row in prediction_features_df.iterrows():
            use_test_model = random.random() < self.ab_test_config['test_ratio']
            
            # Make prediction (simplified - you'd have different model versions)
            if use_test_model:
                # Use test model (could be different parameters/weights)
                prediction = self._predict_with_test_model(row)
                model_version = 'test'
            else:
                # Use control model
                prediction = self._predict_with_control_model(row)
                model_version = 'control'
            
            results.append({
                'product_code': row['Product Code'],
                'prediction': prediction,
                'model_version': model_version,
                'timestamp': datetime.now().isoformat()
            })
        
        return results
    
    def _predict_with_test_model(self, row):
        """Placeholder for test model prediction"""
        # Implement test model logic here
        return np.random.randint(1, 20)  # Placeholder
    
    def _predict_with_control_model(self, row):
        """Placeholder for control model prediction"""
        # Implement control model logic here
        return np.random.randint(1, 15)  # Placeholder
    
    # ========================
    # HELPER METHODS
    # ========================
    
    def _get_default_parameters(self):
        """Default parameters if optimization is not available"""
        return {
            'rf': {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgb': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': 0
            } if XGBOOST_AVAILABLE else {},
            'lgb': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            } if LIGHTGBM_AVAILABLE else {}
        }
    
    def _optimize_with_grid_search(self, X_train, y_train):
        """Grid search optimization (simplified)"""
        print("🔍 Using Grid Search optimization...")
        
        # Simplified parameter grid
        param_grid = {
            'rf': {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 12, 16],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        best_params = {}
        
        # Grid search for Random Forest
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid['rf'], 
            cv=TimeSeriesSplit(n_splits=3), 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        try:
            grid_search.fit(X_train, y_train)
            best_params['rf'] = grid_search.best_params_
            print(f"✅ Grid Search completed. Best RF params: {best_params['rf']}")
        except Exception as e:
            print(f"⚠️ Grid Search failed: {e}")
            best_params = self._get_default_parameters()
        
        return best_params
    
    def _optimize_with_random_search(self, X_train, y_train):
        """Random search optimization (simplified)"""
        print("🎲 Using Random Search optimization...")
        
        # Simplified parameter distributions
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [8, 10, 12, 15, 18],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6]
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        random_search = RandomizedSearchCV(
            rf, param_dist,
            n_iter=20,
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1
        )
        
        try:
            random_search.fit(X_train, y_train)
            best_params = {'rf': random_search.best_params_}
            print(f"✅ Random Search completed. Best params: {best_params}")
            return best_params
        except Exception as e:
            print(f"⚠️ Random Search failed: {e}")
            return self._get_default_parameters()
    
    def get_optimization_report(self):
        """
        Generate comprehensive optimization report
        """
        report = {
            'current_weights': self.dynamic_weights,
            'performance_history': self.performance_history[-5:],  # Last 5 entries
            'best_parameters': self.best_params,
            'learning_rate': self.learning_rate,
            'baseline_performance': getattr(self, 'baseline_performance', None),
            'category_models': len(getattr(self, 'category_models', {})),
            'model_versions_saved': len([d for d in os.listdir(self.model_save_path) if d.startswith('version_')]) if os.path.exists(self.model_save_path) else 0,
            'total_features': len(getattr(self, 'feature_columns', [])),
            'available_optimizations': {
                'optuna': OPTUNA_AVAILABLE,
                'xgboost': XGBOOST_AVAILABLE,
                'lightgbm': LIGHTGBM_AVAILABLE
            }
        }
        
        return report
    def _add_temporal_features(self, features_df, sales_df):
        """Add seasonal and temporal performance features - FIXED VERSION"""
        try:
            # Seasonal performance if Season is available
            if 'Season' in self.brand_features:
                # Create seasonal numeric mapping
                sales_df_copy = sales_df.copy()
                sales_df_copy['Season_Numeric'] = sales_df_copy['Sale Date'].dt.month.map({
                    12: 1, 1: 1, 2: 1,  # Winter
                    3: 2, 4: 2, 5: 2,   # Spring  
                    6: 3, 7: 3, 8: 3,   # Summer
                    9: 4, 10: 4, 11: 4  # Fall
                })
                
                seasonal_sales = sales_df_copy.groupby(['Product Code', 'Season_Numeric']).size().reset_index(name='seasonal_sales')
                seasonal_pivot = seasonal_sales.pivot(index='Product Code', columns='Season_Numeric', values='seasonal_sales').fillna(0)
                seasonal_pivot.columns = [f'season_{int(col)}_sales' for col in seasonal_pivot.columns]
                
                # Convert to regular DataFrame to avoid categorical issues
                seasonal_pivot = seasonal_pivot.reset_index()
                
                features_df = features_df.merge(seasonal_pivot, left_on='Product Code', right_on='Product Code', how='left')
                features_df = features_df.fillna(0)
            
            # Monthly patterns - Fixed to avoid categorical issues
            sales_df_copy = sales_df.copy()
            sales_df_copy['Month'] = sales_df_copy['Sale Date'].dt.month
            monthly_sales = sales_df_copy.groupby(['Product Code', 'Month']).size().reset_index(name='monthly_sales')
            monthly_stats = monthly_sales.groupby('Product Code')['monthly_sales'].agg(['mean', 'std']).reset_index()
            monthly_stats.columns = ['Product Code', 'monthly_avg_sales', 'monthly_std_sales']
            monthly_stats['monthly_std_sales'] = monthly_stats['monthly_std_sales'].fillna(0)
            
            features_df = features_df.merge(monthly_stats, on='Product Code', how='left')
            
        except Exception as e:
            print(f"   ⚠️ Warning: Could not create temporal features: {e}")
        
        return features_df
    
    # ========================
    # ENHANCED TRAINING METHODS
    # ========================
    
    def train_ensemble_model_advanced(self, features_df, model_params=None):
        """
        Enhanced ensemble training with proper feature preparation and detailed error handling
        """
        print("🚀 Training advanced ensemble model...")
        
        # Prepare features for ML
        X, feature_columns = self._prepare_features_for_ml(features_df)
        y = features_df['target_demand']
        
        print(f"Training ensemble on {len(X)} samples with {len(feature_columns)} features")
        print(f"Target range: {y.min()} to {y.max()}, mean: {y.mean():.2f}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().head()}")
        
        # Check for data issues
        if len(X) == 0:
            raise ValueError("No training data available")
        
        if X.isnull().any().any():
            print("⚠️ Warning: Found NaN values in features, filling with 0")
            X = X.fillna(0)
        
        if y.isnull().any():
            print("⚠️ Warning: Found NaN values in target, removing those rows")
            valid_mask = ~y.isnull()
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Use optimized parameters or defaults
        if model_params is None:
            if hasattr(self, 'best_params') and self.best_params:
                model_params = self.best_params
                print("📊 Using optimized parameters from hyperparameter tuning")
            else:
                model_params = self._get_default_parameters()
                print("📊 Using default parameters")
        
        # Initialize models
        self.models = {}
        training_errors = {}
        
        # Random Forest - Most reliable, try first
        print("\n🌳 Training Random Forest...")
        if 'rf' in model_params and model_params['rf']:
            try:
                rf_params = model_params['rf'].copy()
                # Ensure parameters are reasonable for the dataset size
                rf_params['n_estimators'] = min(rf_params.get('n_estimators', 200), 500)
                rf_params['max_depth'] = min(rf_params.get('max_depth', 12), 20)
                rf_params['min_samples_leaf'] = max(rf_params.get('min_samples_leaf', 2), 1)
                rf_params['min_samples_split'] = max(rf_params.get('min_samples_split', 5), 2)
                
                print(f"   RF params: {rf_params}")
                
                self.models['rf'] = RandomForestRegressor(**rf_params)
                self.models['rf'].fit(X, y)
                
                # Test prediction
                test_pred = self.models['rf'].predict(X[:10])
                print(f"  ✅ Random Forest trained successfully - sample predictions: {test_pred[:3]}")
                
            except Exception as e:
                training_errors['rf'] = str(e)
                print(f"  ❌ Random Forest training failed: {e}")
                if 'rf' in self.models:
                    del self.models['rf']
        
        # XGBoost - Try with reduced complexity
        print("\n🚀 Training XGBoost...")
        if 'xgb' in model_params and model_params['xgb'] and XGBOOST_AVAILABLE:
            try:
                xgb_params = model_params['xgb'].copy()
                # Conservative parameters for stability
                xgb_params['n_estimators'] = min(xgb_params.get('n_estimators', 100), 300)
                xgb_params['max_depth'] = min(xgb_params.get('max_depth', 6), 10)
                xgb_params['learning_rate'] = max(xgb_params.get('learning_rate', 0.1), 0.01)
                
                # Add stability parameters
                xgb_params.update({
                    'random_state': 42,
                    'verbosity': 0,
                    'objective': 'reg:squarederror',
                    'eval_metric': 'mae'
                })
                
                print(f"   XGB params: {xgb_params}")
                
                self.models['xgb'] = XGBRegressor(**xgb_params)
                self.models['xgb'].fit(X, y)
                
                # Test prediction
                test_pred = self.models['xgb'].predict(X[:10])
                print(f"  ✅ XGBoost trained successfully - sample predictions: {test_pred[:3]}")
                
            except Exception as e:
                training_errors['xgb'] = str(e)
                print(f"  ❌ XGBoost training failed: {e}")
                if 'xgb' in self.models:
                    del self.models['xgb']
        
        # LightGBM - Try with reduced complexity
        print("\n💡 Training LightGBM...")
        if 'lgb' in model_params and model_params['lgb'] and LIGHTGBM_AVAILABLE:
            try:
                lgb_params = model_params['lgb'].copy()
                # Conservative parameters for stability
                lgb_params['n_estimators'] = min(lgb_params.get('n_estimators', 100), 300)
                lgb_params['max_depth'] = min(lgb_params.get('max_depth', 6), 10)
                lgb_params['learning_rate'] = max(lgb_params.get('learning_rate', 0.1), 0.01)
                
                # Add stability parameters
                lgb_params.update({
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'mae',
                    'force_row_wise': True  # For stability
                })
                
                print(f"   LGB params: {lgb_params}")
                
                self.models['lgb'] = LGBMRegressor(**lgb_params)
                self.models['lgb'].fit(X, y)
                
                # Test prediction
                test_pred = self.models['lgb'].predict(X[:10])
                print(f"  ✅ LightGBM trained successfully - sample predictions: {test_pred[:3]}")
                
            except Exception as e:
                training_errors['lgb'] = str(e)
                print(f"  ❌ LightGBM training failed: {e}")
                if 'lgb' in self.models:
                    del self.models['lgb']
        
        # If no models from parameters worked, try basic RandomForest
        if not self.models:
            print("\n🔄 All parameterized models failed, trying basic RandomForest...")
            try:
                basic_rf_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                self.models['rf_basic'] = RandomForestRegressor(**basic_rf_params)
                self.models['rf_basic'].fit(X, y)
                
                test_pred = self.models['rf_basic'].predict(X[:10])
                print(f"  ✅ Basic Random Forest trained successfully - sample predictions: {test_pred[:3]}")
                
            except Exception as e:
                training_errors['rf_basic'] = str(e)
                print(f"  ❌ Basic Random Forest training failed: {e}")
        
        # Final check
        if not self.models:
            error_summary = "\n".join([f"  {model}: {error}" for model, error in training_errors.items()])
            raise ValueError(f"No models were successfully trained!\n\nDetailed errors:\n{error_summary}\n\nData info:\n  Shape: {X.shape}\n  Target range: {y.min()}-{y.max()}\n  Features: {feature_columns[:5]}...")
        
        # Calculate ensemble weights based on training performance
        model_predictions = {}
        for name, model in self.models.items():
            try:
                train_pred = model.predict(X)
                model_predictions[name] = train_pred
                
                # Calculate and print individual model performance
                mae = mean_absolute_error(y, train_pred)
                print(f"  📊 {name} training MAE: {mae:.4f}")
                
            except Exception as e:
                print(f"    ⚠️ Could not get predictions from {name}: {e}")
        
        if model_predictions:
            self.ensemble_weights = self._calculate_ensemble_weights(model_predictions, y)
        else:
            # Equal weights if no predictions available
            self.ensemble_weights = {name: 1.0/len(self.models) for name in self.models.keys()}
        
        print(f"\n✅ Advanced ensemble trained successfully!")
        print(f"   Models trained: {list(self.models.keys())}")
        print(f"   Ensemble weights: {self.ensemble_weights}")
        print(f"   Total features used: {len(feature_columns)}")
        
        if training_errors:
            print(f"\n⚠️ Some models failed to train:")
            for model, error in training_errors.items():
                print(f"   {model}: {error}")
        
        return self.models
    
    def _calculate_ensemble_weights(self, model_predictions, y_true):
        """Calculate optimal ensemble weights"""
        weights = {}
        total_weight = 0
        
        for name, pred in model_predictions.items():
            # Calculate inverse of MAE as weight (better models get higher weight)
            mae = mean_absolute_error(y_true, pred)
            weight = 1 / (mae + 0.01)  # Add small constant to avoid division by zero
            weights[name] = weight
            total_weight += weight
        
        # Normalize weights to sum to 1
        for name in weights:
            weights[name] = weights[name] / total_weight
        
        return weights
    
    # ========================
    # ENHANCED PREDICTION METHODS
    # ========================
    
    def predict_demand_ensemble_advanced(self, prediction_features_df):
        """
        Generate ensemble predictions with proper feature preparation
        """
        if not self.models:
            raise ValueError("No models trained. Call train_ensemble_model_advanced() first.")
        
        print(f"🔮 Generating advanced ensemble predictions for {len(prediction_features_df)} products")
        
        # Prepare prediction features
        X, _ = self._prepare_features_for_ml(prediction_features_df)
        
        print(f"   Prediction features shape: {X.shape}")
        
        # Get predictions from all models
        model_predictions = {}
        for name, model in self.models.items():
            try:
                raw_pred = model.predict(X)
                model_predictions[name] = raw_pred
                print(f"   ✅ {name} predictions generated")
            except Exception as e:
                print(f"   ⚠️ {name} prediction failed: {e}")
                continue
        
        if not model_predictions:
            raise ValueError("All models failed to generate predictions!")
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, pred in model_predictions.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            ensemble_pred += weight * pred
        
        # Apply business rules and constraints
        adjusted_predictions = self._apply_business_rules_advanced(ensemble_pred, prediction_features_df)
        
        # Create results dataframe
        results_df = pd.DataFrame()
        results_df['Product Code'] = prediction_features_df['Product Code'].values
        results_df['predicted_demand'] = adjusted_predictions.astype(int)
        results_df['confidence_score'] = self._calculate_advanced_confidence_v2(
            prediction_features_df, adjusted_predictions, model_predictions
        )
        results_df['risk_level'] = results_df['confidence_score'].apply(
            lambda x: 'LOW' if x >= 75 else 'MEDIUM' if x >= 50 else 'HIGH'
        )
        
        # Remove any duplicate columns
        results_df = results_df.loc[:, ~results_df.columns.duplicated()]
        
        print(f"✅ Advanced ensemble predictions complete")
        print(f"   Final range: {adjusted_predictions.min():.0f} to {adjusted_predictions.max():.0f}")
        print(f"   Final mean: {adjusted_predictions.mean():.1f}")
        print(f"   Confidence distribution: {results_df['risk_level'].value_counts().to_dict()}")
        
        return results_df
    
    def _apply_business_rules_advanced(self, predictions, prediction_features_df):
        """
        Apply enhanced business constraints and domain knowledge
        """
        adjusted_predictions = predictions.copy()
        
        for i, (_, row) in enumerate(prediction_features_df.iterrows()):
            pred = predictions[i]
            
            # Get product attributes (handle both _first and direct columns)
            category = str(row.get('Category_first', row.get('Category', 'Unknown')))
            season = str(row.get('Season_first', row.get('Season', 'Unknown')))
            gender = str(row.get('Gender_first', row.get('Gender', 'Unknown')))
            size = str(row.get('Size Name_first', row.get('Size Name', 'Unknown')))
            
            # Enhanced category-based adjustments
            if any(keyword in category.lower() for keyword in ['under garments', 'basic']):
                adjusted_predictions[i] = max(pred * 1.3, 6)
            elif any(keyword in category.lower() for keyword in ['top', 'shirt', 'pant', 'trouser']):
                adjusted_predictions[i] = max(pred * 1.15, 4)
            elif any(keyword in category.lower() for keyword in ['dress', 'eastern', 'jacket']):
                adjusted_predictions[i] = max(pred * 1.2, 3)
            elif any(keyword in category.lower() for keyword in ['belt', 'accessories']):
                adjusted_predictions[i] = max(pred * 0.85, 2)
            else:
                adjusted_predictions[i] = max(pred, 3)
            
            # Enhanced seasonal adjustments
            current_month = datetime.now().month
            
            if 'winter' in season.lower() and current_month in [10, 11, 12, 1, 2]:
                adjusted_predictions[i] *= 1.5
            elif 'summer' in season.lower() and current_month in [4, 5, 6, 7, 8]:
                adjusted_predictions[i] *= 1.4
            elif 'open season' in season.lower():
                adjusted_predictions[i] *= 1.2
            elif season.lower() not in ['unknown', 'open season']:
                adjusted_predictions[i] *= 0.8
            
            # Enhanced size popularity adjustments
            popular_kids_sizes = ['5-6y', '7-8y', '9-10y', '11-12y', '13-14y']
            popular_adult_sizes = ['m', 'l', 'xl']
            
            size_lower = size.lower()
            if any(size_pattern in size_lower for size_pattern in popular_kids_sizes):
                adjusted_predictions[i] *= 1.3
            elif any(size_pattern in size_lower for size_pattern in popular_adult_sizes):
                adjusted_predictions[i] *= 1.2
            elif any(size_pattern in size_lower for size_pattern in ['xs', 'xxl', 'xxxl']):
                adjusted_predictions[i] *= 0.7
            
            # Gender-based adjustments
            if 'female' in gender.lower():
                adjusted_predictions[i] *= 1.2
            elif 'male' in gender.lower():
                adjusted_predictions[i] *= 1.1
            
            # Add controlled randomness
            random_factor = np.random.uniform(0.9, 1.15)
            adjusted_predictions[i] *= random_factor
            
            # Final constraints
            adjusted_predictions[i] = max(int(adjusted_predictions[i]), 1)
            adjusted_predictions[i] = min(adjusted_predictions[i], 50)
        
        return adjusted_predictions
    
    def _calculate_advanced_confidence_v2(self, prediction_features_df, predictions, model_predictions):
        """
        Enhanced confidence calculation with more factors
        """
        confidence_scores = []
        
        for i, (_, row) in enumerate(prediction_features_df.iterrows()):
            score = 50  # Base confidence
            
            # Model agreement (ensemble consistency)
            if len(model_predictions) > 1:
                pred_values = [pred[i] for pred in model_predictions.values()]
                pred_std = np.std(pred_values)
                pred_mean = np.mean(pred_values)
                cv = pred_std / (pred_mean + 0.01)
                
                if cv < 0.15:
                    score += 25
                elif cv < 0.3:
                    score += 15
                elif cv < 0.5:
                    score += 5
                else:
                    score -= 10
            
            # Feature quality assessment
            total_sales = row.get('total_sales', 0)
            sales_velocity = row.get('sales_velocity', 0)
            
            if total_sales > 10:
                score += 15
            elif total_sales > 5:
                score += 10
            elif total_sales > 2:
                score += 5
            
            if sales_velocity > 0.05:
                score += 10
            elif sales_velocity > 0.02:
                score += 5
            
            # Prediction reasonableness
            pred_value = predictions[i]
            if 3 <= pred_value <= 25:
                score += 15
            elif 1 <= pred_value <= 35:
                score += 5
            elif pred_value > 40:
                score -= 15
            
            # Category confidence
            category = str(row.get('Category_first', row.get('Category', 'Unknown'))).lower()
            if any(keyword in category for keyword in ['under garments', 'basic', 'top', 'pant']):
                score += 10
            elif any(keyword in category for keyword in ['dress', 'eastern']):
                score += 5
            
            # Historical validation performance
            if hasattr(self, 'baseline_performance') and self.baseline_performance:
                baseline_mae = self.baseline_performance.get('mae', 10)
                if baseline_mae < 5:
                    score += 15
                elif baseline_mae < 8:
                    score += 10
                elif baseline_mae < 12:
                    score += 5
            
            # Final confidence score
            final_score = max(15, min(95, score))
            confidence_scores.append(final_score)
        
        return confidence_scores