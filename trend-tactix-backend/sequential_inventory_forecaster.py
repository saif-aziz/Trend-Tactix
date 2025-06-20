import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using Random Forest")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class SequentialInventoryForecaster:
    """
    Sequential inventory forecasting system optimized for your sales dataset
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type if self._check_model_availability(model_type) else 'random_forest'
        self.models = {}
        self.feature_columns = []
        self.results_history = []
        
    def _check_model_availability(self, model_type):
        """Check if requested model is available"""
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            return False
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            return False
        return True
        
    def load_and_prepare_data(self, csv_path):
        """Load and prepare your sales data"""
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()
            
            # Ensure proper date parsing
            df['Sale Date'] = pd.to_datetime(df['Sale Date'])
            df['Year'] = df['Sale Date'].dt.year
            df['Month'] = df['Sale Date'].dt.month
            df['Quarter'] = df['Sale Date'].dt.quarter
            df['Week'] = df['Sale Date'].dt.isocalendar().week
            df['DayOfWeek'] = df['Sale Date'].dt.dayofweek
            
            # Create season mapping
            df['Season_Numeric'] = df['Sale Date'].dt.month.map({
                12: 1, 1: 1, 2: 1,  # Winter
                3: 2, 4: 2, 5: 2,   # Spring  
                6: 3, 7: 3, 8: 3,   # Summer
                9: 4, 10: 4, 11: 4  # Fall
            })
            
            # Sort by date
            df = df.sort_values('Sale Date').reset_index(drop=True)
            
            print(f"✅ Data loaded successfully:")
            print(f"   📊 {len(df):,} sales records")
            print(f"   📅 Date range: {df['Sale Date'].min()} to {df['Sale Date'].max()}")
            print(f"   🏷️  Unique SKUs: {df['Product Code'].nunique():,}")
            print(f"   🏪 Unique shops: {df['Shop'].nunique()}")
            print(f"   📦 Categories: {df['Category'].nunique()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    train_end_date=datetime(2024, 12, 31)
    def create_features(self, df, train_end_date):
        """Create features for modeling based on your dataset structure"""
        
        # Filter training data up to train_end_date
        train_mask = df['Sale Date'] <= train_end_date
        train_df = df[train_mask].copy()
        
        print(f"Creating features from {len(train_df):,} records up to {train_end_date}")
        
        # SKU-level historical features
        sku_features = train_df.groupby('Product Code').agg({
            'Sale Date': ['count', 'min', 'max'],
            'Category': 'first',
            'Gender': 'first',
            'Season': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'Size Name': 'first',
            'Color Name': 'first'
        }).reset_index()
        
        # Flatten column names
        sku_features.columns = ['Product Code', 'total_sales', 'first_sale_date', 
                               'last_sale_date', 'category', 'gender', 'primary_season',
                               'size_name', 'color_name']
        
        # Calculate velocity and recency features
        reference_date = pd.to_datetime(train_end_date)
        sku_features['days_since_first_sale'] = (reference_date - sku_features['first_sale_date']).dt.days
        sku_features['days_since_last_sale'] = (reference_date - sku_features['last_sale_date']).dt.days
        sku_features['sales_velocity'] = sku_features['total_sales'] / (sku_features['days_since_first_sale'] + 1)
        
        # Seasonal sales patterns
        seasonal_sales = train_df.groupby(['Product Code', 'Season_Numeric']).size().reset_index(name='seasonal_sales')
        seasonal_pivot = seasonal_sales.pivot(index='Product Code', columns='Season_Numeric', values='seasonal_sales').fillna(0)
        seasonal_pivot.columns = [f'season_{int(col)}_sales' for col in seasonal_pivot.columns]
        
        # Monthly patterns
        monthly_sales = train_df.groupby(['Product Code', 'Month']).size().reset_index(name='monthly_sales')
        monthly_avg = monthly_sales.groupby('Product Code')['monthly_sales'].agg(['mean', 'std']).reset_index()
        monthly_avg.columns = ['Product Code', 'monthly_avg_sales', 'monthly_std_sales']
        monthly_avg['monthly_std_sales'] = monthly_avg['monthly_std_sales'].fillna(0)
        
        # Merge all features
        features_df = sku_features.merge(seasonal_pivot, left_on='Product Code', right_index=True, how='left')
        features_df = features_df.merge(monthly_avg, on='Product Code', how='left')
        features_df = features_df.fillna(0)
        
        # Category-level features
        category_stats = train_df.groupby('Category').agg({
            'Product Code': 'nunique',
            'Sale Date': 'count'
        }).reset_index()
        category_stats.columns = ['category', 'category_sku_count', 'category_total_sales']
        
        features_df = features_df.merge(category_stats, on='category', how='left')
        
        # One-hot encode categorical variables
        categorical_columns = ['category', 'gender', 'primary_season']
        for col in categorical_columns:
            if col in features_df.columns:
                dummies = pd.get_dummies(features_df[col], prefix=col)
                features_df = pd.concat([features_df, dummies], axis=1)
                features_df = features_df.drop(col, axis=1)
        
        print(f"✅ Created {len(features_df.columns)} features for {len(features_df)} SKUs")
        
        return features_df
    predict_start_date=datetime(2025, 1, 1)
    predict_end_date=datetime(2025, 12, 31)
    def create_target_variable(self, df, predict_start_date, predict_end_date):
        """Create target variable for the prediction period"""
        
        future_mask = (df['Sale Date'] >= predict_start_date) & (df['Sale Date'] <= predict_end_date)
        future_sales = df[future_mask].groupby('Product Code').size().reset_index(name='future_sales')
        
        print(f"Target period: {predict_start_date} to {predict_end_date}")
        print(f"SKUs with sales in target period: {len(future_sales)}")
        
        return future_sales
    
    def train_model(self, features_df, target_df, model_params=None):
        """Train the forecasting model"""
        
        # Merge features with targets
        train_data = features_df.merge(target_df, on='Product Code', how='left')
        train_data['future_sales'] = train_data['future_sales'].fillna(0)
        
        # Prepare feature matrix (exclude non-feature columns)
        exclude_cols = ['Product Code', 'future_sales', 'first_sale_date', 'last_sale_date', 
                       'size_name', 'color_name']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = train_data[feature_cols]
        y = train_data['future_sales']
        
        # Train model based on type
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                **(model_params or {})
            )
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1,
                **(model_params or {})
            )
        else:  # Random Forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                **(model_params or {})
            )
        
        model.fit(X, y)
        
        print(f"✅ Model trained on {len(X)} SKUs with {len(feature_cols)} features")
        
        return model, train_data
    
    def predict_inventory(self, model, features_df):
        """Make inventory predictions"""
        
        X = features_df[self.feature_columns]
        predictions = model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)
        
        results_df = features_df[['Product Code']].copy()
        results_df['predicted_demand'] = predictions
        
        return results_df
    
    def evaluate_predictions(self, predictions_df, actuals_df):
        """Evaluate model performance"""
        
        evaluation_df = predictions_df.merge(actuals_df, on='Product Code', how='outer')
        evaluation_df['predicted_demand'] = evaluation_df['predicted_demand'].fillna(0)
        evaluation_df['future_sales'] = evaluation_df['future_sales'].fillna(0)
        
        # Calculate metrics
        mae = mean_absolute_error(evaluation_df['future_sales'], evaluation_df['predicted_demand'])
        rmse = np.sqrt(mean_squared_error(evaluation_df['future_sales'], evaluation_df['predicted_demand']))
        
        # MAPE (handle division by zero)
        non_zero_mask = evaluation_df['future_sales'] > 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((evaluation_df.loc[non_zero_mask, 'future_sales'] - 
                                 evaluation_df.loc[non_zero_mask, 'predicted_demand']) / 
                                evaluation_df.loc[non_zero_mask, 'future_sales'])) * 100
        else:
            mape = 0
        
        # Coverage ratio
        total_actual = evaluation_df['future_sales'].sum()
        total_predicted = evaluation_df['predicted_demand'].sum()
        coverage_ratio = total_predicted / total_actual if total_actual > 0 else 0
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Coverage_Ratio': coverage_ratio,
            'Total_Actual_Sales': total_actual,
            'Total_Predicted_Demand': total_predicted,
            'SKUs_Evaluated': len(evaluation_df)
        }
        
        return metrics, evaluation_df
    
    def quick_forecast_single_product(self, df, product_code):
        """Quick forecast for a single product (used by API)"""
        try:
            product_sales = df[df['Product Code'] == product_code]
            
            if len(product_sales) == 0:
                return {
                    'predictedDemand': 0,
                    'confidence': 0,
                    'riskLevel': 'HIGH',
                    'reasoning': 'No historical sales data available'
                }
            
            # Basic calculations
            total_sales = len(product_sales)
            dates = pd.to_datetime(product_sales['Sale Date'])
            date_range_days = (dates.max() - dates.min()).days
            months_span = max(1, date_range_days / 30)
            velocity = total_sales / months_span
            
            # Predict for next season (3 months)
            predicted_demand = max(1, int(velocity * 3))
            
            # Confidence based on data consistency
            if len(product_sales) >= 5:
                monthly_sales = product_sales.groupby(dates.dt.to_period('M')).size()
                if len(monthly_sales) > 1:
                    cv = monthly_sales.std() / monthly_sales.mean()
                    confidence = max(10, min(95, 80 - (cv * 30)))
                else:
                    confidence = 60
            else:
                confidence = 30
            
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
            return {
                'predictedDemand': 1,
                'confidence': 0,
                'riskLevel': 'HIGH',
                'reasoning': f'Error in calculation: {str(e)}'
            }