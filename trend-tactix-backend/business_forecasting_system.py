# import pandas as pd
# import numpy as np

# class BusinessForecastingSystem:
#     def __init__(self):
#         self.category_trends = {}
#         self.risk_segments = {}
#         self.forecasts = {}
        
#     def analyze_business_patterns(self, data):
#         """Focus on actionable business insights"""
#         print("💼 BUSINESS-FOCUSED ANALYSIS...")
        
#         # Category performance trends
#         category_performance = {}
        
#         for category in data['Category'].unique():
#             cat_data = data[data['Category'] == category]
#             yearly_totals = cat_data.groupby('Year')['quantity_sold'].sum()
            
#             if len(yearly_totals) >= 2:
#                 recent_growth = (yearly_totals.iloc[-1] - yearly_totals.iloc[-2]) / yearly_totals.iloc[-2]
#                 avg_sales = yearly_totals.mean()
#                 volatility = yearly_totals.std() / avg_sales if avg_sales > 0 else 0
                
#                 # Business classification
#                 if recent_growth > 0.1 and avg_sales > 100:
#                     business_status = "🚀 Growth Star"
#                 elif recent_growth > 0.1:
#                     business_status = "📈 Emerging"
#                 elif recent_growth > -0.1 and avg_sales > 200:
#                     business_status = "🔄 Stable Core"
#                 elif recent_growth > -0.3:
#                     business_status = "⚠️ Declining"
#                 else:
#                     business_status = "🔻 Phase Out"
                
#                 category_performance[category] = {
#                     'recent_growth': recent_growth,
#                     'avg_sales': avg_sales,
#                     'volatility': volatility,
#                     'status': business_status,
#                     'latest_sales': yearly_totals.iloc[-1]
#                 }
        
#         # Sort by business priority
#         sorted_categories = sorted(
#             category_performance.items(), 
#             key=lambda x: (x[1]['recent_growth'], x[1]['avg_sales']), 
#             reverse=True
#         )
        
#         print(f"\n📊 CATEGORY BUSINESS ASSESSMENT:")
#         print(f"{'Category':<25} {'Status':<15} {'Growth':<10} {'Volume':<10}")
#         print("-" * 70)
        
#         for cat, metrics in sorted_categories[:10]:
#             print(f"{cat[:24]:<25} {metrics['status']:<15} {metrics['recent_growth']:>+7.1%} {metrics['latest_sales']:>8.0f}")
        
#         self.category_trends = category_performance
#         return category_performance
    
#     def create_risk_segments(self, data):
#         """Segment products by forecasting risk"""
#         print(f"\n🎯 RISK SEGMENTATION...")
        
#         risk_segments = {'LOW': [], 'MEDIUM': [], 'HIGH': []}
        
#         for product in data['Product Code'].unique():
#             product_data = data[data['Product Code'] == product].sort_values('Year')
            
#             if len(product_data) >= 2:
#                 # Calculate risk factors
#                 sales_history = product_data['quantity_sold'].values
#                 volatility = np.std(sales_history) / np.mean(sales_history) if np.mean(sales_history) > 0 else 1
                
#                 recent_sales = sales_history[-1]
#                 trend = (sales_history[-1] - sales_history[0]) / len(sales_history)
                
#                 category = product_data.iloc[-1]['Category']
#                 category_risk = self.category_trends.get(category, {}).get('volatility', 0.5)
                
#                 # Risk scoring
#                 risk_score = 0
#                 risk_score += min(volatility, 1.0) * 40  # Volatility (0-40 points)
#                 risk_score += max(0, -trend) * 20       # Negative trend (0-20 points)
#                 risk_score += category_risk * 30        # Category risk (0-30 points)
#                 risk_score += (1 - min(recent_sales/10, 1)) * 10  # Low volume (0-10 points)
                
#                 # Segment assignment
#                 if risk_score < 30:
#                     segment = 'LOW'
#                 elif risk_score < 60:
#                     segment = 'MEDIUM'
#                 else:
#                     segment = 'HIGH'
                
#                 risk_segments[segment].append({
#                     'product': product,
#                     'category': category,
#                     'risk_score': risk_score,
#                     'recent_sales': recent_sales,
#                     'volatility': volatility
#                 })
        
#         print(f"Risk Distribution:")
#         for segment, products in risk_segments.items():
#             print(f"{segment} RISK: {len(products)} products")
        
#         self.risk_segments = risk_segments
#         return risk_segments
    
#     def generate_business_forecasts(self, data, target_year=2025):
#         """Generate business-ready forecasts with confidence levels"""
#         print(f"\n🔮 BUSINESS FORECASTS FOR {target_year}...")
        
#         forecasts = {}
        
#         # Get latest year data
#         latest_year = data['Year'].max()
#         latest_data = data[data['Year'] == latest_year]
        
#         for _, row in latest_data.iterrows():
#             product = row['Product Code']
#             category = row['Category']
#             current_sales = row['quantity_sold']
            
#             # Get category trend
#             cat_trend = self.category_trends.get(category, {})
#             category_growth = cat_trend.get('recent_growth', 0)
#             category_status = cat_trend.get('status', '🔄 Stable Core')
            
#             # Find product in risk segments
#             product_risk = 'MEDIUM'  # Default
#             for risk_level, products in self.risk_segments.items():
#                 for p in products:
#                     if p['product'] == product:
#                         product_risk = risk_level
#                         break
            
#             # Conservative forecasting based on business logic
#             if '🚀' in category_status:  # Growth categories
#                 base_forecast = current_sales * (1 + min(category_growth * 0.7, 0.3))
#                 confidence = 'HIGH'
#             elif '📈' in category_status:  # Emerging categories
#                 base_forecast = current_sales * (1 + min(category_growth * 0.5, 0.2))
#                 confidence = 'MEDIUM'
#             elif '🔄' in category_status:  # Stable categories
#                 base_forecast = current_sales * 0.95  # Slight decline expected
#                 confidence = 'MEDIUM'
#             elif '⚠️' in category_status:  # Declining categories
#                 base_forecast = current_sales * 0.8
#                 confidence = 'LOW'
#             else:  # Phase out categories
#                 base_forecast = current_sales * 0.5
#                 confidence = 'VERY LOW'
            
#             # Risk adjustment
#             if product_risk == 'HIGH':
#                 base_forecast *= 0.8
#                 confidence = 'LOW' if confidence != 'VERY LOW' else 'VERY LOW'
#             elif product_risk == 'LOW':
#                 base_forecast *= 1.1
            
#             # Business rules
#             base_forecast = max(1, base_forecast)  # Minimum 1 unit
            
#             # Confidence ranges
#             confidence_ranges = {
#                 'VERY LOW': (0.3, 2.0),
#                 'LOW': (0.5, 1.8),
#                 'MEDIUM': (0.7, 1.5),
#                 'HIGH': (0.8, 1.3)
#             }
            
#             range_mult = confidence_ranges[confidence]
            
#             forecasts[product] = {
#                 'forecast': round(base_forecast),
#                 'confidence': confidence,
#                 'range_low': round(base_forecast * range_mult[0]),
#                 'range_high': round(base_forecast * range_mult[1]),
#                 'category': category,
#                 'category_status': category_status,
#                 'current_sales': current_sales,
#                 'product_risk': product_risk
#             }
        
#         self.forecasts = forecasts
#         return forecasts
    
#     def create_inventory_recommendations(self):
#         """Generate actionable inventory recommendations"""
#         print(f"\n📋 INVENTORY RECOMMENDATIONS:")
        
#         # Group by confidence and category status
#         high_priority = []
#         medium_priority = []
#         low_priority = []
#         phase_out = []
        
#         for product, forecast in self.forecasts.items():
#             item = {
#                 'product': product,
#                 'category': forecast['category'],
#                 'forecast': forecast['forecast'],
#                 'confidence': forecast['confidence'],
#                 'status': forecast['category_status']
#             }
            
#             if '🚀' in forecast['category_status'] and forecast['confidence'] in ['HIGH', 'MEDIUM']:
#                 high_priority.append(item)
#             elif '📈' in forecast['category_status'] or '🔄' in forecast['category_status']:
#                 medium_priority.append(item)
#             elif '⚠️' in forecast['category_status']:
#                 low_priority.append(item)
#             else:
#                 phase_out.append(item)
        
#         # Sort by forecast volume
#         high_priority.sort(key=lambda x: x['forecast'], reverse=True)
#         medium_priority.sort(key=lambda x: x['forecast'], reverse=True)
#         low_priority.sort(key=lambda x: x['forecast'], reverse=True)
        
#         recommendations = {
#             'HIGH_PRIORITY': high_priority[:20],  # Top 20 growth products
#             'MEDIUM_PRIORITY': medium_priority[:30],  # Top 30 stable products  
#             'LOW_PRIORITY': low_priority[:20],   # Top 20 declining products
#             'PHASE_OUT': phase_out[:10]          # Top 10 phase-out products
#         }
        
#         # Print recommendations
#         for priority, items in recommendations.items():
#             print(f"\n{priority.replace('_', ' ')} ({len(items)} products):")
#             print(f"{'Product':<15} {'Category':<20} {'Forecast':<10} {'Confidence':<10}")
#             print("-" * 65)
            
#             for item in items[:5]:  # Show top 5 in each category
#                 print(f"{item['product'][:14]:<15} {item['category'][:19]:<20} {item['forecast']:>8} {item['confidence']:<10}")
        
#         return recommendations
    
#     def generate_summary_report(self, target_year=2025):
#         """Generate executive summary"""
#         print(f"\n📊 EXECUTIVE SUMMARY FOR {target_year}:")
#         print("=" * 60)
        
#         total_forecast = sum(f['forecast'] for f in self.forecasts.values())
#         high_confidence = sum(f['forecast'] for f in self.forecasts.values() if f['confidence'] == 'HIGH')
        
#         # Category breakdown
#         category_totals = {}
#         for forecast in self.forecasts.values():
#             cat = forecast['category']
#             if cat not in category_totals:
#                 category_totals[cat] = 0
#             category_totals[cat] += forecast['forecast']
        
#         top_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        
#         print(f"📈 Total Forecasted Sales: {total_forecast:,} units")
#         print(f"🎯 High Confidence Forecast: {high_confidence:,} units ({high_confidence/total_forecast:.1%})")
        
#         print(f"\n🏆 Top 5 Categories by Forecast:")
#         for cat, volume in top_categories:
#             print(f"  {cat}: {volume:,} units")
        
#         # Risk assessment
#         risk_distribution = {}
#         for forecast in self.forecasts.values():
#             risk = forecast['product_risk']
#             if risk not in risk_distribution:
#                 risk_distribution[risk] = 0
#             risk_distribution[risk] += forecast['forecast']
        
#         print(f"\n⚠️ Risk Distribution:")
#         for risk, volume in risk_distribution.items():
#             print(f"  {risk} RISK: {volume:,} units ({volume/total_forecast:.1%})")
        
#         print(f"\n💡 KEY RECOMMENDATIONS:")
#         print("1. Focus inventory on Growth Star and Emerging categories")
#         print("2. Reduce stock levels for declining categories by 30-50%")
#         print("3. Implement frequent reordering for high-risk products")
#         print("4. Consider discontinuing phase-out products")
#         print("5. Monitor Boys Pant category (strong growth trend)")