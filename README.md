
# AI Demand Forecasting System 🚀

A sophisticated AI-powered demand forecasting system designed specifically for new products without sales history. Built with React frontend and Flask backend, utilizing ensemble machine learning models for accurate business predictions.

## 🎯 Overview

This system helps businesses forecast demand for new products by leveraging historical sales patterns, product attributes, and advanced machine learning algorithms. Perfect for fashion retail, e-commerce, and any business launching new products.

### Key Features

- **🤖 AI Ensemble Models**: Random Forest + XGBoost + LightGBM for maximum accuracy
- **📊 Real Target Training**: Uses actual future sales instead of synthetic targets
- **⏱️ Time-Series Validation**: Proper historical validation with accuracy metrics
- **🎨 Modern UI**: Beautiful React interface with loading states and progress tracking
- **📈 Business Intelligence**: Category-based insights, confidence scoring, and risk assessment
- **🔄 Flexible Workflow**: Optional validation step with bypass options
- **📦 Multi-Product Support**: Batch forecasting for efficient planning

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **LightGBM** - Gradient boosting
- **NumPy** - Numerical computing

### Frontend
- **React 18+**
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Modern JavaScript (ES6+)**

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher**
- **Node.js 16 or higher**
- **npm or yarn**
- **Git**

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-demand-forecasting.git
cd ai-demand-forecasting




# Navigate to backend directory
cd trend-tactix-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install flask flask-cors pandas numpy scikit-learn xgboost lightgbm

# Or install from requirements.txt (if available)
pip install -r requirements.txt

pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install pandas==2.1.1
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install xgboost==1.7.6
pip install lightgbm==4.1.0


# Navigate to frontend directory
cd trend-tactix

# Install dependencies
npm install


npm install react react-dom
npm install @tailwindcss/forms
npm install lucide-react
npm install @headlessui/react


# Navigate to backend directory
cd trend-tactix-backend

# Activate virtual environment (if not already active)
# Windows:
venv\Scripts\activate
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Start Flask server
python api_routes.py


# RUN FRONTEND
# Open new terminal and navigate to frontend
cd trend-tactix

# Start React development server
npm start / npm run dev