#!/usr/bin/env python3
"""
Bitcoin Price Prediction Analysis
Complete pipeline with data loading, model training, and visualization
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class BitcoinPricePredictor:
    def __init__(self):
        self.data = None
        self.features = None
        self.model = None
        self.predictions = None
        
    def load_data(self, file_path=None):
        """Load Bitcoin data from Excel file or download from Yahoo Finance"""
        if file_path:
            try:
                # Try to load from Excel file
                print(f"Loading data from {file_path}...")
                self.data = pd.read_excel(file_path)
                
                # Standardize column names
                self.data.columns = [col.strip().title() for col in self.data.columns]
                
                # Ensure Date column is datetime
                if 'Date' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Date'])
                    self.data.set_index('Date', inplace=True)
                elif 'Timestamp' in self.data.columns:
                    self.data['Date'] = pd.to_datetime(self.data['Timestamp'])
                    self.data.set_index('Date', inplace=True)
                    
                print(f"Successfully loaded {len(self.data)} rows of data")
                print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
                
            except Exception as e:
                print(f"Error loading file: {e}")
                print("Falling back to Yahoo Finance download...")
                self.download_from_yahoo()
        else:
            self.download_from_yahoo()
            
        # Ensure we have required columns
        self.standardize_columns()
        return self.data
    
    def download_from_yahoo(self):
        """Download Bitcoin data from Yahoo Finance"""
        print("Downloading Bitcoin data from Yahoo Finance...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)  # 3 years of data
        
        ticker = "BTC-USD"
        self.data = yf.download(ticker, start=start_date, end=end_date)
        
        print(f"Downloaded {len(self.data)} days of Bitcoin data")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
    
    def standardize_columns(self):
        """Ensure we have standard column names"""
        # Common column mappings
        column_mappings = {
            'close': 'Close',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'adjusted close': 'Adj Close'
        }
        
        # Rename columns if needed
        self.data.columns = [column_mappings.get(col.lower(), col) for col in self.data.columns]
        
        # Ensure we have essential columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            print(f"Available columns: {self.data.columns.tolist()}")
    
    def create_features(self):
        """Create technical indicators and features for prediction"""
        print("Creating technical features...")
        
        df = self.data.copy()
        
        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price direction (target variable)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'SMA_{period}_Ratio'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential moving averages
        for period in [12, 26]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        df['Volatility_50'] = df['Returns'].rolling(window=50).std()
        
        # Price patterns
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Lag features
        for i in range(1, 11):
            df[f'Returns_Lag_{i}'] = df['Returns'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        # Store features
        self.features = df
        
        # Drop NaN values
        self.features = self.features.dropna()
        
        print(f"Created {len(self.features.columns)} features")
        print(f"Dataset size after feature engineering: {len(self.features)} rows")
        
        return self.features
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data_for_training(self):
        """Prepare features and target for model training"""
        # Define feature columns (exclude target and price columns)
        feature_columns = [col for col in self.features.columns 
                          if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close']]
        
        X = self.features[feature_columns]
        y = self.features['Target']
        
        # Split data - use last 20% for testing
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train Random Forest and XGBoost models"""
        results = {}
        
        # Random Forest
        print("\nTraining Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate Random Forest
        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        results['rf'] = {
            'model': rf_model,
            'predictions': rf_pred,
            'probabilities': rf_pred_proba,
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        
        # XGBoost
        print("Training XGBoost model...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            objective='binary:logistic',
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        
        # Evaluate XGBoost
        xgb_pred = xgb_model.predict(X_test)
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        results['xgb'] = {
            'model': xgb_model,
            'predictions': xgb_pred,
            'probabilities': xgb_pred_proba,
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred),
            'recall': recall_score(y_test, xgb_pred),
            'f1': f1_score(y_test, xgb_pred)
        }
        
        # Print results
        print("\n" + "="*50)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*50)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()} Model:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        # Select best model based on F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
        self.model = results[best_model_name]['model']
        self.predictions = results[best_model_name]
        
        print(f"\nBest model: {best_model_name.upper()} (F1 Score: {results[best_model_name]['f1']:.4f})")
        
        return results, X_test, y_test
    
    def create_prediction_chart(self, X_test, y_test, results):
        """Create comprehensive prediction visualization"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Get test period data
        test_dates = self.features.index[-len(X_test):]
        test_prices = self.features['Close'].iloc[-len(X_test):]
        
        # 1. Price Chart with Predictions
        ax1 = plt.subplot(3, 2, 1)
        
        # Plot actual prices
        ax1.plot(test_dates, test_prices, 'k-', label='Actual Price', linewidth=2)
        
        # Add prediction markers
        rf_predictions = results['rf']['predictions']
        xgb_predictions = results['xgb']['predictions']
        
        # Mark correct predictions
        rf_correct = rf_predictions == y_test
        xgb_correct = xgb_predictions == y_test
        
        ax1.scatter(test_dates[rf_correct & (rf_predictions == 1)], 
                   test_prices[rf_correct & (rf_predictions == 1)], 
                   color='green', marker='^', s=50, alpha=0.7, label='RF Correct Up')
        ax1.scatter(test_dates[rf_correct & (rf_predictions == 0)], 
                   test_prices[rf_correct & (rf_predictions == 0)], 
                   color='red', marker='v', s=50, alpha=0.7, label='RF Correct Down')
        
        ax1.set_title('Bitcoin Price with Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Prediction Accuracy Over Time
        ax2 = plt.subplot(3, 2, 2)
        
        # Calculate rolling accuracy
        window = 50
        rf_rolling_acc = pd.Series(rf_correct).rolling(window).mean()
        xgb_rolling_acc = pd.Series(xgb_correct).rolling(window).mean()
        
        ax2.plot(test_dates, rf_rolling_acc, label='Random Forest', linewidth=2)
        ax2.plot(test_dates, xgb_rolling_acc, label='XGBoost', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        
        ax2.set_title(f'{window}-Day Rolling Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Confusion Matrices
        ax3 = plt.subplot(3, 2, 3)
        cm_rf = confusion_matrix(y_test, rf_predictions)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax3.set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        ax4 = plt.subplot(3, 2, 4)
        cm_xgb = confusion_matrix(y_test, xgb_predictions)
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax4.set_title('XGBoost Confusion Matrix', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 4. Feature Importance (Top 15)
        ax5 = plt.subplot(3, 2, 5)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        ax5.barh(feature_importance['feature'], feature_importance['importance'])
        ax5.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Importance')
        
        # 5. Prediction Confidence Distribution
        ax6 = plt.subplot(3, 2, 6)
        
        # Plot probability distributions
        ax6.hist(results['rf']['probabilities'], bins=30, alpha=0.5, label='Random Forest', density=True)
        ax6.hist(results['xgb']['probabilities'], bins=30, alpha=0.5, label='XGBoost', density=True)
        ax6.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        
        ax6.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Predicted Probability (Up)')
        ax6.set_ylabel('Density')
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig('bitcoin_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create summary statistics
        self.create_summary_report(results, X_test, y_test)
    
    def create_summary_report(self, results, X_test, y_test):
        """Create a summary report of the analysis"""
        print("\n" + "="*60)
        print("BITCOIN PRICE PREDICTION SUMMARY REPORT")
        print("="*60)
        
        print(f"\nData Summary:")
        print(f"- Total data points: {len(self.data)}")
        print(f"- Date range: {self.data.index.min().strftime('%Y-%m-%d')} to {self.data.index.max().strftime('%Y-%m-%d')}")
        print(f"- Features created: {len(X_test.columns)}")
        print(f"- Test period: {len(X_test)} days")
        
        print(f"\nPrice Statistics (Test Period):")
        test_prices = self.features['Close'].iloc[-len(X_test):]
        print(f"- Starting price: ${test_prices.iloc[0]:,.2f}")
        print(f"- Ending price: ${test_prices.iloc[-1]:,.2f}")
        print(f"- Price change: {((test_prices.iloc[-1] / test_prices.iloc[0]) - 1) * 100:.2f}%")
        print(f"- Volatility (std): ${test_prices.std():,.2f}")
        
        print(f"\nModel Performance:")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"- Accuracy: {metrics['accuracy']:.2%}")
            print(f"- Precision: {metrics['precision']:.2%} (When predicting UP, correct {metrics['precision']:.2%} of time)")
            print(f"- Recall: {metrics['recall']:.2%} (Caught {metrics['recall']:.2%} of actual UP movements)")
            
            # Calculate profit if following predictions
            predictions = metrics['predictions']
            returns = self.features['Returns'].iloc[-len(X_test):].values
            strategy_returns = returns[1:] * predictions[:-1]  # Buy when predict up
            
            total_return = (1 + strategy_returns).prod() - 1
            print(f"- Strategy return: {total_return:.2%}")
            
        print(f"\nNext Day Prediction:")
        latest_features = X_test.iloc[-1:].values
        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0, 1]
        
        print(f"- Direction: {'UP ↑' if prediction == 1 else 'DOWN ↓'}")
        print(f"- Confidence: {probability:.2%}")
        
        print("\n" + "="*60)
    
    def run_complete_analysis(self, file_path=None):
        """Run the complete analysis pipeline"""
        print("Starting Bitcoin Price Prediction Analysis...")
        print("="*60)
        
        # 1. Load data
        self.load_data(file_path)
        
        # 2. Create features
        self.create_features()
        
        # 3. Prepare data
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_data_for_training()
        
        # 4. Train models
        results, X_test_data, y_test_data = self.train_models(X_train, X_test, y_train, y_test)
        
        # 5. Create visualizations
        self.create_prediction_chart(X_test_data, y_test_data, results)
        
        return self.model, results

# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = BitcoinPricePredictor()
    
    # Run analysis
    # If you have the Excel file, use: file_path='btc_USD.xlsx'
    # Otherwise, it will download from Yahoo Finance
    
    try:
        # Try to use the uploaded Excel file first
        model, results = predictor.run_complete_analysis(file_path='btc_USD.xlsx')
    except:
        # If file not found, download from Yahoo Finance
        print("Excel file not found, downloading fresh data from Yahoo Finance...")
        model, results = predictor.run_complete_analysis()
    
    print("\nAnalysis complete! Check 'bitcoin_prediction_analysis.png' for visualizations.")
