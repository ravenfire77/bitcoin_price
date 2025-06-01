import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import mwclient
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class BitcoinPricePredictor:
    def __init__(self):
        self.btc_data = None
        self.wikipedia_data = None
        self.merged_data = None
        self.model = None
        self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def load_bitcoin_data(self, file_path=None):
        """Load Bitcoin price data from file or download from Yahoo Finance"""
        if file_path:
            print(f"Loading Bitcoin data from {file_path}...")
            self.btc_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        else:
            print("Downloading Bitcoin data from Yahoo Finance...")
            # Download last 3 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            
            btc = yf.download('BTC-USD', start=start_date, end=end_date)
            self.btc_data = btc
            # Save for future use
            self.btc_data.to_csv('btc.csv')
        
        print(f"Loaded {len(self.btc_data)} days of Bitcoin price data")
        return self.btc_data
    
    def create_price_features(self):
        """Create technical indicators and features from price data"""
        df = self.btc_data.copy()
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        df['MA_90'] = df['Close'].rolling(window=90).mean()
        
        # Moving average ratios
        df['MA_7_30_Ratio'] = df['MA_7'] / df['MA_30']
        df['MA_30_90_Ratio'] = df['MA_30'] / df['MA_90']
        
        # Volatility
        df['Volatility_7'] = df['Close'].rolling(window=7).std()
        df['Volatility_30'] = df['Close'].rolling(window=30).std()
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_7'] = df['Volume'].rolling(window=7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_7']
        
        # Price position within daily range
        df['High_Low_Ratio'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Momentum indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # Lag features
        for i in range(1, 8):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        self.btc_data = df
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fetch_wikipedia_data(self, save_path='wikipedia_edits.csv'):
        """Fetch Wikipedia edit data for Bitcoin page"""
        print("Fetching Wikipedia edit data...")
        
        site = mwclient.Site('en.wikipedia.org')
        page = site.pages['Bitcoin']
        
        edits_data = []
        
        # Get revisions (last 1000)
        revisions = list(page.revisions(limit=1000))
        
        for rev in revisions:
            edit_data = {
                'timestamp': rev['timestamp'],
                'user': rev.get('user', 'Anonymous'),
                'comment': rev.get('comment', ''),
                'size': rev.get('size', 0),
                'revid': rev['revid']
            }
            edits_data.append(edit_data)
        
        # Convert to DataFrame
        wiki_df = pd.DataFrame(edits_data)
        wiki_df['timestamp'] = pd.to_datetime(wiki_df['timestamp'])
        wiki_df['date'] = wiki_df['timestamp'].dt.date
        
        # Aggregate by date
        daily_edits = wiki_df.groupby('date').agg({
            'revid': 'count',  # Number of edits
            'size': ['mean', 'std', 'max'],  # Size statistics
            'comment': lambda x: ' '.join(x.fillna(''))  # Concatenate comments
        })
        
        daily_edits.columns = ['edit_count', 'avg_size', 'std_size', 'max_size', 'comments']
        daily_edits.index = pd.to_datetime(daily_edits.index)
        
        # Add sentiment analysis on comments
        print("Analyzing sentiment of edit comments...")
        sentiments = []
        for comments in daily_edits['comments']:
            if comments.strip():
                try:
                    # Truncate long comments for sentiment analysis
                    truncated = comments[:512]
                    sentiment = self.sentiment_analyzer(truncated)[0]
                    sentiments.append(1 if sentiment['label'] == 'POSITIVE' else 0)
                except:
                    sentiments.append(0.5)  # Neutral if error
            else:
                sentiments.append(0.5)  # Neutral if no comments
        
        daily_edits['sentiment_score'] = sentiments
        
        # Calculate rolling features
        daily_edits['edit_count_ma_7'] = daily_edits['edit_count'].rolling(window=7).mean()
        daily_edits['sentiment_ma_7'] = daily_edits['sentiment_score'].rolling(window=7).mean()
        
        # Save data
        daily_edits.to_csv(save_path)
        self.wikipedia_data = daily_edits
        
        print(f"Fetched {len(daily_edits)} days of Wikipedia edit data")
        return daily_edits
    
    def merge_data(self):
        """Merge Bitcoin price data with Wikipedia data"""
        print("Merging datasets...")
        
        # Ensure both datasets have datetime index
        if not isinstance(self.btc_data.index, pd.DatetimeIndex):
            self.btc_data.index = pd.to_datetime(self.btc_data.index)
        if not isinstance(self.wikipedia_data.index, pd.DatetimeIndex):
            self.wikipedia_data.index = pd.to_datetime(self.wikipedia_data.index)
        
        # Merge on date
        merged = self.btc_data.join(self.wikipedia_data, how='left')
        
        # Fill missing Wikipedia data
        wiki_columns = ['edit_count', 'avg_size', 'std_size', 'max_size', 
                       'sentiment_score', 'edit_count_ma_7', 'sentiment_ma_7']
        
        for col in wiki_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(method='ffill').fillna(0)
        
        # Drop rows with missing target or key features
        merged = merged.dropna(subset=['Price_Direction', 'Close', 'MA_7', 'MA_30'])
        
        self.merged_data = merged
        print(f"Merged dataset contains {len(merged)} rows")
        return merged
    
    def prepare_features(self):
        """Prepare features for model training"""
        feature_columns = [
            # Price features
            'Close', 'Volume', 'Price_Change', 'MA_7', 'MA_30', 'MA_90',
            'MA_7_30_Ratio', 'MA_30_90_Ratio', 'Volatility_7', 'Volatility_30',
            'Volume_Change', 'Volume_MA_7', 'Volume_Ratio', 'High_Low_Ratio', 'RSI',
            
            # Lag features
            *[f'Close_Lag_{i}' for i in range(1, 8)],
            *[f'Volume_Lag_{i}' for i in range(1, 8)],
            
            # Wikipedia features
            'edit_count', 'avg_size', 'std_size', 'max_size',
            'sentiment_score', 'edit_count_ma_7', 'sentiment_ma_7'
        ]
        
        # Filter available features
        available_features = [f for f in feature_columns if f in self.merged_data.columns]
        
        X = self.merged_data[available_features]
        y = self.merged_data['Price_Direction']
        
        return X, y, available_features
    
    def backtest_split(self, X, y, test_size=0.2):
        """Create train/test split maintaining temporal order"""
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        print("Training Random Forest model...")
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        print("Training XGBoost model...")
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.01,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False
        )
        
        xgb_model.fit(X_train, y_train)
        return xgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        predictions = model.predict(X_test)
        
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracy = (predictions == y_test).mean()
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }
    
    def run_full_pipeline(self, btc_file=None, wiki_file='wikipedia_edits.csv'):
        """Run the complete prediction pipeline"""
        print("Starting Bitcoin Price Prediction Pipeline...")
        print("="*50)
        
        # 1. Load Bitcoin data
        self.load_bitcoin_data(btc_file)
        
        # 2. Create price features
        self.create_price_features()
        
        # 3. Load or fetch Wikipedia data
        try:
            self.wikipedia_data = pd.read_csv(wiki_file, index_col=0, parse_dates=True)
            print(f"Loaded existing Wikipedia data from {wiki_file}")
        except:
            self.fetch_wikipedia_data(wiki_file)
        
        # 4. Merge datasets
        self.merge_data()
        
        # 5. Prepare features
        X, y, feature_names = self.prepare_features()
        print(f"\nUsing {len(feature_names)} features for prediction")
        
        # 6. Create train/test split
        X_train, X_test, y_train, y_test = self.backtest_split(X, y)
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # 7. Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_results = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # 8. Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_results = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        # 9. Store best model
        if xgb_results['f1'] > rf_results['f1']:
            self.model = xgb_model
            print("\nXGBoost selected as the best model")
        else:
            self.model = rf_model
            print("\nRandom Forest selected as the best model")
        
        return self.model, rf_results, xgb_results
    
    def predict_tomorrow(self):
        """Predict Bitcoin price direction for tomorrow"""
        if self.model is None:
            raise ValueError("Model not trained. Run the full pipeline first.")
        
        # Get the latest features
        X, _, _ = self.prepare_features()
        latest_features = X.iloc[-1:].values
        
        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0]
        
        direction = "INCREASE" if prediction == 1 else "DECREASE"
        confidence = probability[int(prediction)] * 100
        
        print(f"\nTomorrow's Bitcoin Price Prediction:")
        print(f"Direction: {direction}")
        print(f"Confidence: {confidence:.2f}%")
        
        return {
            'direction': direction,
            'confidence': confidence,
            'prediction': prediction
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = BitcoinPricePredictor()
    
    # Run full pipeline
    # Pass btc_file='btc.csv' if you have existing data
    model, rf_results, xgb_results = predictor.run_full_pipeline(btc_file='btc.csv')
    
    # Make prediction for tomorrow
    tomorrow_prediction = predictor.predict_tomorrow()
