import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class BitcoinPredictionExperiment:
    def __init__(self):
        self.data = None
        self.features = None
        self.results = {}
        
    def run_complete_experiment(self, file_path='btc_USD.xlsx'):
        """Run the complete Bitcoin prediction experiment"""
        print("="*70)
        print("BITCOIN PRICE PREDICTION EXPERIMENT")
        print("="*70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data(file_path)
        
        # Step 2: Create technical indicators
        self.create_technical_indicators()
        
        # Step 3: Prepare features for ML
        X_train, X_test, y_train, y_test = self.prepare_ml_features()
        
        # Step 4: Train and evaluate models
        self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Generate prediction analysis
        self.create_prediction_analysis(X_test, y_test)
        
        # Step 6: Create experiment report
        self.generate_experiment_report()
        
        print(f"\n✅ Experiment Complete!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results
    
    def load_and_prepare_data(self, file_path):
        """Load Bitcoin data and prepare for analysis"""
        print("\n📊 STEP 1: DATA LOADING AND PREPARATION")
        print("-"*50)
        
        # Load data
        self.data = pd.read_excel(file_path, parse_dates=['Date'], dayfirst=True)
        self.data.columns = self.data.columns.str.replace('\xa0', ' ')
        self.data = self.data.sort_values('Date')
        self.data.set_index('Date', inplace=True)
        
        print(f"✓ Loaded {len(self.data):,} days of Bitcoin data")
        print(f"✓ Date range: {self.data.index.min().strftime('%d/%m/%Y')} to {self.data.index.max().strftime('%d/%m/%Y')}")
        print(f"✓ Price range: ${self.data['Close'].min():,.2f} to ${self.data['Close'].max():,.2f}")
        
        # Basic statistics
        self.results['data_stats'] = {
            'total_days': len(self.data),
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
            'min_price': self.data['Close'].min(),
            'max_price': self.data['Close'].max(),
            'avg_price': self.data['Close'].mean(),
            'total_return': (self.data['Close'].iloc[-1] / self.data['Close'].iloc[0] - 1) * 100
        }
        
        print(f"✓ Total return: {self.results['data_stats']['total_return']:.1f}%")
    
    def create_technical_indicators(self):
        """Create all technical indicators"""
        print("\n🔧 STEP 2: TECHNICAL INDICATORS CREATION")
        print("-"*50)
        
        # Price changes
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Target variable (1 if price goes up tomorrow, 0 if down)
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Moving averages
        for period in [7, 14, 30, 50, 100, 200]:
            self.data[f'SMA_{period}'] = self.data['Close'].rolling(window=period).mean()
            self.data[f'Price_to_SMA_{period}'] = self.data['Close'] / self.data[f'SMA_{period}']
        
        # EMA
        for period in [12, 26]:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # RSI
        delta = self.data['Close'].diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        rs = avg_gains / avg_losses
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
        self.data['BB_Width'] = self.data['BB_Upper'] - self.data['BB_Lower']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / self.data['BB_Width']
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # Volatility
        self.data['Volatility_20'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
        self.data['Volatility_50'] = self.data['Returns'].rolling(window=50).std() * np.sqrt(252)
        
        # Price patterns
        self.data['High_Low_Range'] = self.data['High'] - self.data['Low']
        self.data['Close_Position'] = (self.data['Close'] - self.data['Low']) / self.data['High_Low_Range']
        
        # Momentum
        self.data['ROC_10'] = ((self.data['Close'] - self.data['Close'].shift(10)) / self.data['Close'].shift(10)) * 100
        
        # Lag features
        for i in range(1, 6):
            self.data[f'Returns_Lag_{i}'] = self.data['Returns'].shift(i)
            self.data[f'Volume_Ratio_Lag_{i}'] = self.data['Volume_Ratio'].shift(i)
        
        # Feature count
        feature_count = len(self.data.columns) - 7  # Subtract original columns
        print(f"✓ Created {feature_count} technical indicators")
        
        # Store enhanced data
        self.features = self.data.copy()
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("\n🤖 STEP 3: MACHINE LEARNING PREPARATION")
        print("-"*50)
        
        # Drop NaN values
        ml_data = self.features.dropna()
        
        # Define feature columns (exclude price columns and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Target']
        feature_cols = [col for col in ml_data.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = ml_data[feature_cols]
        y = ml_data['Target']
        
        # Train-test split (80-20, time-based)
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"✓ Features prepared: {len(feature_cols)} features")
        print(f"✓ Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"✓ Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"✓ Class balance - Train: {y_train.mean():.1%} UP days")
        print(f"✓ Class balance - Test: {y_test.mean():.1%} UP days")
        
        # Store test dates and prices for later analysis
        self.test_dates = ml_data.index[split_idx:]
        self.test_prices = ml_data['Close'].iloc[split_idx:]
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train Random Forest and XGBoost models"""
        print("\n🎯 STEP 4: MODEL TRAINING AND EVALUATION")
        print("-"*50)
        
        models = {}
        
        # Train Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01,
            random_state=42
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        # Evaluate models
        print("\n" + "="*50)
        print("MODEL PERFORMANCE RESULTS")
        print("="*50)
        
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate trading performance
            test_returns = self.test_prices.pct_change().iloc[1:]
            strategy_returns = test_returns[y_pred[:-1] == 1]  # Only trade on UP predictions
            
            total_trades = sum(y_pred[:-1] == 1)
            winning_trades = sum((test_returns > 0) & (y_pred[:-1] == 1))
            
            if total_trades > 0:
                win_rate = winning_trades / total_trades
                avg_win = strategy_returns[strategy_returns > 0].mean() if len(strategy_returns[strategy_returns > 0]) > 0 else 0
                avg_loss = strategy_returns[strategy_returns < 0].mean() if len(strategy_returns[strategy_returns < 0]) > 0 else 0
                cumulative_return = (1 + strategy_returns).prod() - 1
            else:
                win_rate = avg_win = avg_loss = cumulative_return = 0
            
            # Store results
            self.results[name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'cumulative_return': cumulative_return
            }
            
            # Print results
            print(f"\n{name.upper()} RESULTS:")
            print(f"├─ Accuracy: {accuracy:.2%}")
            print(f"├─ Precision: {precision:.2%} (When predicting UP, correct {precision:.1%} of time)")
            print(f"├─ Recall: {recall:.2%} (Caught {recall:.1%} of actual UP days)")
            print(f"├─ F1 Score: {f1:.2%}")
            print(f"├─ Total Trades: {total_trades}")
            print(f"├─ Win Rate: {win_rate:.2%}")
            print(f"├─ Avg Win: {avg_win:.2%}")
            print(f"├─ Avg Loss: {avg_loss:.2%}")
            print(f"└─ Strategy Return: {cumulative_return:.2%}")
        
        # Select best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        self.best_model = models[best_model]
        self.best_model_name = best_model
        
        print(f"\n🏆 Best Model: {best_model} (F1 Score: {self.results[best_model]['f1']:.2%})")
        
        # Store test data for visualization
        self.X_test = X_test
        self.y_test = y_test
    
    def create_prediction_analysis(self, X_test, y_test):
        """Create detailed prediction analysis and visualizations"""
        print("\n📈 STEP 5: PREDICTION ANALYSIS AND VISUALIZATION")
        print("-"*50)
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # Use best model predictions
        best_results = self.results[self.best_model_name]
        predictions = best_results['predictions']
        probabilities = best_results['probabilities']
        
        # 1. Price chart with predictions
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(self.test_dates, self.test_prices, 'k-', linewidth=1, alpha=0.8, label='Bitcoin Price')
        
        # Mark predictions
        correct_up = (predictions == 1) & (y_test == 1)
        correct_down = (predictions == 0) & (y_test == 0)
        wrong_up = (predictions == 1) & (y_test == 0)
        wrong_down = (predictions == 0) & (y_test == 1)
        
        ax1.scatter(self.test_dates[correct_up], self.test_prices[correct_up], 
                   color='green', marker='^', s=30, alpha=0.7, label='Correct UP')
        ax1.scatter(self.test_dates[correct_down], self.test_prices[correct_down], 
                   color='darkgreen', marker='v', s=30, alpha=0.7, label='Correct DOWN')
        ax1.scatter(self.test_dates[wrong_up], self.test_prices[wrong_up], 
                   color='red', marker='^', s=30, alpha=0.7, label='Wrong UP')
        ax1.scatter(self.test_dates[wrong_down], self.test_prices[wrong_down], 
                   color='darkred', marker='v', s=30, alpha=0.7, label='Wrong DOWN')
        
        ax1.set_title(f'Bitcoin Price with {self.best_model_name} Predictions', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative returns comparison
        ax2 = plt.subplot(4, 2, 2)
        
        # Calculate cumulative returns
        buy_hold_returns = (1 + self.test_prices.pct_change()).cumprod()
        strategy_returns = []
        capital = 1
        
        for i in range(len(predictions) - 1):
            if predictions[i] == 1:  # Predict UP, so buy
                capital *= (1 + self.test_prices.pct_change().iloc[i + 1])
            strategy_returns.append(capital)
        
        ax2.plot(self.test_dates[1:], buy_hold_returns[1:], label='Buy & Hold', linewidth=2)
        ax2.plot(self.test_dates[1:-1], strategy_returns, label=f'{self.best_model_name} Strategy', linewidth=2)
        ax2.set_title('Cumulative Returns: Strategy vs Buy & Hold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Prediction confidence distribution
        ax3 = plt.subplot(4, 2, 3)
        ax3.hist(probabilities[y_test == 1], bins=30, alpha=0.6, label='Actual UP days', density=True)
        ax3.hist(probabilities[y_test == 0], bins=30, alpha=0.6, label='Actual DOWN days', density=True)
        ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
        ax3.set_title('Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted Probability of UP')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        ax4 = plt.subplot(4, 2, 4)
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
        ax4.set_title(f'{self.best_model_name} Confusion Matrix', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. Feature Importance (Top 15)
        ax5 = plt.subplot(4, 2, 5)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        ax5.barh(feature_importance['feature'], feature_importance['importance'])
        ax5.set_title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Importance')
        ax5.grid(True, alpha=0.3)
        
        # 6. Rolling accuracy
        ax6 = plt.subplot(4, 2, 6)
        window = 50
        rolling_correct = pd.Series(predictions == y_test).rolling(window).mean()
        ax6.plot(self.test_dates, rolling_correct, linewidth=2)
        ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax6.fill_between(self.test_dates, 0.5, rolling_correct, 
                        where=rolling_correct > 0.5, alpha=0.3, color='green', label='Above 50%')
        ax6.fill_between(self.test_dates, rolling_correct, 0.5, 
                        where=rolling_correct <= 0.5, alpha=0.3, color='red', label='Below 50%')
        ax6.set_title(f'{window}-Day Rolling Accuracy', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Accuracy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Monthly performance
        ax7 = plt.subplot(4, 2, 7)
        monthly_data = pd.DataFrame({
            'date': self.test_dates[:-1],
            'actual': y_test.values[:-1],
            'predicted': predictions[:-1],
            'returns': self.test_prices.pct_change().values[1:]
        })
        monthly_data['month'] = monthly_data['date'].dt.to_period('M')
        
        monthly_perf = monthly_data.groupby('month').apply(
            lambda x: ((1 + x['returns'][x['predicted'] == 1]).prod() - 1) * 100
        )
        
        ax7.bar(range(len(monthly_perf)), monthly_perf.values, 
                color=['green' if x > 0 else 'red' for x in monthly_perf.values])
        ax7.set_title('Monthly Strategy Performance', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Return (%)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Model comparison
        ax8 = plt.subplot(4, 2, 8)
        models_comp = pd.DataFrame({
            'Model': list(self.results.keys()),
            'F1 Score': [self.results[m]['f1'] for m in self.results.keys()],
            'Strategy Return': [self.results[m]['cumulative_return'] for m in self.results.keys()]
        })
        
        x = np.arange(len(models_comp))
        width = 0.35
        
        ax8_2 = ax8.twinx()
        bars1 = ax8.bar(x - width/2, models_comp['F1 Score'], width, label='F1 Score', color='skyblue')
        bars2 = ax8_2.bar(x + width/2, models_comp['Strategy Return'] * 100, width, label='Strategy Return (%)', color='lightgreen')
        
        ax8.set_xlabel('Model')
        ax8.set_ylabel('F1 Score')
        ax8_2.set_ylabel('Strategy Return (%)')
        ax8.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(models_comp['Model'])
        ax8.legend(loc='upper left')
        ax8_2.legend(loc='upper right')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bitcoin_prediction_experiment_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Created comprehensive prediction analysis")
        print("✓ Saved visualization to 'bitcoin_prediction_experiment_results.png'")
    
    def generate_experiment_report(self):
        """Generate final experiment report"""
        print("\n📊 STEP 6: EXPERIMENT SUMMARY REPORT")
        print("="*70)
        
        # Data summary
        print("\n🔹 DATA SUMMARY:")
        print(f"   Total Days: {self.results['data_stats']['total_days']:,}")
        print(f"   Date Range: {self.results['data_stats']['start_date'].strftime('%d/%m/%Y')} to {self.results['data_stats']['end_date'].strftime('%d/%m/%Y')}")
        print(f"   Price Range: ${self.results['data_stats']['min_price']:,.2f} to ${self.results['data_stats']['max_price']:,.2f}")
        print(f"   Bitcoin Growth: {self.results['data_stats']['total_return']:.1f}%")
        
        # Model comparison
        print("\n🔹 MODEL COMPARISON:")
        print("   " + "-"*60)
        print(f"   {'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Return':<10}")
        print("   " + "-"*60)
        
        for model_name in ['Random Forest', 'XGBoost']:
            r = self.results[model_name]
            print(f"   {model_name:<15} {r['accuracy']:<10.2%} {r['precision']:<10.2%} {r['recall']:<10.2%} {r['f1']:<10.2%} {r['cumulative_return']:<10.2%}")
        
        # Best model details
        best = self.results[self.best_model_name]
        print(f"\n🔹 BEST MODEL: {self.best_model_name}")
        print(f"   ├─ Prediction Accuracy: {best['accuracy']:.2%}")
        print(f"   ├─ Total Trading Signals: {best['total_trades']}")
        print(f"   ├─ Win Rate: {best['win_rate']:.2%}")
        print(f"   ├─ Average Winning Trade: +{best['avg_win']:.2%}")
        print(f"   ├─ Average Losing Trade: {best['avg_loss']:.2%}")
        print(f"   └─ Total Strategy Return: {best['cumulative_return']:.2%}")
        
        # Key insights
        print("\n🔹 KEY INSIGHTS:")
        
        # Calculate some insights
        buy_hold_return = (self.test_prices.iloc[-1] / self.test_prices.iloc[0] - 1)
        outperformance = best['cumulative_return'] - buy_hold_return
        
        print(f"   • Model achieved {best['accuracy']:.1%} accuracy in predicting daily price direction")
        print(f"   • Strategy return: {best['cumulative_return']:.1%} vs Buy & Hold: {buy_hold_return:.1%}")
        print(f"   • Outperformance: {outperformance:+.1%}")
        print(f"   • The model is {'outperforming' if outperformance > 0 else 'underperforming'} buy & hold strategy")
        
        # Feature importance insights
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(5)
        print(f"\n   • Top 5 Most Important Features:")
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"     {idx}. {row['feature']} ({row['importance']:.3f})")
        
        # Trading strategy insights
        if best['total_trades'] > 0:
            avg_trades_per_month = best['total_trades'] / (len(self.test_dates) / 30)
            print(f"\n   • Average trades per month: {avg_trades_per_month:.1f}")
            print(f"   • Risk/Reward Ratio: {abs(best['avg_win'] / best['avg_loss']):.2f}:1")
        
        # Tomorrow's prediction
        print("\n🔹 TOMORROW'S PREDICTION:")
        latest_features = self.X_test.iloc[-1:].values
        tomorrow_pred = self.best_model.predict(latest_features)[0]
        tomorrow_prob = self.best_model.predict_proba(latest_features)[0, 1]
        
        print(f"   Direction: {'📈 UP' if tomorrow_pred == 1 else '📉 DOWN'}")
        print(f"   Confidence: {tomorrow_prob:.1%}")
        print(f"   Current Price: ${self.test_prices.iloc[-1]:,.2f}")
        
        print("\n" + "="*70)
        print("✅ EXPERIMENT COMPLETE - All results saved!")
        print("="*70)

# Run the complete experiment
if __name__ == "__main__":
    experiment = BitcoinPredictionExperiment()
    results = experiment.run_complete_experiment('btc_USD.xlsx')
