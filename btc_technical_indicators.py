import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class TechnicalIndicators:
    def __init__(self):
        self.data = None
        self.indicators = None
    
    def load_bitcoin_data(self, file_path='btc_USD.xlsx'):
        """Load Bitcoin data with proper date parsing"""
        print("Loading Bitcoin data...")
        
        # Load with DD/MM/YYYY format
        df = pd.read_excel(file_path, parse_dates=['Date'], dayfirst=True)
        
        # Fix column names with non-breaking spaces
        df.columns = df.columns.str.replace('\xa0', ' ')
        
        # Sort by date (ascending) for proper time series analysis
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        print(f"✓ Loaded {len(df)} days of data")
        print(f"✓ Date range: {df.index.min().strftime('%d/%m/%Y')} to {df.index.max().strftime('%d/%m/%Y')}")
        
        self.data = df
        return df
    
    def calculate_sma(self, column='Close', periods=[7, 14, 30, 50, 100, 200]):
        """Calculate Simple Moving Averages"""
        print("\nCalculating Simple Moving Averages...")
        
        for period in periods:
            self.data[f'SMA_{period}'] = self.data[column].rolling(window=period).mean()
            
            # Calculate price position relative to SMA
            self.data[f'Price_to_SMA_{period}'] = self.data[column] / self.data[f'SMA_{period}']
        
        print(f"✓ Calculated {len(periods)} SMAs")
    
    def calculate_ema(self, column='Close', periods=[12, 26, 50]):
        """Calculate Exponential Moving Averages"""
        print("Calculating Exponential Moving Averages...")
        
        for period in periods:
            self.data[f'EMA_{period}'] = self.data[column].ewm(span=period, adjust=False).mean()
        
        print(f"✓ Calculated {len(periods)} EMAs")
    
    def calculate_macd(self):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        print("Calculating MACD...")
        
        # MACD Line: 12-day EMA - 26-day EMA
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        
        # Signal Line: 9-day EMA of MACD
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        
        # MACD Histogram
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        print("✓ Calculated MACD indicators")
    
    def calculate_rsi(self, column='Close', period=14):
        """Calculate Relative Strength Index"""
        print("Calculating RSI...")
        
        # Calculate price changes
        delta = self.data[column].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        self.data['RSI'] = rsi
        
        # Add RSI levels
        self.data['RSI_Oversold'] = (rsi < 30).astype(int)
        self.data['RSI_Overbought'] = (rsi > 70).astype(int)
        
        print("✓ Calculated RSI")
    
    def calculate_bollinger_bands(self, column='Close', period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        print("Calculating Bollinger Bands...")
        
        # Middle Band = 20-day SMA
        self.data['BB_Middle'] = self.data[column].rolling(window=period).mean()
        
        # Calculate standard deviation
        rolling_std = self.data[column].rolling(window=period).std()
        
        # Upper and Lower Bands
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * std_dev)
        
        # Band width and position
        self.data['BB_Width'] = self.data['BB_Upper'] - self.data['BB_Lower']
        self.data['BB_Position'] = (self.data[column] - self.data['BB_Lower']) / self.data['BB_Width']
        
        # Signals
        self.data['BB_Squeeze'] = (self.data['BB_Width'] < self.data['BB_Width'].rolling(window=120).mean() * 0.75).astype(int)
        
        print("✓ Calculated Bollinger Bands")
    
    def calculate_volume_indicators(self):
        """Calculate volume-based indicators"""
        print("Calculating Volume indicators...")
        
        # Volume moving averages
        self.data['Volume_SMA_10'] = self.data['Volume'].rolling(window=10).mean()
        self.data['Volume_SMA_30'] = self.data['Volume'].rolling(window=30).mean()
        
        # Volume ratio
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA_10']
        
        # Price-Volume trend
        self.data['PV'] = self.data['Close'] * self.data['Volume']
        self.data['PV_SMA'] = self.data['PV'].rolling(window=10).mean()
        
        # On-Balance Volume (OBV)
        self.data['OBV'] = (self.data['Volume'] * (~self.data['Close'].diff().le(0) * 2 - 1)).cumsum()
        
        # Volume-Weighted Average Price (VWAP)
        self.data['VWAP'] = (self.data['PV'].rolling(window=14).sum() / 
                             self.data['Volume'].rolling(window=14).sum())
        
        print("✓ Calculated Volume indicators")
    
    def calculate_volatility_indicators(self):
        """Calculate volatility indicators"""
        print("Calculating Volatility indicators...")
        
        # Daily returns
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        
        # Historical volatility (different periods)
        for period in [7, 14, 30, 60]:
            self.data[f'Volatility_{period}'] = self.data['Returns'].rolling(window=period).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=14).mean()
        
        # Normalized ATR
        self.data['ATR_Percent'] = (self.data['ATR'] / self.data['Close']) * 100
        
        print("✓ Calculated Volatility indicators")
    
    def calculate_price_patterns(self):
        """Calculate price pattern indicators"""
        print("Calculating Price patterns...")
        
        # Candlestick patterns
        self.data['Body'] = self.data['Close'] - self.data['Open']
        self.data['Body_Percent'] = (self.data['Body'] / self.data['Open']) * 100
        self.data['Shadow_Upper'] = self.data['High'] - self.data[['Open', 'Close']].max(axis=1)
        self.data['Shadow_Lower'] = self.data[['Open', 'Close']].min(axis=1) - self.data['Low']
        
        # Price position within daily range
        self.data['High_Low_Range'] = self.data['High'] - self.data['Low']
        self.data['Close_Position'] = (self.data['Close'] - self.data['Low']) / self.data['High_Low_Range']
        
        # Trend strength
        self.data['Trend_7'] = (self.data['Close'] - self.data['Close'].shift(7)) / self.data['Close'].shift(7)
        self.data['Trend_30'] = (self.data['Close'] - self.data['Close'].shift(30)) / self.data['Close'].shift(30)
        
        # Support and Resistance levels
        self.data['Resistance_20'] = self.data['High'].rolling(window=20).max()
        self.data['Support_20'] = self.data['Low'].rolling(window=20).min()
        self.data['SR_Position'] = (self.data['Close'] - self.data['Support_20']) / (self.data['Resistance_20'] - self.data['Support_20'])
        
        print("✓ Calculated Price patterns")
    
    def calculate_momentum_indicators(self):
        """Calculate momentum indicators"""
        print("Calculating Momentum indicators...")
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            self.data[f'ROC_{period}'] = ((self.data['Close'] - self.data['Close'].shift(period)) / 
                                          self.data['Close'].shift(period)) * 100
        
        # Stochastic Oscillator
        period = 14
        low_min = self.data['Low'].rolling(window=period).min()
        high_max = self.data['High'].rolling(window=period).max()
        
        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min))
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=3).mean()
        
        # Money Flow Index (MFI)
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        money_flow = typical_price * self.data['Volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        self.data['MFI'] = mfi
        
        print("✓ Calculated Momentum indicators")
    
    def create_lag_features(self):
        """Create lag features for time series prediction"""
        print("Creating Lag features...")
        
        # Price lags
        for i in range(1, 11):
            self.data[f'Close_Lag_{i}'] = self.data['Close'].shift(i)
            self.data[f'Returns_Lag_{i}'] = self.data['Returns'].shift(i)
        
        # Volume lags
        for i in range(1, 6):
            self.data[f'Volume_Lag_{i}'] = self.data['Volume'].shift(i)
            self.data[f'Volume_Ratio_Lag_{i}'] = self.data['Volume_Ratio'].shift(i)
        
        # Indicator lags
        self.data['RSI_Lag_1'] = self.data['RSI'].shift(1)
        self.data['MACD_Lag_1'] = self.data['MACD'].shift(1)
        
        print("✓ Created Lag features")
    
    def create_target_variable(self):
        """Create target variable for prediction"""
        print("Creating Target variable...")
        
        # Binary target: 1 if price goes up tomorrow, 0 if down
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        
        # Additional targets for analysis
        self.data['Target_Return'] = self.data['Close'].shift(-1) / self.data['Close'] - 1
        self.data['Target_5Day'] = (self.data['Close'].shift(-5) > self.data['Close']).astype(int)
        
        print("✓ Created Target variables")
    
    def create_all_indicators(self):
        """Create all technical indicators"""
        print("\n" + "="*50)
        print("CREATING ALL TECHNICAL INDICATORS")
        print("="*50)
        
        # Calculate all indicators
        self.calculate_sma()
        self.calculate_ema()
        self.calculate_macd()
        self.calculate_rsi()
        self.calculate_bollinger_bands()
        self.calculate_volume_indicators()
        self.calculate_volatility_indicators()
        self.calculate_price_patterns()
        self.calculate_momentum_indicators()
        self.create_lag_features()
        self.create_target_variable()
        
        # Store enhanced data
        self.indicators = self.data.copy()
        
        print(f"\n✓ Created {len(self.data.columns) - 7} technical indicators")
        print(f"✓ Total features: {len(self.data.columns)}")
        
        return self.indicators
    
    def visualize_indicators(self):
        """Create visualization of key indicators"""
        print("\nCreating indicator visualizations...")
        
        # Select recent data for clearer visualization
        recent_data = self.data.iloc[-365:]  # Last year
        
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
        fig.suptitle('Bitcoin Technical Indicators Dashboard', fontsize=16, y=0.995)
        
        # 1. Price with Moving Averages
        ax1 = axes[0]
        ax1.plot(recent_data.index, recent_data['Close'], label='Close Price', color='black', linewidth=2)
        ax1.plot(recent_data.index, recent_data['SMA_50'], label='SMA 50', alpha=0.7)
        ax1.plot(recent_data.index, recent_data['SMA_200'], label='SMA 200', alpha=0.7)
        ax1.fill_between(recent_data.index, recent_data['BB_Upper'], recent_data['BB_Lower'], 
                        alpha=0.2, label='Bollinger Bands')
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Price with Moving Averages and Bollinger Bands')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Volume
        ax2 = axes[1]
        colors = ['green' if x > 0 else 'red' for x in recent_data['Returns']]
        ax2.bar(recent_data.index, recent_data['Volume'], color=colors, alpha=0.7)
        ax2.plot(recent_data.index, recent_data['Volume_SMA_30'], color='blue', label='30-day Avg')
        ax2.set_ylabel('Volume')
        ax2.set_title('Trading Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3 = axes[2]
        ax3.plot(recent_data.index, recent_data['RSI'], label='RSI', color='purple', linewidth=2)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax3.fill_between(recent_data.index, 30, 70, alpha=0.1)
        ax3.set_ylabel('RSI')
        ax3.set_title('Relative Strength Index')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = axes[3]
        ax4.plot(recent_data.index, recent_data['MACD'], label='MACD', color='blue')
        ax4.plot(recent_data.index, recent_data['MACD_Signal'], label='Signal', color='red')
        ax4.bar(recent_data.index, recent_data['MACD_Histogram'], label='Histogram', alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_ylabel('MACD')
        ax4.set_title('MACD (Moving Average Convergence Divergence)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Volatility
        ax5 = axes[4]
        ax5.plot(recent_data.index, recent_data['Volatility_30'] * 100, label='30-day Volatility', linewidth=2)
        ax5.fill_between(recent_data.index, 0, recent_data['Volatility_30'] * 100, alpha=0.3)
        ax5.set_ylabel('Volatility (%)')
        ax5.set_title('Historical Volatility (Annualized)')
        ax5.set_xlabel('Date')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bitcoin_technical_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Saved visualization to 'bitcoin_technical_indicators.png'")
    
    def get_feature_summary(self):
        """Print summary of all features created"""
        print("\n" + "="*50)
        print("FEATURE SUMMARY")
        print("="*50)
        
        # Group features by category
        categories = {
            'Price': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
            'Moving Averages': [col for col in self.data.columns if 'SMA' in col or 'EMA' in col],
            'MACD': [col for col in self.data.columns if 'MACD' in col],
            'RSI': [col for col in self.data.columns if 'RSI' in col],
            'Bollinger Bands': [col for col in self.data.columns if 'BB_' in col],
            'Volume': [col for col in self.data.columns if 'Volume' in col or 'OBV' in col or 'VWAP' in col],
            'Volatility': [col for col in self.data.columns if 'Volatility' in col or 'ATR' in col],
            'Momentum': [col for col in self.data.columns if 'ROC' in col or 'Stoch' in col or 'MFI' in col],
            'Patterns': [col for col in self.data.columns if any(x in col for x in ['Body', 'Shadow', 'Trend', 'Support', 'Resistance'])],
            'Lags': [col for col in self.data.columns if 'Lag' in col],
            'Targets': [col for col in self.data.columns if 'Target' in col]
        }
        
        for category, features in categories.items():
            if features:
                print(f"\n{category} ({len(features)} features):")
                print(f"  {', '.join(features[:5])}" + (" ..." if len(features) > 5 else ""))
        
        # Data quality check
        print("\n" + "-"*50)
        print("DATA QUALITY CHECK")
        print("-"*50)
        
        # Check for NaN values
        nan_counts = self.data.isnull().sum()
        nan_features = nan_counts[nan_counts > 0]
        
        if len(nan_features) > 0:
            print(f"\nFeatures with NaN values (due to lookback periods):")
            print(f"  Maximum NaN count: {nan_features.max()}")
            print(f"  Affected features: {len(nan_features)}")
            print(f"  Clean rows after removing NaN: {len(self.data.dropna())}")
        else:
            print("\n✓ No NaN values found")
        
        return self.data
    
    def save_indicators(self, filename='bitcoin_with_indicators.csv'):
        """Save the data with all indicators"""
        self.data.to_csv(filename)
        print(f"\n✓ Saved enhanced dataset to '{filename}'")
        return filename

# Main execution
if __name__ == "__main__":
    # Initialize the indicator generator
    ti = TechnicalIndicators()
    
    # Load Bitcoin data
    ti.load_bitcoin_data('btc_USD.xlsx')
    
    # Create all indicators
    enhanced_data = ti.create_all_indicators()
    
    # Visualize indicators
    ti.visualize_indicators()
    
    # Get feature summary
    ti.get_feature_summary()
    
    # Save enhanced dataset
    ti.save_indicators()
    
    print("\n✅ Technical indicators creation complete!")
    print("Ready for model training with enhanced features.")
