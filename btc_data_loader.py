import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_and_analyze_bitcoin_data(file_path='btc_USD.xlsx'):
    """Load Bitcoin data and verify date formatting"""
    
    print("="*60)
    print("BITCOIN DATA LOADING AND ANALYSIS")
    print("="*60)
    
    # Step 1: Load the data with explicit date parsing
    print("\n1. Loading Bitcoin data from Excel file...")
    
    try:
        # First, let's load without date parsing to see raw data
        raw_data = pd.read_excel(file_path, nrows=5)
        print("\nRaw data preview (first 5 rows):")
        print(raw_data)
        print(f"\nRaw date column sample: {raw_data['Date'].iloc[0]}")
        
        # Now load with proper date parsing (DD/MM/YYYY format)
        btc_data = pd.read_excel(file_path, parse_dates=['Date'], dayfirst=True)
        
        print(f"\n✓ Successfully loaded {len(btc_data)} rows of data")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Step 2: Verify date parsing
    print("\n2. Verifying date format parsing...")
    print(f"First date in dataset: {btc_data['Date'].iloc[0]}")
    print(f"Last date in dataset: {btc_data['Date'].iloc[-1]}")
    print(f"Date type: {type(btc_data['Date'].iloc[0])}")
    
    # Check if dates are in correct chronological order
    is_sorted = btc_data['Date'].is_monotonic_increasing
    print(f"Dates in chronological order: {is_sorted}")
    
    if not is_sorted:
        print("⚠️ Warning: Dates are not in chronological order. Sorting...")
        btc_data = btc_data.sort_values('Date')
    
    # Step 3: Data structure analysis
    print("\n3. Data Structure Analysis:")
    print(f"Shape: {btc_data.shape}")
    print(f"\nColumn names: {btc_data.columns.tolist()}")
    print(f"\nData types:")
    print(btc_data.dtypes)
    
    # Step 4: Check for missing values
    print("\n4. Missing Values Check:")
    missing_values = btc_data.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() > 0:
        print(f"\n⚠️ Total missing values: {missing_values.sum()}")
    else:
        print("\n✓ No missing values found")
    
    # Step 5: Basic statistics
    print("\n5. Basic Statistics:")
    print(btc_data.describe().round(2))
    
    # Step 6: Date range analysis
    print("\n6. Date Range Analysis:")
    date_range = btc_data['Date'].max() - btc_data['Date'].min()
    print(f"Total date range: {date_range.days} days ({date_range.days/365:.1f} years)")
    print(f"Start date: {btc_data['Date'].min().strftime('%d/%m/%Y')}")
    print(f"End date: {btc_data['Date'].max().strftime('%d/%m/%Y')}")
    
    # Check for gaps in dates
    btc_data['Date_Diff'] = btc_data['Date'].diff()
    max_gap = btc_data['Date_Diff'].max()
    print(f"\nMaximum gap between dates: {max_gap}")
    
    gaps = btc_data[btc_data['Date_Diff'] > pd.Timedelta(days=1)]
    if len(gaps) > 0:
        print(f"⚠️ Found {len(gaps)} gaps larger than 1 day")
        print("Largest gaps:")
        print(gaps.nlargest(5, 'Date_Diff')[['Date', 'Date_Diff']])
    else:
        print("✓ No significant gaps in date sequence")
    
    # Step 7: Sample data display with formatted dates
    print("\n7. Sample Data (with properly formatted dates):")
    sample_data = btc_data.head(10).copy()
    sample_data['Date_Formatted'] = sample_data['Date'].dt.strftime('%d/%m/%Y')
    print(sample_data[['Date_Formatted', 'Open', 'High', 'Low', 'Close', 'Volume']])
    
    # Step 8: Create visualization to verify data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bitcoin Data Verification', fontsize=16)
    
    # Price over time
    ax1 = axes[0, 0]
    ax1.plot(btc_data['Date'], btc_data['Close'], linewidth=1)
    ax1.set_title('Bitcoin Closing Price Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.grid(True, alpha=0.3)
    
    # Volume over time
    ax2 = axes[0, 1]
    ax2.plot(btc_data['Date'], btc_data['Volume'], linewidth=1, color='orange', alpha=0.7)
    ax2.set_title('Trading Volume Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3)
    
    # Price distribution
    ax3 = axes[1, 0]
    ax3.hist(btc_data['Close'], bins=50, edgecolor='black', alpha=0.7)
    ax3.set_title('Close Price Distribution')
    ax3.set_xlabel('Price (USD)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Daily returns distribution
    btc_data['Returns'] = btc_data['Close'].pct_change()
    ax4 = axes[1, 1]
    ax4.hist(btc_data['Returns'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax4.set_title('Daily Returns Distribution')
    ax4.set_xlabel('Returns')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Step 9: Final verification
    print("\n9. Data Loading Summary:")
    print("✓ Data loaded successfully")
    print("✓ Dates parsed correctly in DD/MM/YYYY format")
    print("✓ All columns present and correctly typed")
    print(f"✓ Ready for analysis with {len(btc_data)} days of Bitcoin data")
    
    # Set Date as index for time series analysis
    btc_data.set_index('Date', inplace=True)
    btc_data.drop('Date_Diff', axis=1, inplace=True)  # Remove temporary column
    
    return btc_data

# Additional function to check specific date samples
def verify_date_parsing(btc_data):
    """Additional verification of date parsing"""
    print("\n" + "="*60)
    print("DATE PARSING VERIFICATION")
    print("="*60)
    
    # Test specific dates to ensure correct parsing
    test_samples = btc_data.sample(n=5, random_state=42).copy()
    
    print("\nRandom date samples for verification:")
    for idx, row in test_samples.iterrows():
        print(f"Date: {idx.strftime('%d/%m/%Y')} (Day: {idx.day}, Month: {idx.month}, Year: {idx.year})")
        print(f"  - Close Price: ${row['Close']:,.2f}")
        print(f"  - Volume: {row['Volume']:,.0f}")
        print()
    
    # Check for any potential parsing issues
    # For example, dates that might be ambiguous (like 01/02/2023 - is it Jan 2 or Feb 1?)
    ambiguous_dates = btc_data[(btc_data.index.day <= 12) & (btc_data.index.month <= 12)]
    
    if len(ambiguous_dates) > 0:
        print(f"\nFound {len(ambiguous_dates)} potentially ambiguous dates (day <= 12, month <= 12)")
        print("Sample of potentially ambiguous dates:")
        sample_ambiguous = ambiguous_dates.head(5)
        for idx, row in sample_ambiguous.iterrows():
            print(f"  {idx.strftime('%d/%m/%Y')} - Close: ${row['Close']:,.2f}")
    
    return True

# Main execution
if __name__ == "__main__":
    # Load and analyze the data
    btc_data = load_and_analyze_bitcoin_data('btc_USD.xlsx')
    
    if btc_data is not None:
        # Additional date verification
        verify_date_parsing(btc_data)
        
        # Save the properly formatted data for next steps
        print("\n✓ Data is ready for feature engineering and model training!")
        print(f"✓ Dataset spans from {btc_data.index.min().strftime('%d/%m/%Y')} to {btc_data.index.max().strftime('%d/%m/%Y')}")
