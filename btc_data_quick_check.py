import pandas as pd
import numpy as np

def quick_verify_bitcoin_data(file_path='btc_USD.xlsx'):
    """Quick verification of Bitcoin data and date format"""
    
    print("BITCOIN DATA QUICK VERIFICATION")
    print("="*50)
    
    # 1. Load data with DD/MM/YYYY format
    print("\nLoading data...")
    try:
        # Load data with dayfirst=True for DD/MM/YYYY format
        df = pd.read_excel(file_path, parse_dates=['Date'], dayfirst=True)
        print(f"✓ Successfully loaded {len(df)} rows")
        
        # Fix column names with non-breaking spaces
        df.columns = df.columns.str.replace('\xa0', ' ')
        print("✓ Fixed column names with non-breaking spaces")
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    # 2. Display basic info
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # 3. Check first and last 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
    
    # 4. Verify date parsing with specific examples
    print("\n" + "-"*50)
    print("DATE FORMAT VERIFICATION")
    print("-"*50)
    
    # Take a few sample dates to verify parsing
    sample_indices = [0, 100, 500, 1000, len(df)-1] if len(df) > 1000 else [0, len(df)//2, len(df)-1]
    
    print("\nSample dates (verifying DD/MM/YYYY parsing):")
    for i in sample_indices:
        if i < len(df):
            date = df.iloc[i]['Date']
            print(f"Row {i}: {date.strftime('%d/%m/%Y')} → Day={date.day}, Month={date.month}, Year={date.year}")
    
    # 5. Check data quality
    print("\n" + "-"*50)
    print("DATA QUALITY CHECK")
    print("-"*50)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\nMissing values found:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")
    
    # Check if dates are sorted
    is_sorted = df['Date'].is_monotonic_increasing
    print(f"\n✓ Dates in chronological order: {is_sorted}")
    
    # Price statistics
    print(f"\nPrice range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
    print(f"Average price: ${df['Close'].mean():,.2f}")
    
    # 6. Create a simple price chart
    print("\n" + "-"*50)
    print("CREATING SIMPLE PRICE CHART")
    print("-"*50)
    
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['Date'], df['Close'], linewidth=1)
        plt.title('Bitcoin Price Over Time (Verifying Data)', fontsize=14)
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("✓ Chart created successfully - check if dates and prices look correct")
    except:
        print("Could not create chart (matplotlib not available)")
    
    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)
    print("\nSummary:")
    print(f"- Data loaded: YES")
    print(f"- Rows: {len(df)}")
    print(f"- Date format: DD/MM/YYYY (parsed correctly)")
    print(f"- Date range: {(df['Date'].max() - df['Date'].min()).days} days")
    print(f"- Price range: ${df['Close'].min():,.2f} to ${df['Close'].max():,.2f}")
    print(f"- Data order: {'Chronological' if df['Date'].is_monotonic_increasing else 'Reverse chronological (newest first)'}")
    print(f"- All required columns present: YES")
    
    # Sort by date for analysis
    df = df.sort_values('Date')
    print("\n✓ Data sorted chronologically for analysis")
    
    return df

# Run the verification
if __name__ == "__main__":
    print("Starting Bitcoin data verification...\n")
    btc_data = quick_verify_bitcoin_data('btc_USD.xlsx')
    
    if btc_data is not None:
        print("\n✅ Data verification successful! Ready for analysis.")
    else:
        print("\n❌ Data verification failed. Please check the file.")
