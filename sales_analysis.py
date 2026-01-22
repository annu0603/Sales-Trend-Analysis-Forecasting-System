# Sales Trend Analysis & Forecasting System
# Using Superstore Sales Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================
# PART 1: LOAD SUPERSTORE DATA
# ============================================

def load_superstore_data(filepath):
    """Load and prepare Superstore dataset"""
    
    print("📂 Loading Superstore dataset...")
    
    # Try different common file names
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Please ensure the CSV file path is correct")
            return None
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    return df

# ============================================
# PART 2: DATA CLEANING AND PREPARATION
# ============================================

def clean_superstore_data(df):
    """Clean and prepare Superstore data for analysis"""
    
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\n📊 Missing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # Standardize column names (Superstore can have different formats)
    # Common column names: Order Date, Ship Date, Sales, Profit, Quantity, Category, Sub-Category
    df.columns = df.columns.str.strip()
    
    # Convert date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Determine the main date column
    if 'Order Date' in df.columns:
        df['Date'] = df['Order Date']
    elif 'Order_Date' in df.columns:
        df['Date'] = df['Order_Date']
    else:
        # Find the first date column
        df['Date'] = df[date_columns[0]]
    
    # Extract time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Month_Name'] = df['Date'].dt.month_name()
    
    # Remove rows with missing dates
    initial_rows = len(df)
    df = df.dropna(subset=['Date'])
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"\n⚠️ Removed {removed_rows} rows with missing dates")
    
    # Sort by date
    df = df.sort_values('Date')
    
    print(f"\n✅ Data cleaned! Final shape: {df.shape}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

# ============================================
# PART 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

def perform_eda(df):
    """Generate key insights and statistics"""
    
    print("\n" + "="*50)
    print("KEY BUSINESS METRICS")
    print("="*50)
    
    # Overall metrics
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
    total_orders = len(df)
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    print(f"\n💰 Total Sales: ${total_sales:,.2f}")
    if 'Profit' in df.columns:
        print(f"📈 Total Profit: ${total_profit:,.2f}")
        print(f"📊 Profit Margin: {profit_margin:.2f}%")
    print(f"🛒 Total Orders: {total_orders:,}")
    print(f"💵 Average Order Value: ${total_sales/total_orders:,.2f}")
    
    # Year-over-Year Growth
    yearly_sales = df.groupby('Year')['Sales'].sum()
    print("\n📅 Year-wise Sales:")
    for year, sales in yearly_sales.items():
        print(f"   {year}: ${sales:,.2f}")
    
    if len(yearly_sales) > 1:
        yoy_growth = ((yearly_sales.iloc[-1] - yearly_sales.iloc[-2]) / yearly_sales.iloc[-2]) * 100
        print(f"\n📈 Latest YoY Growth: {yoy_growth:.2f}%")
    
    # Top performing categories
    if 'Category' in df.columns:
        print("\n🏆 Top Categories by Sales:")
        top_categories = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
        for i, (category, sales) in enumerate(top_categories.items(), 1):
            pct = (sales / total_sales) * 100
            print(f"   {i}. {category}: ${sales:,.2f} ({pct:.1f}%)")
    
    # Top performing sub-categories
    if 'Sub-Category' in df.columns:
        print("\n🎯 Top 5 Sub-Categories by Sales:")
        top_subcats = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(5)
        for i, (subcat, sales) in enumerate(top_subcats.items(), 1):
            print(f"   {i}. {subcat}: ${sales:,.2f}")
    
    # Regional performance
    if 'Region' in df.columns:
        print("\n🌍 Regional Performance:")
        regional_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        for region, sales in regional_sales.items():
            pct = (sales / total_sales) * 100
            print(f"   {region}: ${sales:,.2f} ({pct:.1f}%)")
    
    # Best month for sales
    monthly_sales = df.groupby('Month_Name')['Sales'].sum()
    best_month = monthly_sales.idxmax()
    print(f"\n🔥 Best performing month: {best_month} (${monthly_sales.max():,.2f})")
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'profit_margin': profit_margin
    }

# ============================================
# PART 4: COMPREHENSIVE VISUALIZATIONS
# ============================================

def create_visualizations(df):
    """Generate comprehensive visualizations"""
    
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    fig = plt.figure(figsize=(18, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Monthly Sales Trend
    ax1 = fig.add_subplot(gs[0, :])
    monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Sales'].sum()
    monthly_sales.index = monthly_sales.index.to_timestamp()
    ax1.plot(monthly_sales.index, monthly_sales.values, linewidth=2, color='#2E86AB', marker='o')
    ax1.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales ($)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Sales by Category
    ax2 = fig.add_subplot(gs[1, 0])
    if 'Category' in df.columns:
        category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
        ax2.barh(category_sales.index, category_sales.values, color='#A23B72')
        ax2.set_title('Sales by Category', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sales ($)')
    
    # 3. Top 10 Sub-Categories
    ax3 = fig.add_subplot(gs[1, 1])
    if 'Sub-Category' in df.columns:
        subcat_sales = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(10)
        ax3.bar(range(len(subcat_sales)), subcat_sales.values, color='#F18F01')
        ax3.set_title('Top 10 Sub-Categories', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Sub-Category')
        ax3.set_ylabel('Sales ($)')
        ax3.set_xticks(range(len(subcat_sales)))
        ax3.set_xticklabels(subcat_sales.index, rotation=45, ha='right')
    
    # 4. Regional Performance
    ax4 = fig.add_subplot(gs[2, 0])
    if 'Region' in df.columns:
        region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=True)
        colors = ['#C73E1D', '#F18F01', '#2E86AB', '#A23B72']
        ax4.barh(region_sales.index, region_sales.values, color=colors[:len(region_sales)])
        ax4.set_title('Sales by Region', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Sales ($)')
    
    # 5. Sales by Month (Seasonal Pattern)
    ax5 = fig.add_subplot(gs[2, 1])
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_avg = df.groupby('Month_Name')['Sales'].sum().reindex(month_order)
    ax5.bar(range(12), monthly_avg.values, color='#2E86AB')
    ax5.set_title('Sales by Month (Seasonal Pattern)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Month')
    ax5.set_ylabel('Total Sales ($)')
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    
    plt.suptitle('Superstore Sales Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('superstore_dashboard.png', dpi=300, bbox_inches='tight')
    print("\n✅ Dashboard saved as 'superstore_dashboard.png'")
    plt.show()

# ============================================
# PART 5: SALES FORECASTING
# ============================================

def forecast_sales(df, forecast_days=90):
    """Sales forecasting using linear regression"""
    
    print("\n" + "="*50)
    print("SALES FORECASTING")
    print("="*50)
    
    # Aggregate daily sales
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales['Days'] = (daily_sales['Date'] - daily_sales['Date'].min()).dt.days
    
    # Split into train/test (80/20)
    split_idx = int(len(daily_sales) * 0.8)
    train = daily_sales[:split_idx]
    test = daily_sales[split_idx:]
    
    # Train model
    X_train = train[['Days']].values
    y_train = train['Sales'].values
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    X_test = test[['Days']].values
    y_test = test['Sales'].values
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    print(f"\n📊 Model Performance Metrics:")
    print(f"   Mean Absolute Error: ${mae:,.2f}")
    print(f"   Root Mean Squared Error: ${rmse:,.2f}")
    print(f"   Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Forecast future
    last_day = daily_sales['Days'].max()
    future_days = np.array([[last_day + i] for i in range(1, forecast_days + 1)])
    future_predictions = model.predict(future_days)
    
    # Calculate 7-day moving average for smoother visualization
    daily_sales['MA_7'] = daily_sales['Sales'].rolling(window=7, min_periods=1).mean()
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Historical with forecast
    ax1.plot(daily_sales['Date'], daily_sales['Sales'], 
             label='Actual Daily Sales', linewidth=1, alpha=0.5, color='gray')
    ax1.plot(daily_sales['Date'], daily_sales['MA_7'], 
             label='7-Day Moving Average', linewidth=2, color='#2E86AB')
    
    future_dates = [daily_sales['Date'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
    ax1.plot(future_dates, future_predictions, 
             label=f'{forecast_days}-Day Forecast', 
             linestyle='--', linewidth=2, color='red')
    
    ax1.set_title('Sales Forecast with Historical Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Actual vs Predicted (Test Set)
    ax2.plot(test['Date'], y_test, label='Actual', linewidth=2, color='#2E86AB')
    ax2.plot(test['Date'], predictions, label='Predicted', linewidth=2, linestyle='--', color='red')
    ax2.fill_between(test['Date'], y_test, predictions, alpha=0.3, color='gray')
    ax2.set_title('Model Performance: Actual vs Predicted (Test Set)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Sales ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('superstore_forecast.png', dpi=300, bbox_inches='tight')
    print("\n✅ Forecast chart saved as 'superstore_forecast.png'")
    plt.show()
    
    # Print forecast summary
    avg_forecast = np.mean(future_predictions)
    print(f"\n📈 Forecast Summary:")
    print(f"   Next {forecast_days} days average daily sales: ${avg_forecast:,.2f}")
    print(f"   Total forecasted sales: ${np.sum(future_predictions):,.2f}")
    
    return model, future_predictions

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("🚀 SUPERSTORE SALES ANALYSIS & FORECASTING SYSTEM")
    print("="*60)
    
    # IMPORTANT: Update this path to your downloaded file
    filepath = 'Sample - Superstore.csv'  # or 'superstore_sales.csv'
    
    # You can also try these common names:
    # filepath = 'Superstore.csv'
    # filepath = 'superstore.csv'
    # filepath = 'Sample-Superstore.csv'
    
    # Step 1: Load Data
    df = load_superstore_data('train.csv')
    
    if df is None:
        print("\n❌ Failed to load data. Please check the file path and try again.")
        print("Common file names: 'Sample - Superstore.csv', 'superstore.csv'")
        exit()
    
    # Step 2: Clean Data
    df = clean_superstore_data(df)
    
    # Step 3: Exploratory Analysis
    metrics = perform_eda(df)
    
    # Step 4: Create Visualizations
    create_visualizations(df)
    
    # Step 5: Forecast Future Sales
    model, forecast = forecast_sales(df, forecast_days=90)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   1. superstore_dashboard.png - Comprehensive visualization dashboard")
    print("   2. superstore_forecast.png - Sales forecast with model performance")
    print("\n📝 Next Steps:")
    print("   1. Review the generated visualizations")
    print("   2. Extract key insights for your presentation")
    print("   3. Create executive summary slides")
    print("   4. Update your resume with project details")
    print("\n💡 Pro Tip: Open the PNG files to see your analysis results!")
    print("="*60)