import pandas as pd
import numpy as np

# Create dates
dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

# Create sales with some missing values
sales = [
    101, 102, 103, np.nan, 105,  # Jan 1-5
    106, 107, 108, 109, np.nan,  # Jan 6-10
    110, 111, 112, 113, 114,     # Jan 11-15
    115, np.nan, 117, 118, 119,  # Jan 16-20
    120, 121, 122, 123, 124,     # Jan 21-25
    125, 126, 127, 128, 129      # Jan 26-30
]

# Create DataFrame
df = pd.DataFrame({'Date': dates, 'Daily_Sales': sales})
df.set_index('Date', inplace=True)

print("Original Data with Missing Values:")
print(df)

# Check missing values
print(f"\nNumber of missing values: {df['Daily_Sales'].isna().sum()}")
print("\nRows with missing values:")
print(df[df['Daily_Sales'].isna()])

# ============================================
# METHOD 1: Fill with 0 (FIXED)
# ============================================
df_zero = df.copy()
df_zero['Daily_Sales'] = df_zero['Daily_Sales'].fillna(0)
print("\nMethod 1 - Fill with 0 (First 10 rows):")
print(df_zero.head(10))

# ============================================
# METHOD 2: Fill with Mean (FIXED)
# ============================================
mean_sales = df['Daily_Sales'].mean()
print(f"\nMean (Average) sales: {mean_sales:.2f}")

df_mean = df.copy()
df_mean['Daily_Sales'] = df_mean['Daily_Sales'].fillna(mean_sales)
print("\nMethod 2 - Fill with Mean (First 10 rows):")
print(df_mean.head(10))

# ============================================
# METHOD 3: Fill with Median (FIXED)
# ============================================
median_sales = df['Daily_Sales'].median()
print(f"\nMedian sales: {median_sales:.2f}")

df_median = df.copy()
df_median['Daily_Sales'] = df_median['Daily_Sales'].fillna(median_sales)
print("\nMethod 3 - Fill with Median (First 10 rows):")
print(df_median.head(10))

# ============================================
# METHOD 4: Forward Fill (FIXED - using ffill)
# ============================================
df_ffill = df.copy()
df_ffill['Daily_Sales'] = df_ffill['Daily_Sales'].ffill()
print("\nMethod 4 - Forward Fill (ffill) (First 10 rows):")
print(df_ffill.head(10))

# ============================================
# METHOD 5: Backward Fill (FIXED - using bfill)
# ============================================
df_bfill = df.copy()
df_bfill['Daily_Sales'] = df_bfill['Daily_Sales'].bfill()
print("\nMethod 5 - Backward Fill (bfill) (First 10 rows):")
print(df_bfill.head(10))

# ============================================
# METHOD 6: Interpolation (The Professional's Choice)
# ============================================
df_interp = df.copy()
df_interp['Daily_Sales'] = df_interp['Daily_Sales'].interpolate()

print("\nMethod 6 - Interpolation (First 10 rows):")
print(df_interp.head(10))

# ============================================
# FINAL COMPARISON - ALL 6 METHODS
# ============================================
comparison_full = pd.DataFrame({
    'Original': df['Daily_Sales'],
    'Zero': df_zero['Daily_Sales'],
    'Mean': df_mean['Daily_Sales'],
    'Median': df_median['Daily_Sales'],
    'FFill': df_ffill['Daily_Sales'],
    'BFill': df_bfill['Daily_Sales'],
    'Interp': df_interp['Daily_Sales']
})

print("\n" + "="*50)
print("COMPARISON OF ALL 6 METHODS (Rows with missing values):")
print("="*50)
print(comparison_full.loc[df['Daily_Sales'].isna()])

# ============================================
# STATISTICS
# ============================================
print(f"\nOriginal rows: {len(df)}")
print(f"Rows with missing values: {df['Daily_Sales'].isna().sum()}")
print(f"Mean: {mean_sales:.2f}")
print(f"Median: {median_sales:.2f}")