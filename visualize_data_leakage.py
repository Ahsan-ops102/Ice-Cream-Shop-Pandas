import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# PART 1: CREATE THE DATASET WITH MISSING VALUES
# ============================================

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

print("📊 Dataset loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"Missing values: {df['Daily_Sales'].isna().sum()}")

# ============================================
# PART 2: FILL MISSING VALUES (2 METHODS)
# ============================================

# Method 1: Forward Fill (CORRECT - uses past data)
df_ffill = df.copy()
df_ffill['Daily_Sales'] = df_ffill['Daily_Sales'].ffill()

# Method 2: Backward Fill (WRONG - uses future data)
df_bfill = df.copy()
df_bfill['Daily_Sales'] = df_bfill['Daily_Sales'].bfill()

print("✅ Missing values filled successfully!")
print(f"FFill filled {df_ffill['Daily_Sales'].isna().sum()} missing values")
print(f"BFill filled {df_bfill['Daily_Sales'].isna().sum()} missing values")

# ============================================
# PART 3: VISUALIZATION - SEE THE DIFFERENCE!
# ============================================

# Create a figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# ============================================
# PLOT 1: The CHEATER way (using future data)
# ============================================

# Plot the actual sales (blue line)
axes[0].plot(df.index, df_ffill['Daily_Sales'], 
             label='Actual Sales', 
             color='blue', 
             linewidth=2)

# Plot the BFill (red dashed line - CHEATING!)
axes[0].plot(df.index, df_bfill['Daily_Sales'], 
             label='BFill (USES FUTURE DATA!)', 
             color='red', 
             linestyle='--', 
             linewidth=2)

# Mark the filled values with red circles
axes[0].scatter(df_bfill.loc[df['Daily_Sales'].isna()].index, 
                df_bfill.loc[df['Daily_Sales'].isna()]['Daily_Sales'], 
                color='red', 
                s=100, 
                marker='o', 
                label='Filled using FUTURE data')

axes[0].set_title('❌ WRONG: Using Future Data (B-Fill) - CHEATING!', fontsize=14)
axes[0].set_xlabel('Date', fontsize=12)
axes[0].set_ylabel('Daily Sales', fontsize=12)
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# ============================================
# PLOT 2: The CORRECT way (using past data)
# ============================================

# Plot the actual sales (blue line)
axes[1].plot(df.index, df_ffill['Daily_Sales'], 
             label='Actual Sales', 
             color='blue', 
             linewidth=2)

# Plot the FFill (green dashed line - CORRECT!)
axes[1].plot(df.index, df_ffill['Daily_Sales'], 
             label='FFill (USES PAST DATA!)', 
             color='green', 
             linestyle='--', 
             linewidth=2)

# Mark the filled values with green triangles
axes[1].scatter(df_ffill.loc[df['Daily_Sales'].isna()].index, 
                df_ffill.loc[df['Daily_Sales'].isna()]['Daily_Sales'], 
                color='green', 
                s=100, 
                marker='^', 
                label='Filled using PAST data')

axes[1].set_title('✅ CORRECT: Using Past Data (F-Fill) - REAL Prediction!', fontsize=14)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Daily Sales', fontsize=12)
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# ============================================
# PART 4: ADD ANNOTATIONS TO EXPLAIN THE DATA LEAKAGE
# ============================================

# Add a text box explaining the difference
fig.text(0.5, 0.02, 
         "💡 KEY INSIGHT: Top graph uses FUTURE data (CHEATING!). Bottom graph uses PAST data (CORRECT!).\n"
         "In real-world prediction, you CANNOT use tomorrow's data to predict today!",
         ha='center', 
         fontsize=12, 
         bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='black'))

# ============================================
# PART 5: SAVE AND SHOW
# ============================================

# Tighten the layout
plt.tight_layout(rect=[0, 0.07, 1, 1])  # Leave space for the text box

# Save the plot
plt.savefig('data_leakage_comparison.png', dpi=300, bbox_inches='tight')
print("\n📊 Plot saved as 'data_leakage_comparison.png'")

# Show the plot
plt.show()

# ============================================
# PART 6: PRINT THE NUMBERS IN THE TERMINAL
# ============================================

print("\n" + "="*50)
print("📊 DATA LEAKAGE - THE NUMBERS:")
print("="*50)

# Show the missing rows with both filling methods
missing_rows = df[df['Daily_Sales'].isna()].index

print("\n🔴 WRONG (Using Future Data - BFill):")
for date in missing_rows:
    original = df.loc[date, 'Daily_Sales']
    bfill_val = df_bfill.loc[date, 'Daily_Sales']
    ffill_val = df_ffill.loc[date, 'Daily_Sales']
    print(f"  {date.strftime('%Y-%m-%d')}: Original = {original}, BFill = {bfill_val:.1f} (USES TOMORROW'S DATA!)")

print("\n🟢 CORRECT (Using Past Data - FFill):")
for date in missing_rows:
    original = df.loc[date, 'Daily_Sales']
    bfill_val = df_bfill.loc[date, 'Daily_Sales']
    ffill_val = df_ffill.loc[date, 'Daily_Sales']
    print(f"  {date.strftime('%Y-%m-%d')}: Original = {original}, FFill = {ffill_val:.1f} (USES YESTERDAY'S DATA!)")

print("\n" + "="*50)
print("💡 WHY THIS MATTERS:")
print("="*50)
print("• BFill uses TOMORROW's sales to fill TODAY's missing value")
print("• This is CHEATING because you wouldn't know tomorrow's sales yet!")
print("• FFill uses YESTERDAY's sales - which you DO know!")
print("• This is why FFill is CORRECT for prediction, BFill is WRONG!")