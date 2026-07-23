import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CREATE DATASET WITH OUTLIERS
# ============================================

# Generate normal data (100 values around 50 with some variation)
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=100)

# Add some outliers
outliers = np.array([120, 130, 5, 3, 140, 150])

# Combine data
all_data = np.concatenate([normal_data, outliers])

# Create DataFrame
df = pd.DataFrame({
    'Value': all_data
})

print("=" * 50)
print("📊 DATASET WITH OUTLIERS")
print("=" * 50)
print(f"Total values: {len(df)}")
print(f"Mean: {df['Value'].mean():.2f}")
print(f"Standard Deviation: {df['Value'].std():.2f}")
print(f"Min: {df['Value'].min():.2f}")
print(f"Max: {df['Value'].max():.2f}")

# ============================================
# METHOD 1: Z-SCORE DETECTION
# ============================================

print("\n" + "=" * 50)
print("📊 METHOD 1: Z-SCORE DETECTION")
print("=" * 50)

# Calculate Z-Score for each value
mean_val = df['Value'].mean()
std_val = df['Value'].std()
df['Z_Score'] = (df['Value'] - mean_val) / std_val

# Identify outliers (Z-Score > 3 or < -3)
df['Is_Z_Outlier'] = abs(df['Z_Score']) > 3

# Show outliers
z_outliers = df[df['Is_Z_Outlier']]
print(f"\n🔴 Z-Score Outliers (Z > 3 or Z < -3): {len(z_outliers)}")
print("\nOutlier values and their Z-Scores:")
print(z_outliers[['Value', 'Z_Score']])

# Show summary
print(f"\nZ-Score Range: {df['Z_Score'].min():.2f} to {df['Z_Score'].max():.2f}")

# ============================================
# METHOD 2: IQR DETECTION
# ============================================

print("\n" + "=" * 50)
print("📊 METHOD 2: IQR DETECTION")
print("=" * 50)

# Calculate Q1, Q3, and IQR
Q1 = df['Value'].quantile(0.25)
Q3 = df['Value'].quantile(0.75)
IQR = Q3 - Q1

# Calculate bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 (25th percentile): {Q1:.2f}")
print(f"Q3 (75th percentile): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")

# Identify outliers
df['Is_IQR_Outlier'] = (df['Value'] < lower_bound) | (df['Value'] > upper_bound)

# Show outliers
iqr_outliers = df[df['Is_IQR_Outlier']]
print(f"\n🔴 IQR Outliers: {len(iqr_outliers)}")
print("\nOutlier values and their bounds:")
print(iqr_outliers[['Value']])

# ============================================
# COMPARE BOTH METHODS
# ============================================

print("\n" + "=" * 50)
print("📊 COMPARISON: Z-Score vs IQR")
print("=" * 50)

# Show which outliers each method found
df['Outlier_Type'] = 'Normal'
df.loc[df['Is_Z_Outlier'] & df['Is_IQR_Outlier'], 'Outlier_Type'] = 'Both'
df.loc[df['Is_Z_Outlier'] & ~df['Is_IQR_Outlier'], 'Outlier_Type'] = 'Z-Score Only'
df.loc[~df['Is_Z_Outlier'] & df['Is_IQR_Outlier'], 'Outlier_Type'] = 'IQR Only'

print("\nOutlier detection summary:")
print(df['Outlier_Type'].value_counts())

# ============================================
# VISUALIZATION
# ============================================

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Histogram with outliers marked
axes[0].hist(df['Value'], bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
axes[0].axvline(mean_val + 3*std_val, color='orange', linestyle='--', label='+3 STD')
axes[0].axvline(mean_val - 3*std_val, color='orange', linestyle='--', label='-3 STD')
axes[0].set_title('Histogram with Z-Score Bounds')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Boxplot (IQR method)
box = axes[1].boxplot(df['Value'], vert=True, patch_artist=True)
axes[1].set_title('Boxplot (IQR Method)')
axes[1].set_ylabel('Value')
axes[1].grid(True, alpha=0.3)

# Plot 3: Scatter plot showing outliers
colors = ['red' if x else 'blue' for x in df['Is_Z_Outlier']]
axes[2].scatter(df.index, df['Value'], c=colors, alpha=0.7)
axes[2].axhline(mean_val, color='green', linestyle='--', label='Mean')
axes[2].axhline(upper_bound, color='red', linestyle='--', label='IQR Upper')
axes[2].axhline(lower_bound, color='red', linestyle='--', label='IQR Lower')
axes[2].set_title('Outliers (Red = Z-Score Outlier)')
axes[2].set_xlabel('Index')
axes[2].set_ylabel('Value')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outlier_detection.png', dpi=300, bbox_inches='tight')
print("\n📊 Plot saved as 'outlier_detection.png'")
plt.show()

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 50)
print("📊 FINAL SUMMARY")
print("=" * 50)

print("\n🔴 OUTLIERS FOUND:")
print(f"Z-Score Method: {len(z_outliers)} outliers")
print(f"IQR Method: {len(iqr_outliers)} outliers")

print("\n💡 KEY INSIGHTS:")
print("1. Z-Score works best for normally distributed data")
print("2. IQR works best when data has outliers or isn't normal")
print("3. Both methods are used in real-world data science")
print("4. Always investigate outliers before removing them!")