import numpy as np
import pandas as pd

np.random.seed(42)

dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
daily_sales = 100 + np.cumsum(np.random.randn(100) * 3)
df = pd.DataFrame({'Date': dates, 'Daily_Sales': daily_sales})
df.set_index('Date', inplace=True)

#print("First 5 rows of the ice cream sales data:")
#print(df.head())

df['MA_7_Wrong'] = df['Daily_Sales'].rolling(window=7).mean()
df['Sales_Shifted_1'] = df['Daily_Sales'].shift(1)
df['MA_7_Correct'] = df['Sales_Shifted_1'].rolling(window=7).mean()

print("\n" + "="*50)
print("The first 15 rows of our spreadsheet:")
print("="*50)
print(df.head(15))