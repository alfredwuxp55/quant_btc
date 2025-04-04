import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from sklearn.linear_model import LinearRegression

# Define the ticker symbol for Nasdaq-100 Futures and the date range
ticker = "NQ=F"
start_date = "2015-01-01"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Download historical Nasdaq-100 Futures data using yfinance
Nas_data = yf.download(ticker, start=start_date, end=end_date)

# Retrieve the 10-Year Breakeven Inflation Rate from FRED
breakeven = web.DataReader('T10YIE', 'fred', start_date, end_date)

# Download historical DS Dollar Index (using "DX-Y.NYB") data using yfinance
usd_data = yf.download("DX-Y.NYB", start=start_date, end=end_date)

# Download historical Bitcoin daily data using yfinance
btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

# Create a single figure with 4 subplots (stacked vertically) sharing the x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# First subplot: Nasdaq-100 Futures Close Price
ax1.plot(Nas_data.index, Nas_data['Close'], label='Nasdaq-100 Futures')
ax1.set_ylabel('Nasdaq-100 Futures')
ax1.set_title('Nasdaq-100 Futures Close Price (2015 - Today)')
ax1.legend()
ax1.grid(True)

# Second subplot: 10-Year Breakeven Inflation Rate
ax2.plot(breakeven.index, breakeven['T10YIE'], label='10-Year Breakeven Inflation Rate', color='blue')
ax2.set_ylabel('Inflation Rate (%)')
ax2.set_title('10-Year Breakeven Inflation Rate (2015 - Today)')
ax2.legend()
ax2.grid(True)

# Third subplot: DS Dollar Index Data
ax3.plot(usd_data.index, usd_data['Close'], label='DS Dollar Index')
ax3.set_ylabel('DS Dollar Index Close Price')
ax3.set_title('DS Dollar Index (2015 - Today)')
ax3.legend()
ax3.grid(True)

# Fourth subplot: Bitcoin Daily Close Price
ax4.plot(btc_data.index, btc_data['Close'], label='Bitcoin Close Price', color='orange')
ax4.set_xlabel('Date')
ax4.set_ylabel('Bitcoin Price (USD)')
ax4.set_title('Bitcoin Daily Close Price (2015 - Today)')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()

# --- New Feature: Sliding Window Regression ---

# Merge the datasets into one DataFrame based on common dates.
# Using .squeeze() ensures that each column is 1-dimensional.
df = pd.DataFrame({
    'Nas': Nas_data['Close'].squeeze(),
    'usd': usd_data['Close'].squeeze(),
    'breakeven': breakeven['T10YIE'].squeeze(),
    'btc': btc_data['Close'].squeeze()
})
df.dropna(inplace=True)  # Remove any dates with missing data

# Normalize the data using z-score normalization
df_norm = (df - df.mean()) / df.std()

# Set the sliding window size (10 days)
window_size = 5

# Lists to store the regression coefficients and corresponding dates
dates = []
coef_Nas = []
coef_usd = []
coef_breakeven = []

# Perform sliding window regression
for i in range(len(df_norm) - window_size + 1):
    window = df_norm.iloc[i:i+window_size]
    # Independent variables: Nasdaq-100 Futures, DS Dollar Index, and 10-Year Breakeven Inflation Rate
    X = window[['Nas', 'usd', 'breakeven']].values
    # Dependent variable: Bitcoin price
    y = window['btc'].values
    model = LinearRegression()
    model.fit(X, y)
    # Save the coefficients and use the last date of the window as reference
    coef_Nas.append(model.coef_[0])
    coef_usd.append(model.coef_[1])
    coef_breakeven.append(model.coef_[2])
    dates.append(window.index[-1])

# Create a DataFrame to store all the regression coefficients with their dates
coef_df = pd.DataFrame({
    'Date': dates,
    'Coef_Nas': coef_Nas,
    'Coef_usd': coef_usd,
    'Coef_breakeven': coef_breakeven
})



# Save the coefficients DataFrame to a CSV file
coef_df.to_csv('coefficients.csv', index=False)

# Normalize each coefficient by dividing by its maximum absolute value
coef_df['Coef_Nas_norm'] = coef_df['Coef_Nas'] / coef_df['Coef_Nas'].abs().max()
coef_df['Coef_usd_norm'] = coef_df['Coef_usd'] / coef_df['Coef_usd'].abs().max()
coef_df['Coef_breakeven_norm'] = coef_df['Coef_breakeven'] / coef_df['Coef_breakeven'].abs().max()

print(coef_df)

# Figure 1: Three subplots, one for each coefficient
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(coef_df['Date'], coef_df['Coef_Nas_norm'], label='Tech Factor')
ax1.set_ylabel('Tech factor')
ax1.set_title('Tech factor Coefficient Over Time')
ax1.legend()
ax1.grid(True)

ax2.plot(coef_df['Date'], coef_df['Coef_usd_norm'], label='Outside Payment Factor(Inverse Better)', color='orange')
ax2.set_ylabel('Outside Payment & Gold')
ax2.set_title('Outside Payment and Gold Coefficient Over Time')
ax2.legend()
ax2.grid(True)

ax3.plot(coef_df['Date'], coef_df['Coef_breakeven_norm'], label='Digital Gold Factor', color='green')
ax3.set_xlabel('Date')
ax3.set_ylabel('Digital Gold')
ax3.set_title('Digital Gold Coefficient Over Time')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

# Figure 2: All coefficients in one figure
plt.figure(figsize=(12, 6))
plt.plot(coef_df['Date'], coef_df['Coef_Nas_norm'], label='Tech factor')
plt.plot(coef_df['Date'], coef_df['Coef_usd_norm'], label='Outside Payment Factor (Inverse Better)', color='orange')
plt.plot(coef_df['Date'], coef_df['Coef_breakeven_norm'], label='Digital Gold Factor', color='green')
plt.xlabel('Date')
plt.ylabel('Regression Coefficient')
plt.title('Sliding Window Regression Coefficients (Window Size = 5 Days)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()