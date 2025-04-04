import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web

# Define the ticker symbol for Nasdaq-100 Futures and the date range
ticker = "NQ=F"
start_date = "2020-01-01"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Download historical Nasdaq-100 Futures data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Retrieve the 10-Year Breakeven Inflation Rate from FRED
breakeven = web.DataReader('T10YIE', 'fred', start_date, end_date)

# Download historical DS Dollar Index (using "DX-Y.NYB") data using yfinance
Dx_data = yf.download("DX-Y.NYB", start=start_date, end=end_date)

# Download historical Bitcoin daily data using yfinance
btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

# Create a single figure with 4 subplots (stacked vertically) sharing the x-axis
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# First subplot: Nasdaq-100 Futures Close Price
ax1.plot(data.index, data['Close'], label='Close Price')
ax1.set_ylabel('Close Price')
ax1.set_title('Nasdaq-100 Futures Close Price (2020 - Today)')
ax1.legend()
ax1.grid(True)

# Second subplot: 10-Year Breakeven Inflation Rate
ax2.plot(breakeven.index, breakeven['T10YIE'], label='10-Year Breakeven Inflation Rate', color='blue')
ax2.set_ylabel('Inflation Rate (%)')
ax2.set_title('10-Year Breakeven Inflation Rate (2020 - Today)')
ax2.legend()
ax2.grid(True)

# Third subplot: DS Dollar Index Data
ax3.plot(Dx_data.index, Dx_data['Close'], label='Close Price')
ax3.set_ylabel('DS Dollar Index Close Price')
ax3.set_title('DS Dollar Index (2020 - Today)')
ax3.legend()
ax3.grid(True)

# Fourth subplot: Bitcoin Daily Close Price
ax4.plot(btc_data.index, btc_data['Close'], label='Bitcoin Close Price', color='orange')
ax4.set_xlabel('Date')
ax4.set_ylabel('Bitcoin Price (USD)')
ax4.set_title('Bitcoin Daily Close Price (2020 - Today)')
ax4.legend()
ax4.grid(True)

# Adjust layout for a clean look
plt.tight_layout()
plt.show()
