import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# Define the ticker symbol for Nasdaq-100 Futures
ticker = "NQ=F"

# Set the start and end dates
start_date = "2020-01-01"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Download historical data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Plot the close price vs. time
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Nasdaq-100 Futures Close Price (2020 - Today)')
plt.legend()
plt.grid(True)
plt.show()
