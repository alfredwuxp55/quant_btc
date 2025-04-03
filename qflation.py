import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

# Define the date range
start_date = "2020-01-01"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Retrieve the 10-Year Breakeven Inflation Rate from FRED
breakeven = web.DataReader('T10YIE', 'fred', start_date, end_date)

# Print the first few rows to inspect the data
print(breakeven)

# Plot the Breakeven Inflation Rate vs. Time
plt.figure(figsize=(12, 6))
plt.plot(breakeven.index, breakeven['T10YIE'], label='10-Year Breakeven Inflation Rate', color='blue')
plt.xlabel('Date')
plt.ylabel('Inflation Rate (%)')
plt.title('10-Year Breakeven Inflation Rate (2020 - Today)')
plt.legend()
plt.grid(True)
plt.show()
