# stock-prediction
code that uses historical stock price data to train a linear regression model and make future price predictions:
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data into a Pandas dataframe
df = pd.read_csv('stock_data.csv')

# Preprocess data by dropping any missing values and selecting relevant columns
df = df.dropna()
df = df[['Date', 'Close']]

# Define the features and target variable
X = np.array(df['Close'].shift(1)).reshape(-1, 1)  # yesterday's closing price
y = np.array(df['Close'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R2 score:', r2)

# Make predictions for future stock prices
last_price = np.array(df.tail(1)['Close']).reshape(-1, 1)
next_day_price = model.predict(last_price)
print('Predicted price for tomorrow:', next_day_price)

We load historical stock price data from a CSV file and preprocess it by dropping any missing values and selecting the relevant columns. We then define yesterday's closing price as the feature and today's closing price as the target variable. We split the data into training and testing sets, and train a linear regression model on the training data. We evaluate the model on the testing data using mean squared error and R2 score. Finally, we use the trained model to make a prediction for the next day's stock price based on the most recent closing price.


Certainly! Here is a step-by-step explanation of the provided Python code:

●	The necessary libraries are imported at the beginning of the code. These include Pandas, NumPy, LinearRegression, train_test_split, mean_squared_error, and r2_score.
●	The code reads the CSV file 'stock_data.csv' and loads it into a Pandas dataframe called df.
●	The df dataframe is preprocessed by dropping any missing values and selecting only the 'Date' and 'Close' columns.
●	The X and y variables are defined as the features and target variable, respectively. The X variable is set to yesterday's closing price, shifted by 1 day, and reshaped to be a 2D NumPy array. The y variable is set to the closing price.
●	The data is split into training and testing sets using the train_test_split function. The test_size argument is set to 0.2, meaning that 20% of the data will be used for testing and 80% will be used for training. The shuffle argument is set to False, meaning that the data is not shuffled before splitting.
●	A linear regression model is trained on the training data using the LinearRegression function.
●	The trained model is evaluated on the testing data using the predict method. The mean squared error (MSE) and R-squared (R2) score are calculated using the mean_squared_error and r2_score functions, respectively.
●	The MSE and R2 scores are printed to the console.
●	The last_price variable is set to the closing price of the last day in the df dataframe, reshaped to be a 2D NumPy array.
●	The next_day_price variable is set to the predicted closing price for the next day using the predict method of the trained model.
●	The predicted closing price for the next day is printed to the console.

Overall, this code loads and preprocesses stock price data, trains a linear regression model on the data, evaluates the model's performance on a testing set, and makes a prediction for the next day's closing price based on the trained model.

