# Part 2: Machine Learning-Based Stock Trading Strategy with Optimized LSTM

## Overview
This program uses machine learning to develop a robust trading/investing strategy. By processing historical stock data and integrating advanced technical analysis indicators, it aims to predict stock movements, evaluate market trends, and simulate strategy performance. Additionally, it includes a "Golden Magic Model" powered by Optuna for hyperparameter optimization and feature weighting.

## Features
- Reads and processes historical stock data from a CSV file.
- Adds advanced technical analysis indicators (e.g., SMA, EMA, RSI, MACD).
- Implements multiple models, including a standard LSTM and the optimized "Golden Magic Model."
- Compares strategy performance against traditional buy-and-hold.
- Visualizes cumulative returns for comprehensive analysis.
- Includes Optuna-based hyperparameter optimization.

## Prerequisites

### Libraries
Make sure the following Python libraries are installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `ta`
- `optuna`

To install these libraries, run:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow ta optuna
```

### File
Ensure you have the necessary CSV file containing historical stock data. The file should include columns like `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. Update the file path in the script to match your file location.

## Usage

### 1. **Set Up File Path**
Update the `file_path` variable in the script to point to your CSV file. Example:
```python
file_path = r"C:\path\to\your\file.csv"
```

### 2. **Run the Script**
Execute the script to read, preprocess, and feature-engineer the data. The key steps include:
- Loading the CSV file.
- Adding advanced features (e.g., SMA, EMA, RSI, MACD).
- Preparing the data for LSTM and Golden Magic Model training.

### 3. **Feature Engineering**
The script calculates additional technical indicators:
```python
def add_features(df):
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], window=50).sma_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['52W_High'] = df['Close'].rolling(window=252).max()
    df['52W_Low'] = df['Close'].rolling(window=252).min()
    df['52W_High_Ratio'] = df['Close'] / df['52W_High']
    df['52W_Low_Ratio'] = df['Close'] / df['52W_Low']
    return df
```

### 4. **Prepare Data for LSTM**
The program scales the data using `MinMaxScaler` and prepares sequences for LSTM modeling:
```python
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 3])  # Index 3 corresponds to 'Close'
    return np.array(X), np.array(y)
```
- Train and test datasets are created with a lookback window of 60 days.

### 5. **Train the LSTM Model**
The script builds and trains an LSTM model:
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
```

### 6. **Golden Magic Model**
This optimized model integrates Optuna for hyperparameter tuning and feature weighting:
```python
import optuna

# Define the model
def build_model(trial):
    model = Sequential([
        LSTM(trial.suggest_int("units_lstm1", 30, 100), return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(trial.suggest_float("dropout_lstm1", 0.1, 0.5)),
        LSTM(trial.suggest_int("units_lstm2", 30, 100), return_sequences=False),
        Dropout(trial.suggest_float("dropout_lstm2", 0.1, 0.5)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=trial.suggest_float("lr", 1e-4, 1e-2, log=True)), loss='mse')
    return model

# Optuna optimization
def objective(trial):
    model = build_model(trial)
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
    val_loss = history.history['val_loss'][-1]
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
final_model = build_model(optuna.trial.FixedTrial(best_params))
final_model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
```

### 7. **Make Predictions**
The script generates predictions and evaluates the strategy:
```python
predictions = final_model.predict(X_test)

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(np.concatenate((X_test[:, -1, :3], predictions, X_test[:, -1, 4:]), axis=1))[:, 3]
actual = scaler.inverse_transform(np.concatenate((X_test[:, -1, :3], y_test.reshape(-1, 1), X_test[:, -1, 4:]), axis=1))[:, 3]

# Calculate returns
test_returns = pd.Series(actual).pct_change()
model_positions = np.where(predictions[:-1] < predictions[1:], 1, -1)

# Ensure model_positions and test_returns have the same length
min_length = min(len(model_positions), len(test_returns))
model_positions = model_positions[:min_length]
test_returns = test_returns[:min_length]

model_returns = test_returns * model_positions

# Calculate cumulative returns
cumulative_test_returns2 = (1 + test_returns).cumprod()
cumulative_model_returns2 = (1 + model_returns).cumprod()

# Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[seq_length+1:seq_length+1+min_length], cumulative_test_returns2, label='Buy and Hold')
plt.plot(test_data.index[seq_length+1:seq_length+1+min_length], cumulative_model_returns, label='LSTM Model')
plt.plot(test_data.index[seq_length+1:seq_length+1+min_length], cumulative_model_returns2, label='Optimized LSTM Model')
plt.title('Cumulative Returns: Optimized LSTM Model vs STD. LSTM vs Buy and Hold (2024)')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Print performance metrics
model_total_return = cumulative_model_returns.iloc[-1] - 1
optimized_model_total_return = cumulative_model_returns2.iloc[-1] - 1
buy_hold_return = cumulative_test_returns.iloc[-1] - 1

print(f"Optimized LSTM Model Return: {optimized_model_total_return:.2%}")
print(f"LSTM Model Return: {model_total_return:.2%}")
print(f"Buy and Hold Return: {buy_hold_return:.2%}")
```

### 8. **Visualize Results**
The cumulative returns are plotted for comparison:
```python
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[seq_length+1:seq_length+1+min_length], cumulative_test_returns, label='Buy and Hold')
plt.plot(test_data.index[seq_length+1:seq_length+1+min_length], cumulative_model_returns, label='LSTM Strategy')
plt.title('Cumulative Returns: Buy and Hold vs LSTM Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
```

### 9. **Performance Metrics**
The program calculates and displays performance metrics:
```python
model_total_return = cumulative_model_returns.iloc[-1] - 1
buy_hold_return = cumulative_test_returns.iloc[-1] - 1

print(f"LSTM Model Return: {model_total_return:.2%}")
print(f"Buy and Hold Return: {buy_hold_return:.2%}")
```

## Example Input Data
Ensure your CSV file is structured as follows:

| Date       | Open   | High   | Low    | Close  | Volume   |
|------------|--------|--------|--------|--------|----------|
| 2019-12-11 | 34.843 | 34.990 | 34.600 | 34.630 | 1740162  |
| 2019-12-12 | 34.533 | 34.827 | 34.403 | 34.710 | 1253115  |

## Future Enhancements
- Incorporate risk-adjusted performance metrics (e.g., Sharpe Ratio).
- Test on additional time periods for robustness.
- Add transaction cost modeling.
- Explore ensemble models and advanced deep learning architectures for enhanced prediction accuracy.


**Back to [Part 1](README.md)**
