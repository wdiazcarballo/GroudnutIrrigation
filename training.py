import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load the preprocessed data
data = pd.read_csv('daily_average_with_et0_etc_groundnut_data.csv')

# Separate the features (X) and target (assuming 'ETC' is the target)
X = data[['Avg_Temperature', 'Avg_Humidity', 'Avg_Soil_Moisture', 'Light']]
y = data['ETC']  # Replace 'ETC' with the actual target column if available

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ANN Model
def build_ann():
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer for ETC prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the ANN model
ann_model = build_ann()
ann_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict using ANN model
y_pred_ann = ann_model.predict(X_test)

# LSTM Model
def build_lstm():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))  # Output layer for ETC prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Reshape data for LSTM
X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the LSTM model
lstm_model = build_lstm()
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

# Predict using LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Model evaluation using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))

# Print the results
print(f"ANN MAE: {mae_ann}, RMSE: {rmse_ann}")
print(f"LSTM MAE: {mae_lstm}, RMSE: {rmse_lstm}")
