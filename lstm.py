import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

# Preprocessing function
def preprocess_data(data, save_scalers_encoders=False):
    processed_data = data.copy()
    label_encoders = {}
    categorical_cols = ['airline', 'source', 'destination']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        processed_data[col] = label_encoders[col].fit_transform(processed_data[col])
        if save_scalers_encoders:
            joblib.dump(label_encoders[col], f"LSTM_Model/{col}_encoder.pkl")

    # Scale numerical features
    numerical_cols = ['duration', 'stops', 'day_diff', 'day_of_week', 'day_of_month', 'month']
    numerical_scaler = MinMaxScaler()
    processed_data[numerical_cols] = numerical_scaler.fit_transform(processed_data[numerical_cols])
    if save_scalers_encoders:
        joblib.dump(numerical_scaler, "LSTM_Model/numerical_scaler.pkl")
    
    target_scaler = MinMaxScaler()
    processed_data['price'] = target_scaler.fit_transform(processed_data[['price']])
    if save_scalers_encoders:
        joblib.dump(target_scaler, "LSTM_Model/target_scaler.pkl")
    
    return processed_data, label_encoders, numerical_scaler, target_scaler

# Data preprocessing and model functions
def load_and_preprocess_data(dataset_path, load_scalers_encoders=False):
    dataset = pd.read_csv(dataset_path)
    if load_scalers_encoders:
        label_encoders = {}
        target_scaler = joblib.load("LSTM_Model/target_scaler.pkl")
        numerical_scaler = joblib.load("LSTM_Model/numerical_scaler.pkl")
        for col in ['airline', 'source', 'destination']:
            label_encoders[col] = joblib.load(f"LSTM_Model/{col}_encoder.pkl")
    else:
        label_encoders = None
        target_scaler = None
        numerical_scaler = None
    
    processed_data, label_encoders, numerical_scaler, target_scaler = preprocess_data(dataset, save_scalers_encoders=False)

    X = processed_data.drop(columns=['price'])
    y = processed_data['price']


    # Print the preprocessed data
    print("Preprocessed Data:")
    print(processed_data.head())

    return X, y, numerical_scaler, target_scaler, label_encoders

def initial_training(dataset_path):
    X, y, _, _, _ = load_and_preprocess_data(dataset_path)
    train_size = int(len(X) * 0.8)
    train_data = X[:train_size]
    train_target = y[:train_size]
    test_data = X[train_size:]
    test_target = y[train_size:]
    
    train_data = np.reshape(train_data.values, (train_data.shape[0], train_data.shape[1], 1))
    test_data = np.reshape(test_data.values, (test_data.shape[0], test_data.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=100))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(train_data, train_target, epochs=20, batch_size=32)
    model.save('LSTM_Model/lstm_model.h5')

    # Save scalers and encoders
    _, _, numerical_scaler, target_scaler, label_encoders = load_and_preprocess_data(dataset_path)
    joblib.dump(numerical_scaler, "LSTM_Model/numerical_scaler.pkl")
    joblib.dump(target_scaler, "LSTM_Model/target_scaler.pkl")
    for col in ['airline', 'source', 'destination']:
        joblib.dump(label_encoders[col], f"LSTM_Model/{col}_encoder.pkl")

def update_model(dataset_path):
    X, y, _, _, _ = load_and_preprocess_data(dataset_path)
    X = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = load_model('LSTM_Model/lstm_model.h5')
    model.fit(X, y, epochs=20, batch_size=32)
    model.save('LSTM_Model/lstm_model.h5')
    
    # Save scalers and encoders
    _, _, numerical_scaler, target_scaler, label_encoders = load_and_preprocess_data(dataset_path)
    joblib.dump(numerical_scaler, "LSTM_Model/numerical_scaler.pkl")
    joblib.dump(target_scaler, "LSTM_Model/target_scaler.pkl")
    for col in ['airline', 'source', 'destination']:
        joblib.dump(label_encoders[col], f"LSTM_Model/{col}_encoder.pkl")

def predict(dataset_path):
    X, _, numerical_scaler, target_scaler, label_encoders = load_and_preprocess_data(dataset_path)
    X = np.reshape(X.values, (X.shape[0], X.shape[1], 1))
    model = load_model('LSTM_Model/lstm_model.h5')
    predictions = model.predict(X)
    
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    X = numerical_scaler.inverse_transform(X.reshape(-1, X.shape[2]))
    
    print(predictions)


def test_and_evaluate(test_dataset_path):
    X_test, y_test, numerical_scaler, target_scaler, _ = load_and_preprocess_data(test_dataset_path)
    X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

    model = load_model('LSTM_Model/lstm_model.h5')

    predictions = model.predict(X_test)

    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).ravel()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    # Calculate Price Deviation Index (PDI)
    pdi = np.mean((predictions - y_test) / y_test) * 100

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    # Calculate R-Squared (Coefficient of Determination)
    r_squared = r2_score(y_test, predictions)

    threshold = 0.1 * np.mean(y_test)
    correct = np.sum(np.abs(predictions - y_test) <= threshold)
    total = len(y_test)
    accuracy = (correct / total) * 100

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Price Deviation Index (PDI):", pdi)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-Squared (Coefficient of Determination):", r_squared)
    print("Accuracy (within 10% threshold): {:.2f}%".format(accuracy))

"""def test_and_evaluate(test_dataset_path):
    X_test, y_test, numerical_scaler, target_scaler, _ = load_and_preprocess_data(test_dataset_path)
    X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))
    
    model = load_model('LSTM_Model/lstm_model.h5')
    
    predictions = model.predict(X_test)
    
    predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).ravel()
    
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    threshold = 0.1 * np.mean(y_test)
    correct = np.sum(np.abs(predictions - y_test) <= threshold)
    total = len(y_test)
    accuracy = (correct / total) * 100
    
    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Accuracy (within 10% threshold): {:.2f}%".format(accuracy))"""

def predict_price(input_data, model, numerical_scaler, target_scaler, label_encoders):
    assert len(input_data) == 9, "Input data must contain 9 elements: airline, duration, stops, source, destination, day_diff, day_of_week, day_of_month, and month"

    input_df = pd.DataFrame([input_data], columns=['airline', 'duration', 'source', 'destination', 'stops', 'day_diff', 'day_of_week', 'day_of_month', 'month'])

    # Preprocess input data (including encoding categorical features)
    processed_input = input_df.copy()
    categorical_cols = ['airline', 'source', 'destination']
    for col in categorical_cols:
        label_encoder = label_encoders[col]
        processed_input[col] = label_encoder.transform(processed_input[col])

    processed_input[['duration', 'stops', 'day_diff', 'day_of_week', 'day_of_month', 'month']] = numerical_scaler.transform(processed_input[['duration', 'stops', 'day_diff', 'day_of_week', 'day_of_month', 'month']])

    # Reshape input data to match the expected input shape of the model
    processed_input = processed_input.values.reshape((processed_input.shape[0], processed_input.shape[1], 1))

    # Make predictions using the loaded model
    prediction = model.predict(processed_input)

    # Inverse scaling for prediction
    inverse_prediction = target_scaler.inverse_transform(prediction)[0][0]

    return inverse_prediction


"""# Get today's date
today = datetime.now()

# Format today's date in 'DD-MM-YYYY' format
today_str = today.strftime('%d-%m-%Y')

# Subtract one day from today's date to get yesterday's date
yesterday = today - timedelta(days=1)

# Format yesterday's date in 'DD-MM-YYYY' format
yesterday_str = yesterday.strftime('%d-%m-%Y')
    
# Include today's date in the filename
tod = f'Dataset/flight_data_{today_str}.csv'
yest = f'Dataset/flight_data_{yesterday_str}.csv'
"""
# User interaction
#initial_training("fake/1.csv")
#update_model("fake/7.csv")
# predict("3.csv")
#test_and_evaluate("fake/8.csv")
#predict_price(["Air India",2910,"COK","CCU",2,0,0,15,4])
