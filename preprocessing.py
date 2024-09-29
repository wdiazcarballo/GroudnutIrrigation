import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('groundnut-kk-2024summerseason.csv')

print(data.head())  # Shows the first 5 rows of the dataset
print(data.isnull().sum())  # Displays the number of missing values per column
print(data.dtypes)  # Shows the data type of each column
print(data.shape)  # Returns (number of rows, number of columns)
print(data.sample(5))  # Randomly samples 5 rows from the dataset
print(data.columns)  # Displays all column names
# Fix the date by converting from the Thai Buddhist calendar to the Gregorian calendar
def convert_thai_to_gregorian(date_str):
    day, month, year = date_str.split('/')
    year = str(int(year) - 543)  # Convert Thai year to Gregorian year
    return f"{day}/{month}/{year}"

# Apply the conversion to the 'Date' column
data['Date'] = data['Date'].apply(convert_thai_to_gregorian)

# Combine the 'Date' and 'Time' columns into a 'Datetime' column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Selecting the relevant columns for ANN and LSTM models
data = data[['Datetime', 'Avg-Temperature', 'Rh-Average', 'M-Average', 'Light-01']]

# Renaming columns for clarity
data.columns = ['Datetime', 'Avg_Temperature', 'Avg_Humidity', 'Avg_Soil_Moisture', 'Light']

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data[['Avg_Temperature', 'Avg_Humidity', 'Avg_Soil_Moisture', 'Light']] = scaler.fit_transform(data[['Avg_Temperature', 'Avg_Humidity', 'Avg_Soil_Moisture', 'Light']])

# Save the preprocessed data
data.to_csv('preprocessed_groundnut_data.csv', index=False)
