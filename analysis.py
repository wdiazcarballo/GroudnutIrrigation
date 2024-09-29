# Load the preprocessed data (since the path is already known)
data = pd.read_csv('/mnt/data/daily_average_with_et0_etc_groundnut_data.csv')

# Separate the features (X) and target (ETC)
X = data[['Avg_Temperature', 'Avg_Humidity', 'Avg_Soil_Moisture', 'Light']]
y = data['ETC']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Support Vector Regression
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))

# Gradient Boosting Regression
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

# Compile results
model_results = {
    'Linear Regression': {'MAE': mae_linear, 'RMSE': rmse_linear},
    'Random Forest Regression': {'MAE': mae_rf, 'RMSE': rmse_rf},
    'Support Vector Regression': {'MAE': mae_svr, 'RMSE': rmse_svr},
    'Gradient Boosting Regression': {'MAE': mae_gb, 'RMSE': rmse_gb}
}

# Display the results
import ace_tools as tools; tools.display_dataframe_to_user(name="Regression Model Comparison", dataframe=pd.DataFrame(model_results).T)

model_results
