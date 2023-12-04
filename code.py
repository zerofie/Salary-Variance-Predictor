# @title
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/CSV/Software_Professional_Salaries.csv'

# Specify the encoding (e.g., 'ISO-8859-1')
dataset = pd.read_csv(path, encoding='ISO-8859-1')
dataset.drop(['Company Name'], axis = 1)
x =dataset[['Rating',  'Jobs', 'Salary', 'Reported']].values
x
y = dataset[['Location']].values
y
#handling the missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer( missing_values = np.nan, strategy = "mean" ) #creating variable
imputer = imputer.fit(x[:,2:3])
x[:,2:3]=imputer.transform(x[:,2:3])
x
#converting categorical values into digits
from sklearn.preprocessing import LabelEncoder
label_encoder_x=LabelEncoder()
x[:,1]= label_encoder_x.fit_transform(x[:,1])
x
label_encoder_y=LabelEncoder()
y[:,0]= label_encoder_y.fit_transform(y[:,0])
y
#dummy encoding
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x = onehotencoder.fit_transform(dataset.Jobs.values.reshape(-1,1)).toarray()
x

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Perform label encoding for the 'Jobs' column (assuming it's categorical)
label_encoder = LabelEncoder()
dataset['Jobs'] = label_encoder.fit_transform(dataset['Jobs'])

# Split the dataset into features (X) and the target (y)
X = dataset[['Rating', 'Jobs', 'Reported']]  # Features
y = dataset['Salary']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict salaries on the test set using the Linear Regression model
y_pred_lr = lr_model.predict(X_test)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Linear Regression Results:")
print(f"Mean Squared Error: {mse_lr}")
print(f"R-squared: {r2_lr}")

# Train a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict salaries on the test set using the Random Forest model
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("\nRandom Forest Results:")
print(f"Mean Squared Error: {mse_rf}")
print(f"R-squared: {r2_rf}")
from sklearn.linear_model import Ridge

# Train a Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha (regularization strength)
ridge_model.fit(X_train, y_train)

# Predict salaries using the Ridge model
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the Ridge model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print("Ridge Regression Results:")
print(f"Mean Squared Error: {mse_ridge}")
print(f"R-squared: {r2_ridge}")

from sklearn.feature_selection import SelectKBest, f_regression

# Select the top 'k' features based on F-statistics (you can adjust 'k')
selector = SelectKBest(score_func=f_regression, k=3)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train a model using the selected features (e.g., Ridge regression)
ridge_model.fit(X_train_selected, y_train)

# Evaluate the model
y_pred_ridge_selected = ridge_model.predict(X_test_selected)
mse_ridge_selected = mean_squared_error(y_test, y_pred_ridge_selected)
r2_ridge_selected = r2_score(y_test, y_pred_ridge_selected)
print("Ridge Regression Results (with Feature Selection):")
print(f"Mean Squared Error: {mse_ridge_selected}")
print(f"R-squared: {r2_ridge_selected}")

# Assuming you have already trained your regression model and made predictions
from sklearn.metrics import r2_score

# Calculate R²
r2 = r2_score(y_test, y_pred)

# Convert R² to a percentage
r2_percentage = r2 * 100

print(f"R-squared (as a percentage): {r2_percentage:.2f}%")
