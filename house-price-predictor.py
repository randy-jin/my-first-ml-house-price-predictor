from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import utils

# Load data from csv file
df = pd.read_csv("synthetic_house_data.csv")

test_squre_meter = [150,2]

# ✅ Prepare Data
# X = np.array([[80], [100], [120], [130], [140], [150], [180]])
# y = np.array([300, 380, 450, 480, 510, 540, 560])
X = df[["GrLivArea", "GarageCars"]].values
y = df["SalePrice"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Linear Regression
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

predicted_linear_price = model_linear.predict([test_squre_meter])
print(f"Linear Predicted price: {predicted_linear_price[0]:.2f} (in 10,000s)")

# 🔍 Model Evaluation on test set
linear_preds = model_linear.predict(X_test)
utils.evaluate_regression_model(y_test, linear_preds)

# ✅ Random Forest
model_randomforest = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
model_randomforest.fit(X_train, y_train)

predicted_randomforest_price = model_randomforest.predict([test_squre_meter])
print(f"RandomForest Predicted price: {predicted_randomforest_price[0]:.2f} (in 10,000s)")

# 🔍 Model Evaluation on test set
rf_preds = model_randomforest.predict(X_test)
utils.evaluate_regression_model(y_test, rf_preds)

# call
utils.plot_predictions(y_test, linear_preds, model_name="Linear Regression")
utils.plot_predictions(y_test, rf_preds, model_name="Random Forest")

# call
utils.plot_model_trends(model_linear, model_randomforest)