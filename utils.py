from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def evaluate_regression_model(y_true, y_pred):
    """
    Print and return evaluation metrics for a regression model.
    Parameters:
        y_true: Actual values (array or list)
        y_pred: Predicted values from the model (array or list)
    Returns:
        A dictionary containing MAE, MSE, RMSE, R2, and MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("📊 Model Evaluation Metrics:")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, model_name="Model"):
    os.makedirs("images", exist_ok=True)
    filename = os.path.join("images", f"{model_name.lower().replace(' ', '_')}_prediction_plot.png")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label="Ideal")
    plt.xlabel("Actual Price (in 10,000s)")
    plt.ylabel("Predicted Price (in 10,000s)")
    plt.title(f"{model_name} - Predicted vs Actual Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved chart: {filename}")


def plot_model_trends(model_linear, model_rf, garage_fixed=2, save_path="images/model_trend_comparison.png"):
    """
    Plot prediction trends of LinearRegression and RandomForest models across different house sizes.

    Parameters:
        model_linear: Trained Linear Regression model
        model_rf: Trained Random Forest Regression model
        garage_fixed: Fiexed number of garage spaces (default is 2)
        save_path: File path to save the output image
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a test range of house sizes (from 60 square meters to 260)
    square_meters = np.linspace(60, 260, 100)
    X_test_range = np.column_stack((square_meters, [garage_fixed] * len(square_meters)))

    # Model predictions
    y_pred_linear = model_linear.predict(X_test_range)
    y_pred_rf = model_rf.predict(X_test_range)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(square_meters, y_pred_linear, label="Linear Regression", color='blue')
    plt.plot(square_meters, y_pred_rf, label="Random Forest", color='green')
    plt.xlabel("House Size (㎡)")
    plt.ylabel("Predicted Price (in 10,000s)")
    plt.title(f"Prediction Trend: Linear vs Random Forest (GarageCars={garage_fixed})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Saving images
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Trend chart saved to: {save_path}")