"""
Meteorological Visualization & Plotting Suite (CO3 - CO5).
Generates publication & broadcast-quality graphics using Matplotlib and Seaborn:
1. Multi-parameter historical trend charts with rolling moving averages
2. Classical seasonal decomposition plots (Observed, Trend, Seasonal, Residual)
3. Monthly & seasonal distribution boxplots
4. Inter-parameter correlation heatmap
5. Anomaly timeline & hazard alerts scatter plot
6. Actual vs. Predicted time-series evaluation charts
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set clean professional styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams["figure.autolayout"] = True


def plot_multi_parameter_trends(df: pd.DataFrame, output_path=None):
    """
    Plots multi-panel line chart of historical time series with 30-day smoothed moving average.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "multi_parameter_trends.png")

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    metrics = [
        ("temperature_mean", "Temperature (°C)", "#e63946"),
        ("relative_humidity", "Relative Humidity (%)", "#457b9d"),
        ("dew_point", "Moisture / Dew Point (°C)", "#2a9d8f"),
        ("surface_pressure", "Atmospheric Pressure (hPa)", "#6a4c93"),
        ("wind_speed", "Wind Speed (km/h)", "#f4a261")
    ]

    for ax, (col, label, color) in zip(axes, metrics):
        if col in df.columns:
            ax.plot(df["date"], df[col], color=color, alpha=0.35, linewidth=0.8, label="Daily Observation")
            # 30-day rolling mean
            roll_30 = df[col].rolling(window=30, center=True).mean()
            ax.plot(df["date"], roll_30, color=color, linewidth=2.2, label="30-Day Moving Average")
            ax.set_ylabel(label, fontsize=11, fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Date", fontsize=12, fontweight="bold")
    fig.suptitle("Decadal Weather Parameter Trends & Moving Averages (2015–2024)", fontsize=15, fontweight="bold", y=0.995)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_seasonal_decomposition(decomp_df: pd.DataFrame, target_name="Temperature (°C)", output_path=None):
    """
    Plots 4-panel decomposition: Observed, Trend, Seasonal, Residual.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "seasonal_decomposition.png")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(decomp_df.index, decomp_df["observed"], color="#1d3557", linewidth=1.0)
    axes[0].set_ylabel("Observed", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Time-Series Additive Decomposition: {target_name}", fontsize=13, fontweight="bold")

    axes[1].plot(decomp_df.index, decomp_df["trend"], color="#e63946", linewidth=2.0)
    axes[1].set_ylabel("Trend", fontsize=11, fontweight="bold")

    axes[2].plot(decomp_df.index, decomp_df["seasonal"], color="#2a9d8f", linewidth=1.2)
    axes[2].set_ylabel("Seasonal", fontsize=11, fontweight="bold")

    axes[3].scatter(decomp_df.index, decomp_df["residual"], color="#6c757d", s=4, alpha=0.6)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Residual", fontsize=11, fontweight="bold")
    axes[3].set_xlabel("Date", fontsize=11, fontweight="bold")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_monthly_boxplots(df: pd.DataFrame, output_path=None):
    """
    Generates monthly distribution boxplots showing annual seasonality.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "monthly_seasonal_boxplots.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    params = [
        ("temperature_mean", "Temperature (°C)", "Reds"),
        ("relative_humidity", "Relative Humidity (%)", "Blues"),
        ("dew_point", "Moisture / Dew Point (°C)", "YlGnBu"),
        ("rain", "Rainfall (mm)", "PuBuGn")
    ]

    for ax, (col, title, palette) in zip(axes, params):
        if col in df.columns and "month" in df.columns:
            sns.boxplot(data=df, x="month", y=col, ax=ax, palette=palette, showfliers=False)
            ax.set_title(f"Monthly Distribution: {title}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Month (1 = Jan ... 12 = Dec)", fontsize=10)
            ax.set_ylabel(title, fontsize=10)

    fig.suptitle("Seasonal Variations & Monthly Recurring Patterns", fontsize=15, fontweight="bold", y=0.99)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path=None):
    """
    Plots a formatted correlation heatmap.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")

    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Inter-Parameter Meteorological Correlation Matrix", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_anomalies_and_hazards(anomaly_df: pd.DataFrame, output_path=None):
    """
    Plots temperature anomalies with marked hazard events.
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "anomalies_and_hazards.png")

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(anomaly_df["date"], anomaly_df["temperature_mean"], color="#4a5568", alpha=0.5, label="Temperature Observed")
    if "clim_mean" in anomaly_df.columns:
        ax.plot(anomaly_df["date"], anomaly_df["clim_mean"], color="#3182ce", linestyle="--", linewidth=1.5, label="Climatological Normal")

    # Mark hazard events
    hazards = anomaly_df[anomaly_df["has_hazard"] == 1]
    if len(hazards) > 0:
        ax.scatter(hazards["date"], hazards["temperature_mean"], color="#e53e3e", s=25, label=f"Extreme Hazard Events (N={len(hazards)})", zorder=5)

    ax.set_title("Historical Climatological Baseline & Detected Extreme Weather Anomalies", fontsize=13, fontweight="bold")
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(loc="upper left")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def plot_predictions_vs_actual(test_df: pd.DataFrame, predictions_dict: dict, output_path=None):
    """
    Visualizes predicted vs actual out-of-sample values across the test period (2024).
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "predictions_vs_actual_2024.png")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    metrics = [
        ("temperature_mean", "Temperature (°C)", "#e63946"),
        ("relative_humidity", "Relative Humidity (%)", "#457b9d"),
        ("dew_point", "Moisture / Dew Point (°C)", "#2a9d8f"),
        ("surface_pressure", "Atmospheric Pressure (hPa)", "#6a4c93"),
        ("wind_speed", "Wind Speed (km/h)", "#f4a261"),
        ("rain", "Precipitation (mm)", "#0077b6")
    ]

    for ax, (target, label, color) in zip(axes, metrics):
        if target in test_df.columns and target in predictions_dict:
            y_true = test_df[target].values
            y_pred = predictions_dict[target]

            ax.plot(test_df["date"], y_true, color="gray", alpha=0.6, label="Actual Ground Truth", linewidth=1.2)
            ax.plot(test_df["date"], y_pred, color=color, label="ML Model Prediction", linewidth=1.4)
            ax.set_ylabel(label, fontsize=10, fontweight="bold")
            ax.legend(loc="upper right", framealpha=0.8, fontsize=9)

    axes[-2].set_xlabel("Date (2024 Test Period)", fontsize=11)
    axes[-1].set_xlabel("Date (2024 Test Period)", fontsize=11)
    fig.suptitle("Out-of-Sample Test Evaluation: Ground Truth vs. Machine Learning Forecast (2024)", fontsize=14, fontweight="bold", y=0.995)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
    return output_path
