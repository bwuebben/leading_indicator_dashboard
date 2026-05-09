"""
Backtesting module for Leading Indicators Dashboard
Generates historical predictions and compares with actuals
"""

import pandas as pd
import numpy as np
from indicators import InventoryLeadingIndex


def align_monthly_to_quarterly(monthly_df, quarterly_df):
    """
    Align monthly ILI values with quarterly data for forecasting

    Args:
        monthly_df: DataFrame with monthly data and ILI
        quarterly_df: DataFrame with quarterly GDP data

    Returns:
        DataFrame with aligned ILI and quarterly values
    """
    monthly_df = monthly_df.copy()
    quarterly_df = quarterly_df.copy()

    # Convert dates to period for matching
    monthly_df['quarter'] = monthly_df['date'].dt.to_period('Q')
    quarterly_df['quarter'] = quarterly_df['date'].dt.to_period('Q')

    # Get last month of each quarter (use the ILI from the last month)
    quarterly_ili = monthly_df.groupby('quarter').last()[['ILI']].reset_index()

    # Merge with quarterly data on period
    merged = pd.merge(quarterly_df, quarterly_ili, on='quarter', how='left')

    # Drop the helper column
    merged = merged.drop('quarter', axis=1)

    return merged


def generate_gdp_predictions(monthly_df, quarterly_df, alpha_g, beta_g):
    """
    Generate historical GDP predictions using ILI

    Args:
        monthly_df: DataFrame with monthly signals including ILI
        quarterly_df: DataFrame with quarterly GDP data
        alpha_g, beta_g: Regression coefficients

    Returns:
        DataFrame with predictions and actuals
    """
    # Align ILI with quarterly data
    df = align_monthly_to_quarterly(monthly_df, quarterly_df)

    # Calculate quarter-over-quarter growth rate (annualized)
    df = df.sort_values('date')
    df['real_gdp_qoq_ann'] = ((df['real_gdp'] / df['real_gdp'].shift(1))**4 - 1) * 100

    # Generate predictions using lagged ILI (predict next quarter)
    df['ILI_lag1'] = df['ILI'].shift(1)
    df['gdp_predicted'] = alpha_g + beta_g * df['ILI_lag1']
    df['gdp_actual'] = df['real_gdp_qoq_ann']

    # Calculate prediction error
    df['gdp_error'] = df['gdp_predicted'] - df['gdp_actual']

    return df[['date', 'ILI_lag1', 'gdp_predicted', 'gdp_actual', 'gdp_error']].dropna()


def generate_payroll_predictions(monthly_df, alpha_p, beta_p):
    """
    Generate historical payroll predictions using ILI

    Args:
        monthly_df: DataFrame with monthly signals including ILI and payrolls
        alpha_p, beta_p: Regression coefficients

    Returns:
        DataFrame with predictions and actuals
    """
    df = monthly_df.copy()
    df = df.sort_values('date')

    # Calculate month-over-month payroll change (in thousands)
    if 'mfg_payrolls' in df.columns:
        df['payroll_change'] = df['mfg_payrolls'].diff()

        # Generate predictions using lagged ILI (predict next month)
        df['ILI_lag1'] = df['ILI'].shift(1)
        df['payroll_predicted'] = alpha_p + beta_p * df['ILI_lag1']
        df['payroll_actual'] = df['payroll_change']

        # Calculate prediction error
        df['payroll_error'] = df['payroll_predicted'] - df['payroll_actual']

        return df[['date', 'ILI_lag1', 'payroll_predicted', 'payroll_actual', 'payroll_error']].dropna()
    else:
        return pd.DataFrame()


def estimate_coefficients(monthly_df, quarterly_df):
    """
    Estimate regression coefficients from historical data

    Args:
        monthly_df: DataFrame with monthly signals including ILI
        quarterly_df: DataFrame with quarterly GDP data

    Returns:
        Dictionary with estimated coefficients and R-squared values
    """
    results = {}

    # GDP regression
    gdp_data = align_monthly_to_quarterly(monthly_df, quarterly_df)
    gdp_data = gdp_data.sort_values('date')
    gdp_data['real_gdp_qoq_ann'] = ((gdp_data['real_gdp'] / gdp_data['real_gdp'].shift(1))**4 - 1) * 100
    gdp_data['ILI_lag1'] = gdp_data['ILI'].shift(1)

    valid_gdp = gdp_data[['ILI_lag1', 'real_gdp_qoq_ann']].dropna()

    if len(valid_gdp) > 10:
        # Simple linear regression
        X = valid_gdp['ILI_lag1'].values
        y = valid_gdp['real_gdp_qoq_ann'].values

        # Calculate coefficients
        x_mean = X.mean()
        y_mean = y.mean()

        beta_g = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean)**2)
        alpha_g = y_mean - beta_g * x_mean

        # Calculate R-squared
        y_pred = alpha_g + beta_g * X
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y_mean)**2)
        r2_gdp = 1 - (ss_res / ss_tot)

        results['alpha_g'] = alpha_g
        results['beta_g'] = beta_g
        results['r2_gdp'] = r2_gdp
    else:
        results['alpha_g'] = 0.02
        results['beta_g'] = 0.01
        results['r2_gdp'] = 0.0

    # Payroll regression
    payroll_data = monthly_df.copy()
    payroll_data = payroll_data.sort_values('date')

    if 'mfg_payrolls' in payroll_data.columns:
        payroll_data['payroll_change'] = payroll_data['mfg_payrolls'].diff()
        payroll_data['ILI_lag1'] = payroll_data['ILI'].shift(1)

        valid_payroll = payroll_data[['ILI_lag1', 'payroll_change']].dropna()

        if len(valid_payroll) > 10:
            X = valid_payroll['ILI_lag1'].values
            y = valid_payroll['payroll_change'].values

            x_mean = X.mean()
            y_mean = y.mean()

            beta_p = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean)**2)
            alpha_p = y_mean - beta_p * x_mean

            y_pred = alpha_p + beta_p * X
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y_mean)**2)
            r2_payroll = 1 - (ss_res / ss_tot)

            results['alpha_p'] = alpha_p
            results['beta_p'] = beta_p
            results['r2_payroll'] = r2_payroll
        else:
            results['alpha_p'] = 0.0
            results['beta_p'] = 10.0
            results['r2_payroll'] = 0.0
    else:
        results['alpha_p'] = 0.0
        results['beta_p'] = 10.0
        results['r2_payroll'] = 0.0

    return results


def calculate_prediction_metrics(predicted, actual):
    """
    Calculate prediction accuracy metrics

    Args:
        predicted: array of predicted values
        actual: array of actual values

    Returns:
        Dictionary with metrics
    """
    errors = predicted - actual

    metrics = {
        'mae': np.mean(np.abs(errors)),
        'rmse': np.sqrt(np.mean(errors**2)),
        'bias': np.mean(errors),
        'correlation': np.corrcoef(predicted, actual)[0, 1] if len(predicted) > 1 else 0
    }

    return metrics


if __name__ == "__main__":
    print("Backtesting module loaded successfully")
    print("Functions: estimate_coefficients, generate_gdp_predictions, generate_payroll_predictions")
