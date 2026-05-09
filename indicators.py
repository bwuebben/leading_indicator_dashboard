"""
Indicator Calculation Functions for Leading Indicators Dashboard
Computes z-scores, composite indices, and forecasts
"""

import pandas as pd
import numpy as np


def zscore(series, window=None):
    """
    Calculate z-score for a pandas Series

    Args:
        series: pandas Series
        window: rolling window size (None for full history)

    Returns:
        pandas Series of z-scores
    """
    if window is None:
        mean = series.mean()
        std = series.std()
        return (series - mean) / std
    else:
        rolling = series.rolling(window=window, min_periods=window//2)
        mean = rolling.mean()
        std = rolling.std()
        return (series - mean) / std


def calculate_growth_rates(df, columns, periods=1, annualize=False):
    """
    Calculate period-over-period growth rates

    Args:
        df: DataFrame
        columns: list of column names to calculate growth for
        periods: number of periods (1 for q/q or m/m)
        annualize: if True, annualize quarterly growth rates

    Returns:
        DataFrame with growth rate columns
    """
    result = df.copy()

    for col in columns:
        if col in df.columns:
            growth = df[col].pct_change(periods=periods) * 100
            if annualize and periods == 1:
                # Annualize quarterly growth: ((1 + g)^4 - 1) * 100
                growth = ((1 + growth/100)**4 - 1) * 100
            result[f"{col}_growth"] = growth

    return result


def calculate_moving_average(df, columns, window=3):
    """Calculate moving averages for specified columns"""
    result = df.copy()

    for col in columns:
        if col in df.columns:
            result[f"{col}_ma{window}"] = df[col].rolling(window=window).mean()

    return result


class InventoryLeadingIndex:
    """
    Calculates the Inventory Leading Index (ILI) and related signals
    """

    def __init__(self, alpha_g=0.02, beta_g=0.01, alpha_p=100.0, beta_p=30.0):
        """
        Initialize with regression coefficients

        Args:
            alpha_g, beta_g: GDP forecast coefficients (gdp = alpha + beta * ILI)
            alpha_p, beta_p: Payroll forecast coefficients
        """
        self.alpha_g = alpha_g
        self.beta_g = beta_g
        self.alpha_p = alpha_p
        self.beta_p = beta_p

    def build_inventory_signals(self, monthly_df):
        """
        Build inventory-based signals from monthly data

        Args:
            monthly_df: DataFrame with monthly economic data

        Returns:
            DataFrame with calculated signals and ILI
        """
        df = monthly_df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # --- Inventory/Sales Ratios ---

        # Auto inventory-to-sales ratio
        if 'auto_inv' in df.columns and 'auto_sales' in df.columns:
            df['auto_inv_sales'] = df['auto_inv'] / df['auto_sales']
            df['auto_inv_sales_z'] = zscore(df['auto_inv_sales'], window=60)
            df['auto_inv_sales_z_diff'] = df['auto_inv_sales_z'].diff()
        else:
            df['auto_inv_sales_z'] = 0
            df['auto_inv_sales_z_diff'] = 0

        # Business (non-auto) inventory-to-sales ratio
        if all(col in df.columns for col in ['total_biz_inv', 'total_biz_sales', 'auto_inv', 'auto_sales']):
            non_auto_inv = df['total_biz_inv'] - df['auto_inv']
            non_auto_sales = df['total_biz_sales'] - df['auto_sales']
            df['biz_inv_sales'] = non_auto_inv / non_auto_sales
            df['biz_inv_sales_z'] = zscore(df['biz_inv_sales'], window=60)
            df['biz_inv_sales_z_diff'] = df['biz_inv_sales_z'].diff()
        else:
            df['biz_inv_sales_z'] = 0
            df['biz_inv_sales_z_diff'] = 0

        # --- ISM Survey Signals ---

        # New orders minus inventories (gap signal)
        if 'ism_new_orders' in df.columns and 'ism_inventories' in df.columns:
            df['new_orders_minus_inv'] = df['ism_new_orders'] - df['ism_inventories']
            df['new_orders_minus_inv_z'] = zscore(df['new_orders_minus_inv'], window=60)
        else:
            df['new_orders_minus_inv_z'] = 0

        # Customer inventories (inverted - lower is better)
        if 'ism_customer_inv' in df.columns:
            df['coi_inv'] = -df['ism_customer_inv']  # Invert
            df['coi_inv_z'] = zscore(df['coi_inv'], window=60)
        else:
            df['coi_inv_z'] = 0

        # --- Composite Index ---

        # Sub-factor IA: Inventory accumulation signal (lower is better)
        # Negative diff means inventories declining relative to trend = good
        df['IA'] = (-df['auto_inv_sales_z_diff'] - df['biz_inv_sales_z_diff']) / 2

        # Sub-factor IS: Survey-based inventory signal
        df['IS'] = (df['new_orders_minus_inv_z'] + df['coi_inv_z']) / 2

        # Composite ILI (equal weights)
        df['ILI_raw'] = 0.5 * df['IA'] + 0.5 * df['IS']

        # Standardize ILI to full-history z-score
        df['ILI'] = zscore(df['ILI_raw'])

        return df

    def add_investment_signals(self, quarterly_df):
        """
        Add investment-based growth signals to quarterly data

        Args:
            quarterly_df: DataFrame with quarterly data

        Returns:
            DataFrame with growth rates and z-scores
        """
        df = quarterly_df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Calculate quarter-over-quarter annualized growth rates
        investment_cols = ['res_inv', 'equip_inv', 'ip_inv']
        for col in investment_cols:
            if col in df.columns:
                # q/q annualized: ((value/lag_value)^4 - 1) * 100
                qoq = (df[col] / df[col].shift(1))**4 - 1
                df[f'{col}_qoq_ann'] = qoq * 100

        # Year-over-year for capital goods exports
        if 'cap_goods_exports' in df.columns:
            yoy = df['cap_goods_exports'].pct_change(periods=4) * 100
            df['cap_goods_exports_yoy'] = yoy

        # Year-over-year for durables (if available)
        if 'durables_pce' in df.columns:
            yoy = df['durables_pce'].pct_change(periods=4) * 100
            df['durables_pce_yoy'] = yoy

        # Corporate profits y/y
        if 'corp_profits' in df.columns:
            yoy = df['corp_profits'].pct_change(periods=4) * 100
            df['corp_profits_yoy'] = yoy

        return df

    def add_labor_signals(self, monthly_df):
        """
        Add labor market signals (3-month moving averages)

        Args:
            monthly_df: DataFrame with monthly data

        Returns:
            DataFrame with smoothed labor indicators
        """
        df = monthly_df.copy()

        labor_cols = ['mfg_payrolls', 'transp_equip_jobs']
        for col in labor_cols:
            if col in df.columns:
                df[f'{col}_3mma'] = df[col].rolling(window=3).mean()

        return df

    def forecast_gdp(self, ili_value):
        """
        Forecast next-quarter GDP growth from current ILI

        Args:
            ili_value: Current ILI z-score

        Returns:
            Forecasted GDP growth (q/q annualized, as decimal)
        """
        return self.alpha_g + self.beta_g * ili_value

    def forecast_payrolls(self, ili_value):
        """
        Forecast next-month payroll change from current ILI

        Args:
            ili_value: Current ILI z-score

        Returns:
            Forecasted payroll change (thousands of jobs)
        """
        return self.alpha_p + self.beta_p * ili_value

    def get_regime_color(self, ili_value):
        """
        Get regime color based on ILI value

        Args:
            ili_value: ILI z-score

        Returns:
            Color string for visualization
        """
        if ili_value <= -1.5:
            return 'darkred'
        elif ili_value <= -0.5:
            return 'orange'
        elif ili_value >= 0.5:
            return 'green'
        else:
            return 'gray'

    def get_regime_label(self, ili_value):
        """
        Get regime label based on ILI value

        Args:
            ili_value: ILI z-score

        Returns:
            Human-readable regime label
        """
        if ili_value <= -1.5:
            return 'Recession Risk / Below-Trend'
        elif ili_value <= -0.5:
            return 'Slowdown'
        elif ili_value >= 0.5:
            return 'Above-Trend Growth'
        else:
            return 'Neutral'


def create_heatmap_data(monthly_df, indicators, n_months=6):
    """
    Create heatmap data for dashboard visualization

    Args:
        monthly_df: DataFrame with monthly signals
        indicators: list of indicator column names
        n_months: number of recent months to show

    Returns:
        DataFrame suitable for heatmap plotting
    """
    df = monthly_df.copy()
    df = df.sort_values('date', ascending=False).head(n_months)
    df = df.sort_values('date')

    # Select only requested indicators that exist
    available = [col for col in indicators if col in df.columns]

    heatmap_df = df[['date'] + available].copy()
    heatmap_df = heatmap_df.set_index('date')

    return heatmap_df.T  # Transpose for indicators as rows


if __name__ == "__main__":
    # Test with sample data
    print("Indicator calculation module loaded successfully")
    print("Classes: InventoryLeadingIndex")
    print("Functions: zscore, calculate_growth_rates, create_heatmap_data")
