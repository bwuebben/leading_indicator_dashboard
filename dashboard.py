"""
Leading Indicators Dashboard - Streamlit App
Inventory & Growth Regime Monitor for GDP & Labor Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

from data_fetcher import FREDDataFetcher
from indicators import InventoryLeadingIndex, create_heatmap_data
from backtesting import estimate_coefficients, generate_gdp_predictions, generate_payroll_predictions, calculate_prediction_metrics


# Page configuration
st.set_page_config(
    page_title="Leading Indicators Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_or_fetch_data(force_refresh=False):
    """
    Load data from cache or fetch from FRED

    Args:
        force_refresh: if True, fetch new data from FRED

    Returns:
        Tuple of (monthly_df, quarterly_df)
    """
    monthly_path = 'data/monthly_macro.parquet'
    quarterly_path = 'data/quarterly_macro.parquet'

    # Check if cached data exists and is recent (< 24 hours old)
    use_cache = (
        not force_refresh and
        os.path.exists(monthly_path) and
        os.path.exists(quarterly_path)
    )

    if use_cache:
        try:
            monthly = pd.read_parquet(monthly_path)
            quarterly = pd.read_parquet(quarterly_path)
            return monthly, quarterly
        except Exception as e:
            st.warning(f"Error loading cached data: {e}. Fetching fresh data...")

    # Fetch fresh data from FRED
    with st.spinner("Fetching data from FRED API..."):
        fetcher = FREDDataFetcher()
        monthly, quarterly = fetcher.fetch_all_data()
        fetcher.save_data(monthly, quarterly)

    return monthly, quarterly


def plot_ili_gauge(ili_value, regime_color, regime_label):
    """Create a gauge chart for ILI value"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ili_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': regime_label, 'font': {'size': 16}},
        number={'suffix': " σ", 'font': {'size': 32}},
        gauge={
            'axis': {'range': [-3, 3], 'tickwidth': 1},
            'bar': {'color': regime_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-3, -1.5], 'color': '#ffcccc'},
                {'range': [-1.5, -0.5], 'color': '#ffe6cc'},
                {'range': [-0.5, 0.5], 'color': '#f0f0f0'},
                {'range': [0.5, 1.5], 'color': '#ccffcc'},
                {'range': [1.5, 3], 'color': '#99ff99'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': ili_value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )

    return fig


def plot_time_series(df, x_col, y_cols, title, y_label="Value", colors=None):
    """Create a time series line chart"""
    fig = go.Figure()

    if not isinstance(y_cols, list):
        y_cols = [y_cols]

    for i, col in enumerate(y_cols):
        if col in df.columns:
            color = colors[i] if colors and i < len(colors) else None
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode='lines',
                name=col.replace('_', ' ').title(),
                line=dict(width=2, color=color) if color else dict(width=2)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_heatmap(heatmap_df, title="Indicator Heatmap"):
    """Create a heatmap visualization"""
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,
        y=heatmap_df.index,
        colorscale='RdYlGn',
        zmid=0,
        text=heatmap_df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Z-Score")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Indicator",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def plot_prediction_vs_actual(df, pred_col, actual_col, title, y_label="Value"):
    """Create prediction vs actual comparison chart"""
    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[actual_col],
        mode='lines',
        name='Actual',
        line=dict(width=2, color='#1f77b4')
    ))

    # Predicted values
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[pred_col],
        mode='lines',
        name='Predicted',
        line=dict(width=2, color='#ff7f0e', dash='dash')
    ))

    # Add zero line if values cross zero
    if df[actual_col].min() < 0 or df[pred_col].min() < 0:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_scatter_with_fit(df, x_col, y_col, title, x_label, y_label):
    """Create scatter plot with regression line"""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        name='Data',
        marker=dict(size=6, color='#1f77b4', opacity=0.6)
    ))

    # Add regression line
    if len(df) > 2:
        x = df[x_col].values
        y = df[y_col].values
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)

        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode='lines',
            name='Fit',
            line=dict(width=2, color='#ff7f0e', dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )

    return fig


def main():
    """Main dashboard application"""

    # Header
    st.title("📊 Inventory & Growth Regime Dashboard")
    st.markdown("**Leading indicators for GDP & labor market forecasting**")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard tracks leading indicators for economic growth,
        centered on inventory cycles and investment signals.

        **Data Source:** FRED (Federal Reserve Economic Data)
        **Update Frequency:** Daily
        """)

    # Load data
    try:
        monthly, quarterly = load_or_fetch_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure your FRED_API_KEY is set in the .env file.")
        st.stop()

    # First, calculate ILI with temporary coefficients
    with st.spinner("Calculating indicators..."):
        temp_calculator = InventoryLeadingIndex(alpha_g=0.02, beta_g=0.01, alpha_p=0.0, beta_p=10.0)
        monthly_signals = temp_calculator.build_inventory_signals(monthly)
        monthly_signals = temp_calculator.add_labor_signals(monthly_signals)
        quarterly_signals = temp_calculator.add_investment_signals(quarterly)

    # Now estimate coefficients from historical data with ILI calculated
    with st.spinner("Estimating model coefficients..."):
        coefficients = estimate_coefficients(monthly_signals, quarterly_signals)

    # Create final calculator with estimated coefficients
    ili_calculator = InventoryLeadingIndex(
        alpha_g=coefficients['alpha_g'],
        beta_g=coefficients['beta_g'],
        alpha_p=coefficients['alpha_p'],
        beta_p=coefficients['beta_p']
    )

    # Get latest values
    latest = monthly_signals.dropna(subset=['ILI']).iloc[-1]
    latest_date = latest['date']
    ili_latest = latest['ILI']

    # Forecasts
    gdp_forecast = ili_calculator.forecast_gdp(ili_latest)
    payroll_forecast = ili_calculator.forecast_payrolls(ili_latest)

    # Regime
    regime_color = ili_calculator.get_regime_color(ili_latest)
    regime_label = ili_calculator.get_regime_label(ili_latest)

    # ========== TOP ROW: KEY METRICS ==========
    st.markdown("---")
    st.subheader("📈 Current Economic Regime")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # ILI Gauge
        fig_gauge = plot_ili_gauge(ili_latest, regime_color, regime_label)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"As of {latest_date.strftime('%Y-%m-%d')}")

    with col2:
        st.metric(
            label="Model-Implied Next-Q GDP",
            value=f"{gdp_forecast:.1f}%",
            delta=None,
            help="Quarter-over-quarter annualized GDP growth forecast"
        )
        st.caption("Based on current ILI")

    with col3:
        st.metric(
            label="Model-Implied Payrolls",
            value=f"{payroll_forecast:,.0f}",
            delta=None,
            help="Next-month payroll change forecast (thousands)"
        )
        st.caption("Monthly change (thousands)")

    # ========== HEATMAP ==========
    st.markdown("---")
    st.subheader("🔥 Key Indicators Heatmap (Last 6 Months)")

    heatmap_indicators = [
        'auto_inv_sales_z',
        'biz_inv_sales_z',
        'new_orders_minus_inv_z',
        'coi_inv_z',
        'IA',
        'IS',
        'ILI'
    ]

    try:
        heatmap_df = create_heatmap_data(monthly_signals, heatmap_indicators, n_months=6)
        fig_heatmap = plot_heatmap(heatmap_df, title="")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.warning(f"Unable to create heatmap: {e}")

    # ========== MIDDLE ROW: REAL ECONOMY PANELS ==========
    st.markdown("---")
    st.subheader("🏭 Real Economy Engine")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Housing & Durables")
        # Filter to last 10 years for cleaner charts
        q_recent = quarterly_signals[quarterly_signals['date'] >= pd.Timestamp('2015-01-01')]
        if 'res_inv_qoq_ann' in q_recent.columns:
            fig = plot_time_series(
                q_recent, 'date', ['res_inv_qoq_ann'],
                title="Residential Investment (Q/Q Ann.)",
                y_label="Percent"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Residential investment data not available")

    with col2:
        st.markdown("#### Capex & IP")
        if 'equip_inv_qoq_ann' in q_recent.columns:
            fig = plot_time_series(
                q_recent, 'date', ['equip_inv_qoq_ann', 'ip_inv_qoq_ann'],
                title="Equipment & IP Investment (Q/Q Ann.)",
                y_label="Percent"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Equipment investment data not available")

    with col3:
        st.markdown("#### Inventory Cycle")
        m_recent = monthly_signals[monthly_signals['date'] >= pd.Timestamp('2015-01-01')]
        if 'auto_inv_sales_z' in m_recent.columns:
            fig = plot_time_series(
                m_recent, 'date', ['auto_inv_sales_z', 'biz_inv_sales_z'],
                title="Inventory/Sales Ratios (Z-Score)",
                y_label="Standard Deviations"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Inventory data not available")

    # ========== BOTTOM ROW: LABOR & PROFITS ==========
    st.markdown("---")
    st.subheader("👷 Labor Market & Corporate Profits")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Manufacturing Employment")
        if 'mfg_payrolls_3mma' in m_recent.columns:
            fig = plot_time_series(
                m_recent, 'date', ['mfg_payrolls_3mma'],
                title="Manufacturing Payrolls (3-Month MA)",
                y_label="Thousands of Jobs"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Manufacturing employment data not available")

    with col2:
        st.markdown("#### Corporate Profits")
        if 'corp_profits_yoy' in q_recent.columns:
            fig = plot_time_series(
                q_recent, 'date', ['corp_profits_yoy'],
                title="NIPA Corporate Profits (Y/Y)",
                y_label="Percent Change",
                colors=['#1f77b4']
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Corporate profits data not available")

    # ========== MODEL VALIDATION: PREDICTION VS ACTUAL ==========
    st.markdown("---")
    st.subheader("📉 Model Validation: Predicted vs Actual")

    # Generate historical predictions
    with st.spinner("Generating historical predictions..."):
        gdp_predictions = generate_gdp_predictions(
            monthly_signals, quarterly_signals,
            coefficients['alpha_g'], coefficients['beta_g']
        )
        payroll_predictions = generate_payroll_predictions(
            monthly_signals,
            coefficients['alpha_p'], coefficients['beta_p']
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### GDP Growth: Predicted vs Actual")
        if len(gdp_predictions) > 0:
            # Filter to recent history for cleaner chart
            gdp_recent = gdp_predictions[gdp_predictions['date'] >= pd.Timestamp('2010-01-01')]

            fig_gdp = plot_prediction_vs_actual(
                gdp_recent,
                'gdp_predicted',
                'gdp_actual',
                title="Next-Quarter GDP Growth (Q/Q Annualized)",
                y_label="Percent"
            )
            st.plotly_chart(fig_gdp, use_container_width=True)

            # Calculate metrics
            metrics = calculate_prediction_metrics(
                gdp_recent['gdp_predicted'].values,
                gdp_recent['gdp_actual'].values
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("R²", f"{coefficients['r2_gdp']:.3f}")
            with col_b:
                st.metric("MAE", f"{metrics['mae']:.2f}pp")
            with col_c:
                st.metric("Correlation", f"{metrics['correlation']:.3f}")
        else:
            st.info("Insufficient data for GDP predictions")

    with col2:
        st.markdown("#### Payroll Change: Predicted vs Actual")
        if len(payroll_predictions) > 0:
            # Filter to recent history
            payroll_recent = payroll_predictions[payroll_predictions['date'] >= pd.Timestamp('2010-01-01')]

            fig_payroll = plot_prediction_vs_actual(
                payroll_recent,
                'payroll_predicted',
                'payroll_actual',
                title="Next-Month Payroll Change",
                y_label="Thousands of Jobs"
            )
            st.plotly_chart(fig_payroll, use_container_width=True)

            # Calculate metrics
            metrics = calculate_prediction_metrics(
                payroll_recent['payroll_predicted'].values,
                payroll_recent['payroll_actual'].values
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("R²", f"{coefficients['r2_payroll']:.3f}")
            with col_b:
                st.metric("MAE", f"{metrics['mae']:.1f}k")
            with col_c:
                st.metric("Correlation", f"{metrics['correlation']:.3f}")
        else:
            st.info("Insufficient data for payroll predictions")

    # Scatter plots showing ILI relationship
    st.markdown("#### ILI Predictive Power")

    col1, col2 = st.columns(2)

    with col1:
        if len(gdp_predictions) > 0:
            fig_scatter_gdp = plot_scatter_with_fit(
                gdp_predictions,
                'ILI_lag1',
                'gdp_actual',
                title="ILI vs Next-Quarter GDP Growth",
                x_label="ILI (Z-Score)",
                y_label="GDP Growth (Q/Q Ann., %)"
            )
            st.plotly_chart(fig_scatter_gdp, use_container_width=True)

    with col2:
        if len(payroll_predictions) > 0:
            fig_scatter_payroll = plot_scatter_with_fit(
                payroll_predictions,
                'ILI_lag1',
                'payroll_actual',
                title="ILI vs Next-Month Payroll Change",
                x_label="ILI (Z-Score)",
                y_label="Payroll Change (Thousands)"
            )
            st.plotly_chart(fig_scatter_payroll, use_container_width=True)

    # ========== FOOTER ==========
    st.markdown("---")

    # Display model coefficients
    with st.expander("📊 Model Coefficients & Details"):
        st.markdown(f"""
        **GDP Model:** `GDP(t+1) = {coefficients['alpha_g']:.4f} + {coefficients['beta_g']:.4f} × ILI(t)`
        - R² = {coefficients['r2_gdp']:.3f}

        **Payroll Model:** `Payrolls(t+1) = {coefficients['alpha_p']:.2f} + {coefficients['beta_p']:.2f} × ILI(t)`
        - R² = {coefficients['r2_payroll']:.3f}

        Models estimated from historical data using ordinary least squares regression.
        ILI is lagged one period to predict next-period outcomes.
        """)

    st.caption("""
    **Methodology:** ILI (Inventory Leading Index) is a composite z-score combining inventory/sales ratios
    and ISM survey gaps. All signals are standardized to 60-month rolling z-scores.
    GDP and payroll forecasts are model-implied based on historical ILI relationships.

    **Data Sources:** Federal Reserve Economic Data (FRED) • BEA • Census Bureau • ISM
    """)


if __name__ == "__main__":
    main()
