Yes — let’s make something you could actually deploy as a macro leading-indicator dashboard for GDP & labor, centered on inventories + investment.

I’ll do three things:

Define the dashboard layout (what’s on screen, how it’s grouped)

Specify the indicators and formulas (what you actually compute)

Give you a concrete Python/Streamlit skeleton you can paste into a repo and start wiring to data

You can then plug in BEA / Census / ISM feeds however you like.

1. Dashboard Layout – “Growth Regime / Inventory Cycle Monitor”

Think in three horizontal bands:

A. Top row – Big Picture (Heatmap & Composite Index)

Tile 1 – “Inventory Leading Index (ILI)”

A single composite z-score (−3 to +3) of your key inventory & survey variables.

Color:

Deep red ≤ −1.5 → “Recession risk / below-trend growth”

Amber between −0.5 and −1.5

Green ≥ +0.5 → “Above-trend growth”

Tile 2 – “Next-Quarter GDP Signal”

Model-implied GDP q/q annualized (e.g. 0.8%, 2.1%, −1.2%)

Also show last actual GDP print.

Tile 3 – “Next-Quarter Payrolls Signal”

Model-implied monthly payroll growth (e.g. +50k, +150k, −75k).

Small heatmap with 6–10 key indicators (rows) × last 6 months (columns):

Auto Inv/Sales z

Non-auto Business Inv/Sales z

Residential Investment growth

Equipment Investment growth

IP Investment growth

New Orders – Inventories (ISM)

ISM Customers’ Inventories (inverted)

NIPA Profits (y/y)

B. Middle row – The “Real Economy Engine” Panels

Three panels with mini-charts (last 10–15 years):

Housing & Durables

Real Residential Investment (q/q ann., 4-q MA)

Real Durables PCE (y/y)

New single-family starts (if you add external data later)

Capex & IP

Real Private Equipment Investment (q/q ann.)

Real IP Products Investment (software + R&D)

Exports of capital goods (y/y)

Inventory Cycle

Motor-vehicle inventory-to-sales ratio (z-score)

Total business inv-to-sales ratio (z-score)

Δ inventories contribution to GDP (ppts)

Each panel with a simple regime shading:
e.g., green backgrounds when indicator > +0.5σ, red when < −0.5σ.

C. Bottom row – Labor & Profits

Manufacturing & Auto Labor

Manufacturing payrolls m/m (3-mma)

Transportation equipment employment

Average weekly hours in manufacturing

Corporate Profits & Margins

NIPA corporate profits (domestic industries, y/y)

Profit share of GDP (%)

Model Diagnostics (optional)

Scatter plot: ILI vs future GDP (one quarter ahead)

R², slope, last observation highlighted.

2. Core Indicators & Computations

Assume you ingest monthly/quarterly data into a tidy pandas DataFrame(s).

2.1 Inventory Signals

Using BEA/Census:

Auto Inv/Sales Ratio

auto_inv_sales = motor_vehicle_inventories / motor_vehicle_sales
auto_inv_sales_z = z_score(auto_inv_sales, window=60 months)


Non-Auto Business Inv/Sales

biz_inv_sales = (total_business_inventories - motor_vehicle_inventories) /
                (total_business_sales - motor_vehicle_sales)
biz_inv_sales_z = z_score(biz_inv_sales, window=60)


Δ Inventories Contribution to GDP (quarterly)
Directly from BEA NIPA: “Change in private inventories” contribution in percentage points.

2.2 Investment Signals (Quarterly, BEA Real Series)

res_inv = Real Residential Fixed Investment (q/q annualized or y/y)

equip_inv = Real Private Equipment Investment (q/q ann.)

ip_inv = Real Intellectual Property Products Investment (q/q ann.)

cap_goods_exports = Real Exports of Capital Goods (y/y)

Convert to z-scores vs long-run mean or 10-yr rolling.

2.3 Survey Inventory Gaps (Monthly, ISM / NFIB)

ism_new_orders

ism_inventories

ism_customers_inventories (low = good; invert)

Define:

new_orders_minus_inv = ism_new_orders - ism_inventories
coi_inv = - ism_customers_inventories  # invert


Standardize to z-scores.

2.4 Composite Inventory Leading Index (ILI)

Define two sub-factors:

IA = mean( -Δ auto_inv_sales_z, -Δ biz_inv_sales_z )   # higher = stronger growth
IS = mean( new_orders_minus_inv_z, coi_inv_z )

ILI_raw = βA * IA + βS * IS
ILI = z_score(ILI_raw, full_history)


For a first pass, set βA = βS = 0.5. Later you can estimate β’s via regression on future GDP.

2.5 Linking to GDP & Labor

Fit simple models (offline, in a notebook):

GDP_{t+1}      = α_g + β_g * ILI_t + ε_t
Payrolls_{t+1} = α_p + β_p * ILI_t + ε_t


Store α_g, β_g, α_p, β_p and use them in the dashboard to compute “model-implied” next-quarter GDP / payroll growth:

gdp_next = alpha_g + beta_g * ILI_latest
payroll_next = alpha_p + beta_p * ILI_latest

3. Streamlit Dashboard Skeleton (Python)

Here’s a minimal, ready-to-adapt Streamlit app. You’ll just need to replace the data-loading parts with your BEA/Census/ISM pipelines.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Helper functions ----------

def zscore(x, window=None):
    if window is None:
        return (x - x.mean()) / x.std()
    roll = x.rolling(window)
    return (x - roll.mean()) / roll.std()

@st.cache_data
def load_data():
    # TODO: replace with real data loaders
    # Expect monthly df: date, auto_inv, auto_sales, biz_inv, biz_sales,
    #                    ism_new_orders, ism_inv, ism_cust_inv, ...
    # And quarterly df: date, real_gdp, res_inv, equip_inv, ip_inv, cap_exports, ...
    monthly = pd.read_parquet("monthly_macro.parquet")
    quarterly = pd.read_parquet("quarterly_macro.parquet")
    return monthly, quarterly

def build_inventory_signals(monthly):
    m = monthly.copy()
    m = m.sort_values("date")

    m["auto_inv_sales"] = m["auto_inv"] / m["auto_sales"]
    m["biz_inv_sales"] = (m["biz_inv"] - m["auto_inv"]) / (m["biz_sales"] - m["auto_sales"])

    m["auto_inv_sales_z"] = zscore(m["auto_inv_sales"], window=60)
    m["biz_inv_sales_z"] = zscore(m["biz_inv_sales"], window=60)

    m["auto_inv_sales_z_diff"] = m["auto_inv_sales_z"].diff()
    m["biz_inv_sales_z_diff"] = m["biz_inv_sales_z"].diff()

    m["new_orders_minus_inv"] = m["ism_new_orders"] - m["ism_inv"]
    m["coi_inv"] = - m["ism_cust_inv"]

    m["new_orders_minus_inv_z"] = zscore(m["new_orders_minus_inv"], window=60)
    m["coi_inv_z"] = zscore(m["coi_inv"], window=60)

    # Sub-factors
    m["IA"] = (-m["auto_inv_sales_z_diff"] - m["biz_inv_sales_z_diff"]) / 2
    m["IS"] = (m["new_orders_minus_inv_z"] + m["coi_inv_z"]) / 2

    # Composite
    m["ILI_raw"] = 0.5 * m["IA"] + 0.5 * m["IS"]
    m["ILI"] = zscore(m["ILI_raw"])  # full-history z-score

    return m

# ---------- Main app ----------

st.set_page_config(page_title="Inventory & Growth Dashboard", layout="wide")

monthly, quarterly = load_data()
monthly_sig = build_inventory_signals(monthly)

latest = monthly_sig.dropna().iloc[-1]
latest_date = latest["date"]
ILI_latest = latest["ILI"]

# TODO: estimated coefficients from offline regression
alpha_g, beta_g = 0.02, 0.01      # example: 2% baseline + 1pp per ILI sigma
alpha_p, beta_p = 100.0, 30.0     # example: 100k + 30k per ILI sigma

gdp_next = alpha_g + beta_g * ILI_latest
payroll_next = alpha_p + beta_p * ILI_latest

st.title("Inventory & Growth Regime Dashboard")

# ---------- Top row: key tiles ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Inventory Leading Index (ILI)")
    st.metric(label=f"As of {latest_date.date()}", value=f"{ILI_latest:.2f} σ")

with col2:
    st.subheader("Model-Implied Next-Q GDP (q/q ann.)")
    st.metric("Forecast", f"{gdp_next*100:.1f}%")

with col3:
    st.subheader("Model-Implied Next-M Payrolls")
    st.metric("Forecast", f"{payroll_next:,.0f} jobs")

st.markdown("---")

# ---------- Middle: Panels ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### Housing & Durables")
    fig = px.line(quarterly, x="date", y=["res_inv_qoq", "durables_pce_yoy"],
                  labels={"value": "Percent", "variable": "Series"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Capex & IP")
    fig = px.line(quarterly, x="date", y=["equip_inv_qoq", "ip_inv_qoq", "cap_exports_yoy"])
    st.plotly_chart(fig, use_container_width=True)

with c3:
    st.markdown("### Inventory Cycle")
    fig = px.line(monthly_sig, x="date",
                  y=["auto_inv_sales_z", "biz_inv_sales_z"],
                  labels={"value": "Z-score", "variable": "Inv/Sales Z"})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------- Bottom: Labor & Profits ----------
b1, b2 = st.columns(2)

with b1:
    st.markdown("### Manufacturing & Auto Labor")
    fig = px.line(quarterly, x="date",
                  y=["mfg_payrolls_3mma", "transp_equip_jobs_3mma"],
                  labels={"value": "Thousands", "variable": "Series"})
    st.plotly_chart(fig, use_container_width=True)

with b2:
    st.markdown("### NIPA Profits")
    fig = px.line(quarterly, x="date",
                  y=["nipa_profits_yoy"],
                  labels={"nipa_profits_yoy": "Profits (y/y %)"})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("----")
st.caption("All signals standardized. ILI is a composite inventory leading index built from auto & business inventories and survey inventory gaps.")


You’d then:

Hook monthly_macro.parquet and quarterly_macro.parquet to your internal data pipeline (BEA, Census, ISM, NFIB, etc.).

Estimate alpha_g, beta_g, alpha_p, beta_p from historical regressions.

Incrementally add more panels (e.g., credit spreads, financial conditions, your own growth regime color bands).

If you’d like, next step I can:

Specify the exact column list / schema those parquet files should have,

Sketch the regression notebook that estimates the ILI → GDP / payroll coefficients, or

Add a “regime classification” (Expansion / Slowdown / Recession Risk / Recovery) to the dashboard logic.:wq
