# Leading Indicators Dashboard

A comprehensive macroeconomic dashboard for tracking leading indicators of GDP and labor market performance, centered on inventory cycles and investment signals.

## Features

- **Inventory Leading Index (ILI)**: Composite indicator combining inventory/sales ratios and ISM survey data
- **GDP & Payroll Forecasts**: Model-based predictions using current ILI values
- **Real-time Economic Regime**: Color-coded growth regime classification
- **Interactive Visualizations**: Time series charts, heatmaps, and gauge displays
- **Comprehensive Data**: Housing, capex, inventories, labor, and corporate profits

## Data Sources

All data is sourced from **FRED (Federal Reserve Economic Data)**, including:
- Bureau of Economic Analysis (BEA)
- U.S. Census Bureau
- Bureau of Labor Statistics (BLS)
- Institute for Supply Management (ISM)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get FRED API Key

1. Go to https://fred.stlouisfed.org/
2. Create a free account
3. Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html
4. Add your API key to `.env`:

```bash
FRED_API_KEY=your_api_key_here
```

### 3. Fetch Initial Data

Run the data fetcher to download economic data:

```bash
python data_fetcher.py
```

This will create `data/monthly_macro.parquet` and `data/quarterly_macro.parquet`.

### 4. Launch Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
leading/
├── dashboard.py          # Streamlit dashboard application
├── data_fetcher.py       # FRED API data fetching module
├── indicators.py         # Indicator calculation functions
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
├── data/                 # Cached data files (auto-generated)
│   ├── monthly_macro.parquet
│   └── quarterly_macro.parquet
└── README.md            # This file
```

## Key Indicators

### Inventory Signals
- **Auto Inventory/Sales Ratio**: Motor vehicle inventory accumulation
- **Business Inventory/Sales Ratio**: Non-auto business inventories
- **ISM New Orders - Inventories**: Manufacturing order-inventory gap
- **ISM Customer Inventories**: Inverted (lower = stronger demand)

### Investment Signals
- **Residential Investment**: Real residential fixed investment
- **Equipment Investment**: Real private equipment investment
- **IP Investment**: Real intellectual property products investment
- **Capital Goods Exports**: Real exports of capital goods

### Labor Indicators
- **Manufacturing Payrolls**: Manufacturing sector employment
- **Transportation Equipment Jobs**: Auto sector employment
- **Average Weekly Hours**: Manufacturing hours worked

### Profits
- **Corporate Profits**: NIPA corporate profits (domestic)
- **Profit Share of GDP**: Profits as percentage of GDP

## Methodology

### Inventory Leading Index (ILI)

The ILI is calculated as:

1. **Sub-factor IA (Inventory Accumulation)**:
   - Negative changes in auto and business inventory/sales z-scores
   - Lower inventories = stronger demand signal

2. **Sub-factor IS (Survey Inventory Signals)**:
   - ISM new orders minus inventories
   - Inverted ISM customer inventories

3. **Composite ILI**:
   ```
   ILI = 0.5 * IA + 0.5 * IS
   ```
   Standardized to z-score (mean=0, std=1)

### Forecasting Models

Simple linear models relate ILI to future outcomes:

```
GDP_{t+1} = α_g + β_g * ILI_t
Payrolls_{t+1} = α_p + β_p * ILI_t
```

**Note**: Current coefficients are placeholders. For production use, estimate coefficients from historical regressions.

### Growth Regimes

Based on ILI z-score:
- **ILI ≤ -1.5**: Recession Risk / Below-Trend Growth (Red)
- **-1.5 < ILI ≤ -0.5**: Slowdown (Orange)
- **-0.5 < ILI < 0.5**: Neutral (Gray)
- **ILI ≥ 0.5**: Above-Trend Growth (Green)

## Customization

### Adjusting Forecast Coefficients

To use your own estimated coefficients, edit `dashboard.py`:

```python
ili_calculator = InventoryLeadingIndex(
    alpha_g=0.02,   # Your GDP intercept
    beta_g=0.01,    # Your GDP slope
    alpha_p=100.0,  # Your payroll intercept
    beta_p=30.0     # Your payroll slope
)
```

### Adding New Indicators

1. Add FRED series ID to `data_fetcher.py`:
```python
SERIES_IDS = {
    'new_indicator': 'FRED_SERIES_ID',
    ...
}
```

2. Update fetch methods to include new series
3. Add calculation logic to `indicators.py`
4. Add visualization to `dashboard.py`

## Data Refresh

- **Automatic**: Dashboard caches data for 1 hour
- **Manual**: Click "Refresh Data" button in sidebar
- **Scheduled**: Set up a cron job to run `python data_fetcher.py` daily

## Troubleshooting

### "FRED API key not found"
- Ensure `.env` file exists and contains `FRED_API_KEY=your_key`
- Check that `python-dotenv` is installed

### "Error fetching [series_id]"
- Check FRED series ID is correct
- Verify API key is valid
- Check network connectivity

### Charts showing "data not available"
- Run `python data_fetcher.py` to fetch initial data
- Check that parquet files exist in `data/` directory
- Verify FRED series IDs are still valid

## Future Enhancements

- [ ] Estimate ILI coefficients from historical regression
- [ ] Add regime probability forecasts
- [ ] Include financial conditions index
- [ ] Add NFIB small business survey data
- [ ] Implement backtesting framework
- [ ] Add email/SMS alerts for regime changes
- [ ] Create PDF report export

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Federal Reserve Bank of St. Louis for FRED API
- Streamlit for dashboard framework
- Project specification from project.md
