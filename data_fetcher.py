"""
FRED API Data Fetcher for Leading Indicators Dashboard
Fetches economic data from FRED (Federal Reserve Economic Data)
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FREDDataFetcher:
    """Fetches economic data from FRED API"""

    # FRED Series IDs for all required indicators
    SERIES_IDS = {
        # Inventory Data (Monthly)
        'auto_inv': 'AUINSA',              # Auto inventories (NSA)
        'auto_sales': 'ALTSALES',          # Auto sales
        'total_biz_inv': 'BUSINV',         # Total business inventories
        'total_biz_sales': 'TOTBUSSMSA',   # Total business sales

        # ISM Survey Data (Monthly) - Using available FRED proxies
        'ism_pmi': 'NAPM',                 # ISM Manufacturing PMI
        'ism_new_orders': 'DGORDER',       # New Orders for Durable Goods (proxy)
        'ism_inventories': 'ISRATIO',      # Total Business Inventory/Sales Ratio
        'ism_customer_inv': 'RETAILIRSA',  # Retail Inventories (proxy)

        # GDP & Investment (Quarterly)
        'real_gdp': 'GDPC1',               # Real GDP
        'res_inv': 'PRFI',                 # Real Residential Fixed Investment
        'equip_inv': 'PNFIC1',             # Real Private Equipment Investment
        'ip_inv': 'Y001RC1Q027SBEA',       # Real IP Products Investment
        'cap_goods_exports': 'EXPGSC1',    # Real Exports of Capital Goods
        'inv_change_gdp': 'A014RE1Q156NBEA', # Change in private inventories contribution

        # Labor Data (Monthly)
        'mfg_payrolls': 'MANEMP',          # Manufacturing Employment
        'transp_equip_jobs': 'CES3133600001', # Transportation Equipment Employment
        'mfg_hours': 'AWHAEMAN',           # Avg Weekly Hours Manufacturing

        # Profits (Quarterly)
        'corp_profits': 'CP',              # Corporate Profits
        'profit_share_gdp': 'W273RC1Q156NBEA', # Corporate Profits as % of GDP

        # Durables (Monthly)
        'durables_pce': 'PCEDG',           # Real PCE Durable Goods
    }

    def __init__(self, api_key=None):
        """Initialize FRED API client"""
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
            if not api_key:
                raise ValueError(
                    "FRED API key not found. Please set FRED_API_KEY in .env file.\n"
                    "Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html"
                )

        self.fred = Fred(api_key=api_key)
        self.cache = {}

    def fetch_series(self, series_id, start_date=None):
        """
        Fetch a single time series from FRED

        Args:
            series_id: FRED series identifier
            start_date: Start date for data (default: 25 years ago)

        Returns:
            pandas Series with date index
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365*25)

        try:
            series = self.fred.get_series(series_id, observation_start=start_date)
            return series
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return pd.Series()

    def fetch_monthly_data(self, start_date=None):
        """
        Fetch all monthly economic indicators

        Returns:
            DataFrame with monthly data
        """
        print("Fetching monthly data from FRED...")

        monthly_series = {
            'auto_inv', 'auto_sales', 'total_biz_inv', 'total_biz_sales',
            'ism_pmi', 'ism_new_orders', 'ism_inventories', 'ism_customer_inv',
            'mfg_payrolls', 'transp_equip_jobs', 'mfg_hours', 'durables_pce'
        }

        data = {}
        for name in monthly_series:
            if name in self.SERIES_IDS:
                series_id = self.SERIES_IDS[name]
                print(f"  Fetching {name} ({series_id})...")
                data[name] = self.fetch_series(series_id, start_date)

        # Combine into DataFrame
        df = pd.DataFrame(data)
        df.index.name = 'date'
        df = df.reset_index()

        print(f"Monthly data fetched: {len(df)} observations")
        return df

    def fetch_quarterly_data(self, start_date=None):
        """
        Fetch all quarterly economic indicators

        Returns:
            DataFrame with quarterly data
        """
        print("Fetching quarterly data from FRED...")

        quarterly_series = {
            'real_gdp', 'res_inv', 'equip_inv', 'ip_inv',
            'cap_goods_exports', 'inv_change_gdp',
            'corp_profits', 'profit_share_gdp'
        }

        data = {}
        for name in quarterly_series:
            if name in self.SERIES_IDS:
                series_id = self.SERIES_IDS[name]
                print(f"  Fetching {name} ({series_id})...")
                data[name] = self.fetch_series(series_id, start_date)

        # Combine into DataFrame
        df = pd.DataFrame(data)
        df.index.name = 'date'
        df = df.reset_index()

        print(f"Quarterly data fetched: {len(df)} observations")
        return df

    def fetch_all_data(self, start_date=None):
        """
        Fetch both monthly and quarterly data

        Returns:
            Tuple of (monthly_df, quarterly_df)
        """
        monthly = self.fetch_monthly_data(start_date)
        quarterly = self.fetch_quarterly_data(start_date)
        return monthly, quarterly

    def save_data(self, monthly_df, quarterly_df, output_dir='data'):
        """Save data to parquet files"""
        os.makedirs(output_dir, exist_ok=True)

        monthly_path = os.path.join(output_dir, 'monthly_macro.parquet')
        quarterly_path = os.path.join(output_dir, 'quarterly_macro.parquet')

        monthly_df.to_parquet(monthly_path, index=False)
        quarterly_df.to_parquet(quarterly_path, index=False)

        print(f"\nData saved to:")
        print(f"  {monthly_path}")
        print(f"  {quarterly_path}")


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = FREDDataFetcher()
    monthly, quarterly = fetcher.fetch_all_data()

    print("\n=== Monthly Data Preview ===")
    print(monthly.tail())
    print(f"\nShape: {monthly.shape}")

    print("\n=== Quarterly Data Preview ===")
    print(quarterly.tail())
    print(f"\nShape: {quarterly.shape}")

    # Save to files
    fetcher.save_data(monthly, quarterly)
