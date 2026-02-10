
from global_import import pdr, pd, time, List, Optional, wb, pd, datetime
from clients.database_manager import DatabaseManager

class DatareaderClient:
    def __init__(self):
        self.db = DatabaseManager()

    def get_world_bank_indicators(self, indicator_codes: List[str], db: Optional[int] = 2) -> pd.DataFrame:
        """
        Fetches World Bank indicators using pandas-datareader.
        Default DB is 2 (World Development Indicators).
        """
        try:
            df: pd.DataFrame = wb.download(indicator=indicator_codes, country='all', start=1960, end=datetime.datetime.now().year)
            df = df.reset_index()


            country_meta = self.db.fetch_df("SELECT ISO, country_name FROM countries").set_index('country_name')['ISO']
            df['country'] = df['country'].map(country_meta)
        
            return df
        except Exception as e:
            print(f"DatareaderClient: Error fetching indicators {indicator_codes} from DB {db}: {e}")
            return pd.DataFrame()
        
    def get_fred_indicator(self, indicator_code: str, start: Optional[datetime.datetime] = None, end: Optional[datetime.datetime] = None) -> pd.DataFrame:
        """
        Fetches FRED indicator data using pandas-datareader.
        """
        try:
            df: pd.DataFrame = pdr.DataReader(indicator_code, 'fred', start=start, end=end)
            df = df.reset_index()
            return df
        except Exception as e:
            print(f"DatareaderClient: Error fetching FRED indicator {indicator_code}: {e}")
            return pd.DataFrame()
        

    def get_oecd_indicator(self, dataset_code, slice_keys, slice_levels, countries=None, start=None, end=None):
        """
        Fetches OECD indicator data using dimensional slicing.
        
        Args:
            dataset_code: OECD dataset code (e.g., 'ASA' for Annual Sector Accounts)
            slice_keys: List of dimension values (e.g., ['S14_S15', 'AF5', 'LE', 'A'])
            slice_levels: List of dimension names (e.g., ['Sector', 'Transaction', 'Measure', 'Frequency'])
            countries: List of ISO3 country codes (optional - if None, fetches all)
            start: Start datetime (default: 1960)
            end: End datetime (default: current year)
            
        Returns:
            DataFrame with columns: LOCATION, TIME, Value
        """
        try:
            
            if start is None:
                start = datetime.datetime(1960, 1, 1)  # Fetch as far back as possible
            
            if end is None:
                end = datetime.datetime.now()
            
            # Build OECD query string with slices
            # Format for pandas_datareader: DATASET.LOCATION.dim2.dim3.dim4
            # The first dimension after dataset is typically LOCATION
            
            slice_filter = ".".join(slice_keys)  # e.g., "S14_S15.AF5.LE.A"
            
            # Try simple format first: ASA.AUS+USA.S14_S15.AF5.LE.A
            # pandas_datareader will handle the query construction
            query_string = f"{dataset_code}..{slice_filter}"
            
            print(f"    Fetching OECD: {query_string} (from {start.year} to {end.year})")
            
            # OECD data source in pandas_datareader
            oecd_data = pdr.DataReader(
                name=query_string,
                data_source='oecd',
                start=start,
                end=end
            )
            
            if oecd_data is not None and not oecd_data.empty:
                # Reset index to get columns
                if isinstance(oecd_data.index, pd.MultiIndex):
                    oecd_data = oecd_data.reset_index()
                
                # Ensure we have LOCATION, TIME/Time, and Value columns
                if 'LOCATION' not in oecd_data.columns and 'Country' in oecd_data.columns:
                    oecd_data = oecd_data.rename(columns={'Country': 'LOCATION'})
                
                if 'TIME' not in oecd_data.columns and 'Time' in oecd_data.columns:
                    oecd_data = oecd_data.rename(columns={'Time': 'TIME'})
                
                # Find value column (usually 'Value' or numeric column)
                value_col = None
                if 'Value' in oecd_data.columns:
                    value_col = 'Value'
                elif 'value' in oecd_data.columns:
                    value_col = 'value'
                else:
                    import numpy as np
                    numeric_cols = oecd_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        value_col = numeric_cols[0]
                
                if value_col and value_col != 'Value':
                    oecd_data = oecd_data.rename(columns={value_col: 'Value'})
                
                # Optionally filter to requested countries if provided
                if countries is not None and len(countries) > 0 and 'LOCATION' in oecd_data.columns:
                    oecd_data = oecd_data[oecd_data['LOCATION'].isin(countries)]
                
                print(f"    ✓ Fetched {len(oecd_data)} records for {oecd_data['LOCATION'].nunique() if 'LOCATION' in oecd_data.columns else 'unknown'} countries")
                
                return oecd_data
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"    Error fetching OECD data ({dataset_code}): {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()



        

