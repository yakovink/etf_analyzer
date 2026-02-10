from global_import import sdmx, pd, Any, Dict, List, Optional, Tuple
from clients.database_manager import DatabaseManager


class SDMXClient:

    def __init__(self):
        self.client = sdmx.Client('IMF_DATA')
        self.db = DatabaseManager()
    
    def load_foreign_holdings(self, start_year: int = 1995) -> pd.DataFrame:
        """
        Fetch foreign equity holdings by banks from IMF SDMX API.
        Data: ODCORP_L_F51KOE_ODCS (Listed shares held by Other Depository Corporations)
        Key structure: ISO3+ISO3+....ODCORP_L_F51KOE_ODCS.XDC+USD+EUR.Q
        
        Args:
            start_year: Start year for data fetch
            
        Returns:
            DataFrame with columns: ISO3, QuarterID, value (in USD)
        """
        # 1. Get all ISO3 codes from database
        countries_df = self.db.fetch_df("""
            SELECT DISTINCT ISO3 
            FROM countries 
            LEFT JOIN exchanges ON countries.ISO = exchanges.ISO
            WHERE ISO3 IS NOT NULL AND ISO3 != '' AND exchanges.Currency IS NOT NULL AND exchanges.Currency != ''
        """)
        iso3_list = countries_df['ISO3'].tolist()
        
        # 2. Build SDMX key: join all countries with +
        countries_key = '+'.join(iso3_list)
        indicator = 'ODCORP_L_F51KOE_ODCS'
        transformations = 'XDC+USD+EUR'
        frequency = 'Q'
        
        # Full key: ISO3+ISO3+....INDICATOR.TRANSFORMATION.FREQUENCY
        key = f'{countries_key}.{indicator}.{transformations}.{frequency}'
        
        print(f"Fetching data for {len(iso3_list)} countries...")
        print(f"Key structure: [countries].{indicator}.{transformations}.{frequency}")
        
        # 3. Fetch data from IMF SDMX API
        try:
            data_msg = self.client.data(
                'MFS_ODC',  # Dataflow
                key=key,
                params={'startPeriod': str(start_year)}
            )
            
            print(f"Data fetched successfully")
            
            # 4. Convert SDMX message to pandas DataFrame
            df = sdmx.to_pandas(data_msg).reset_index()
            

            print(f"Raw data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # 5. Extract ISO3, period, transformation, and value
            # The contains 'TIME_PERIOD', 'COUNTRY', 'INDICATOR', 'TYPE_OF_TRANSFORMATION','FREQUENCY', 'IFS_FLAG', 'SCALE', 'ACCESS_SHARING_LEVEL', 'SECURITY_CLASSIFICATION', 'value'
            df = df.rename(columns={'TIME_PERIOD': 'Period', 'COUNTRY': 'ISO3', 'TYPE_OF_TRANSFORMATION': 'Transformation'})

            
            # Ensure we have the required columns
            required_cols = ['ISO3', 'Period', 'value']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols}. Available: {df.columns.tolist()}")
                # Try to extract from index if multiindex
                if hasattr(df.index, 'names'):
                    df = df.reset_index()
            
            # 6. Get currency mapping and exchange rates (same as load_foreign)
            currency_map = self.db.fetch_df("""
                SELECT DISTINCT c.ISO3, e.Currency
                FROM countries c
                LEFT JOIN exchanges e ON c.ISO = e.ISO
                WHERE c.ISO3 IS NOT NULL AND c.ISO3 != ''
            """)
            currency_map = currency_map.set_index('ISO3')['Currency'].to_dict()
            df['Currency'] = df['ISO3'].map(currency_map)
            
            # 7. Parse period to year and quarter
            df['year'] = df['Period'].str[:4].astype(int)
            df['quarter'] = df['Period'].str[-1:].astype(int)
            
            # 8. Get QuarterID from database
            quarters_df = self.db.fetch_df("SELECT QuarterID, year, quarter FROM quarters")
            df = df.merge(quarters_df, on=['year', 'quarter'], how='left')
            
            # 9. Get exchange rates
            rates_df = self.db.fetch_df("SELECT currencyID, QuarterID, RateToUSD FROM currency_rates")
            df = df.merge(rates_df, left_on=['Currency', 'QuarterID'], right_on=['currencyID', 'QuarterID'], how='left')

            # 10. Calculate rate to USD column
            df['Value_USD'] = df['value'] / df['RateToUSD']

            
            
            # 11. Select final columns
            result = df[['ISO3', 'QuarterID', 'Value_USD']].fillna(0)  # Fill NaN with 0 for missing values
            result.rename(columns={'Value_USD': 'value'}, inplace=True)
            
            print(f"\nLoaded {len(result)} records from IMF SDMX API")
            print(f"Countries: {result['ISO3'].nunique()}")
            print(f"Quarter range: {result['QuarterID'].min()} - {result['QuarterID'].max()}")
            
            return result
            
        except Exception as e:
            print(f"Error fetching data from IMF SDMX API: {e}")
            print(f"Key used: {key[:100]}...")  # Show first 100 chars
            raise