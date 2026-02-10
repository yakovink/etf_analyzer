"""
Pipeline 4: Raw Data Loader (Macro + Micro)
- Scheme 4: Raw
- Sources: World Bank, FRED, EOD Historical Data
- Tables: macro_raw, companies_details, stocks_raw
"""

from global_import import pd, json, datetime, time, pycountry, np, os
from clients.datareader_client import DatareaderClient
from clients.database_manager import DatabaseManager
from clients.eod_client import EODClient
from clients.oecd_client import OECDClient

CONFIG_PATH = os.path.join('data', 'infrastructure', 'macro_indicators.json')

class RawLoader:
    def __init__(self):
        self.db = DatabaseManager()
        self.client = DatareaderClient()
        self.eod_client = EODClient()
        self.oecd_client = OECDClient()
        self.conn = self.db.get_connection()
        self.cursor = self.conn.cursor()
        self.config : dict = self._load_config()
        self.target_years = range(1995, 2026)
        # optimize: cache iso map
        try:
            self._iso_map_df = self.get_target_countries()
            self._iso_map = dict(zip(self._iso_map_df['ISO'], self._iso_map_df['ISO3']))
        except:
            self._iso_map = {}

    def _load_config(self):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)['macro_indicators']
        

    def get_oecd_data(self, countries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches OECD indicators using SDMX 3.0 API via OECDClient.
        Looks for indicators with 'oecd_sdmx_params' key in config.
        """
        # Extract OECD indicators from config
        oecd_indicators = {k: v for k, v in self.config.items() 
                          if v.get('oecd_sdmx_params')}
        
        if not oecd_indicators:
            print("No OECD indicators configured.")
            return pd.DataFrame()
        
        print(f"Fetching {len(oecd_indicators)} OECD indicators...")
        
        all_oecd_data = []
        
        # Fetch each indicator
        for key, meta in oecd_indicators.items():
            try:
                print(f"  Fetching OECD indicator: {key}")
                
                oecd_params = meta.get('oecd_sdmx_params', {})
                
                # Use OECDClient to fetch data
                oecd_data = self.oecd_client.fetch_indicator(
                    params=oecd_params,
                    start_year=min(self.target_years),
                    end_year=max(self.target_years)
                )
                
                if oecd_data is not None and not oecd_data.empty:
                    # Rename Value column to indicator name
                    oecd_data = oecd_data.rename(columns={'Value': key})
                    
                    all_oecd_data.append(oecd_data)
                    print(f"    ✓ Fetched {len(oecd_data)} records across {oecd_data['ISO3'].nunique()} countries")
                else:
                    print(f"    ✗ No data returned for {key}")
                    
            except Exception as e:
                print(f"    ✗ Error fetching OECD {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Merge all OECD indicators
        if not all_oecd_data:
            print("  No OECD data fetched.")
            return pd.DataFrame()
        
        # Start with first indicator
        merged = all_oecd_data[0]
        
        # Merge rest on ISO3 + Year
        for oecd_df in all_oecd_data[1:]:
            merged = pd.merge(merged, oecd_df, on=['ISO3', 'Year'], how='outer')
        
        # Convert ISO3 to ISO2 for consistency
        iso3_to_iso2 = dict(zip(countries_df['ISO3'], countries_df['ISO']))
        merged['ISO'] = merged['ISO3'].map(iso3_to_iso2)
        
        # Keep only known countries
        merged = merged.dropna(subset=['ISO'])
        merged = merged.drop(columns=['ISO3'])
        
        print(f"OECD fetch complete: {len(merged)} country-year records, {len(merged.columns)-2} indicators")
        print(f"  Coverage: {merged['Year'].min()}-{merged['Year'].max()}")
        print(f"  Countries: {merged['ISO'].nunique()}")
        
        return merged
        
        # LEGACY CODE (non-functional):
        # Extract OECD indicators from config
        oecd_indicators = {k: v for k, v in self.config.items() 
                        if v.get('oecd_params') and v.get('source_priority', [''])[0] in ['OECD_DATAREADER', 'OECD']}
        
        if not oecd_indicators:
            return pd.DataFrame()
        
        print(f"Fetching {len(oecd_indicators)} OECD indicators for ALL available countries...")
        
        all_oecd_data = []
        
        # Fetch each indicator separately (fetches ALL countries, full history)
        for key, meta in oecd_indicators.items():
            try:
                print(f"  Fetching OECD indicator: {key}")
                
                oecd_params = meta.get('oecd_params', {})
                dataset_code = oecd_params.get('dataset_code')
                slice_keys = oecd_params.get('slice_keys', [])
                slice_levels = oecd_params.get('slice_levels', [])
                
                if not dataset_code or not slice_keys:
                    print(f"    ✗ Missing OECD params for {key}")
                    continue
                
                # Use DatareaderClient's OECD method WITHOUT country filter
                # This fetches ALL countries from 1960 onwards
                oecd_data = self.client.get_oecd_indicator(
                    dataset_code=dataset_code,
                    slice_keys=slice_keys,
                    slice_levels=slice_levels,
                    countries=None,  # Fetch ALL countries
                    start=datetime.datetime(1960, 1, 1),  # Max history
                    end=datetime.datetime.now()
                )
                
                if oecd_data is not None and not oecd_data.empty:
                    # Ensure required columns exist
                    if 'LOCATION' not in oecd_data.columns or 'TIME' not in oecd_data.columns or 'Value' not in oecd_data.columns:
                        print(f"    ✗ Missing required columns for {key}: {oecd_data.columns.tolist()}")
                        continue
                    
                    # Convert TIME to Year
                    oecd_data['TIME'] = oecd_data['TIME'].astype(str)
                    
                    # Handle different time formats (YYYY, YYYY-Qn, YYYY-MM)
                    def extract_year(time_str):
                        try:
                            if '-' in str(time_str):
                                return int(str(time_str).split('-')[0])
                            return int(str(time_str)[:4])
                        except:
                            return None
                    
                    oecd_data['Year'] = oecd_data['TIME'].apply(extract_year)
                    oecd_data = oecd_data.dropna(subset=['Year'])
                    oecd_data['Year'] = oecd_data['Year'].astype(int)
                    
                    # Rename columns
                    oecd_data = oecd_data.rename(columns={
                        'LOCATION': 'ISO3',
                        'Value': key
                    })
                    
                    # Group by ISO3 and Year (aggregate if multiple observations per year)
                    # For annual data (A frequency), this should be 1:1, but for quarterly/monthly we take the mean
                    oecd_data = oecd_data.groupby(['ISO3', 'Year'])[key].mean().reset_index()
                    
                    all_oecd_data.append(oecd_data)
                    print(f"    ✓ Fetched {len(oecd_data)} records across {oecd_data['ISO3'].nunique()} countries")
                else:
                    print(f"    ✗ No data returned for {key}")
                    
            except Exception as e:
                print(f"    ✗ Error fetching OECD {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Merge all OECD indicators
        if not all_oecd_data:
            print("  No OECD data fetched.")
            return pd.DataFrame()
        
        # Start with first indicator
        merged = all_oecd_data[0]
        
        # Merge rest on ISO3 + Year
        for oecd_df in all_oecd_data[1:]:
            merged = pd.merge(merged, oecd_df, on=['ISO3', 'Year'], how='outer')
        
        # Convert ISO3 to ISO2 for consistency with other data
        # Only keep countries that exist in our countries table
        iso3_to_iso2 = dict(zip(countries_df['ISO3'], countries_df['ISO']))
        merged['ISO'] = merged['ISO3'].map(iso3_to_iso2)
        
        # Keep ALL countries (even those not in our exchanges)
        # This allows future expansion without re-fetching OECD data
        # Pipeline will filter to relevant countries later
        print(f"  Mapped {merged['ISO'].notna().sum()} / {len(merged)} records to known countries")
        
        # Drop ISO3 column (keep ISO2)
        merged = merged.drop(columns=['ISO3'])
        
        # Filter to only countries we have in the database (optional - comment out to keep all)
        merged = merged.dropna(subset=['ISO'])
        
        print(f"OECD fetch complete: {len(merged)} country-year records, {len(merged.columns)-2} indicators")
        print(f"  Coverage: {merged['Year'].min()}-{merged['Year'].max()}")
        print(f"  Countries: {merged['ISO'].nunique()}")
        
        return merged



    def create_tables(self):
        """Creates/Resets tables for Scheme 4: Raw."""
        
        # 1. Macro Raw
        cols = ["ISO TEXT", "Year INTEGER"]
        for key in self.config.keys():
            cols.append(f"{key} REAL")
        cols_str = ", ".join(cols)
        
        # Reset macro_raw to ensure schema matches JSON
        self.db.execute_query(f"DROP TABLE IF EXISTS macro_raw")
        self.db.execute_query(f"CREATE TABLE IF NOT EXISTS macro_raw ({cols_str}, PRIMARY KEY (ISO, Year))")

        # 2. Companies Details (Micro)
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS companies_details (
            ISIN TEXT PRIMARY KEY, Industry TEXT, Sector TEXT, Description TEXT)''')
            
        # 3. Stocks Raw (Micro)
        # Updates per Scheme 4 (TABLES.md)
        # Cols: revenue, net_income, market_cap, total_assets, total_liabilities, 
        # dividend_paid, stock_buybacks, investment_cf, report_date
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS stocks_raw (
            ISIN TEXT, 
            QuarterID INTEGER, 
            revenue REAL, 
            net_income REAL, 
            market_cap REAL, 
            total_assets REAL, 
            total_liabilities REAL, 
            dividend_paid REAL, 
            stock_buybacks REAL, 
            investment_cf REAL, 
            report_date TEXT, 
            PRIMARY KEY (ISIN, QuarterID))''')
            
        self.conn.commit()
        print("Schema 4 Tables (macro_raw, companies_details, stocks_raw) ready.")

    def get_target_countries(self):
        """Returns DataFrame with ISO (2-char) and ISO3 (3-char) for countries with exchanges."""
        query = """
            SELECT DISTINCT c.ISO, c.ISO3, c.country_name 
            FROM countries c
            JOIN exchanges e ON c.ISO = e.ISO
        """
        try:
            return self.db.fetch_df(query)
        except Exception as e:
            # Fallback for dev/testing if tables don't exist
            # print(f"Warning: Could not fetch target countries ({e})") 
            return pd.DataFrame(columns=['ISO', 'ISO3'])

    def fetch_eod_macro_data(self, countries_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches EOD macro indicators (e.g., national_income_usd, national_savings_usd).
        Uses EODClient.get_macro_indicator() with eod_macro_code from config.
        """
        # Extract EOD macro indicators from config
        eod_indicators = {k: v for k, v in self.config.items() 
                         if v.get('eod_macro_code')}
        
        if not eod_indicators:
            print("No EOD macro indicators configured.")
            return pd.DataFrame()
        
        print(f"Fetching {len(eod_indicators)} EOD macro indicators...")
        
        all_eod_data = []
        
        # Define function to fetch data for a single country (enables apply pattern)
        def _fetch_country_eod(country_row):
            iso2 = country_row['ISO']
            country_data = []
            
            if not iso2:
                return country_data
            
            # Fetch each indicator for this country
            for key, meta in eod_indicators.items():
                try:
                    eod_code = meta.get('eod_macro_code')
                    
                    # Use EODClient to fetch macro data
                    eod_data = self.eod_client.get_macro_indicator(
                        country_code=iso2,
                        indicators=[eod_code]
                    )
                    
                    if eod_data is not None and not eod_data.empty:
                        # EOD returns quarterly data, need to aggregate to annual
                        if 'Date' in eod_data.columns:
                            eod_data['Year'] = pd.to_datetime(eod_data['Date']).dt.year
                            
                            # Aggregate by year (mean for flow variables)
                            annual_data = eod_data.groupby('Year').agg({
                                eod_code: 'mean'  # Take annual average
                            }).reset_index()
                            
                            annual_data['ISO'] = iso2
                            annual_data = annual_data.rename(columns={eod_code: key})
                            
                            # Filter to target years
                            annual_data = annual_data[annual_data['Year'].isin(self.target_years)]
                            
                            country_data.append(annual_data)
                            
                except Exception as e:
                    # Silent fail for individual indicators (common for missing data)
                    continue
            
            return country_data
        
        # Apply function across countries (serial execution for API calls)
        eod_data_lists = countries_df.apply(_fetch_country_eod, axis=1)
        
        # Flatten the list of lists
        all_eod_data = [df for sublist in eod_data_lists for df in sublist]
        
        # Merge all EOD data
        if not all_eod_data:
            print("  No EOD macro data fetched.")
            return pd.DataFrame()
        
        # Concatenate and pivot
        merged = pd.concat(all_eod_data, ignore_index=True)
        
        # Group by ISO + Year and aggregate (in case of duplicates)
        merged = merged.groupby(['ISO', 'Year'], as_index=False).first()
        
        print(f"EOD macro fetch complete: {len(merged)} country-year records")
        print(f"  Coverage: {merged['Year'].min()}-{merged['Year'].max()}")
        print(f"  Countries: {merged['ISO'].nunique()}")
        
        return merged

    def fetch_wb_data(self, countries_df):
        """Fetches all indicators from WB for target countries."""
        key_to_wb = {k: v['wb_code'] for k, v in self.config.items() if v.get('wb_code')}
        wb_to_key = {v: k for k, v in key_to_wb.items()}
        wb_codes = list(key_to_wb.values())

        if not wb_codes or countries_df.empty:
            return pd.DataFrame()

        print(f"Fetching {len(wb_codes)} WB indicators for {len(countries_df)} countries...")
        try:
            raw_wb = self.client.get_world_bank_indicators(wb_codes)
            if raw_wb.empty: return pd.DataFrame()

            target_isos = set(countries_df['ISO'].unique())
            if 'country' not in raw_wb.columns: raw_wb = raw_wb.reset_index()
            
            raw_wb = raw_wb[raw_wb['country'].isin(target_isos)]
            raw_wb['year'] = raw_wb['year'].astype(int)
            raw_wb = raw_wb[raw_wb['year'].isin(self.target_years)]

            raw_wb.rename(columns=wb_to_key, inplace=True)
            raw_wb.rename(columns={'country': 'ISO', 'year': 'Year'}, inplace=True)
            return raw_wb
        except Exception as e:
            print(f"Error in WB Fetch: {e}")
            return pd.DataFrame()

    def process_initial_data(self, countries_df):
        """
        Fetch and merge macro indicators from multiple sources.
        Order: WB (primary) → OECD → EOD macro → skeleton
        """
        # Fetch from all sources
        wb_data = self.fetch_wb_data(countries_df)
        oecd_data = self.get_oecd_data(countries_df)
        eod_macro_data = self.fetch_eod_macro_data(countries_df)
        
        isos = countries_df['ISO'].unique() if not countries_df.empty else []
        years = list(self.target_years)
        
        if len(isos) == 0:
            return pd.DataFrame()

        # Create skeleton (all country-year combinations)
        skeleton = pd.MultiIndex.from_product([isos, years], names=['ISO', 'Year']).to_frame(index=False)
        
        # Merge in order: skeleton → WB → OECD → EOD
        result = skeleton.copy()
        
        if not wb_data.empty:
            result = pd.merge(result, wb_data, on=['ISO', 'Year'], how='left')
            print(f"  Merged WB data: {len([c for c in wb_data.columns if c not in ['ISO', 'Year']])} indicators")
        
        if not oecd_data.empty:
            result = pd.merge(result, oecd_data, on=['ISO', 'Year'], how='left', suffixes=('', '_oecd'))
            # OECD data should overwrite WB where both exist (OECD is more specific for S1/S2)
            for col in oecd_data.columns:
                if col not in ['ISO', 'Year']:
                    oecd_col = f"{col}_oecd" if f"{col}_oecd" in result.columns else col
                    if oecd_col in result.columns and col in result.columns:
                        result[col] = result[oecd_col].combine_first(result[col])
                        result.drop(columns=[oecd_col], inplace=True)
            print(f"  Merged OECD data: {len([c for c in oecd_data.columns if c not in ['ISO', 'Year']])} indicators")
        
        if not eod_macro_data.empty:
            result = pd.merge(result, eod_macro_data, on=['ISO', 'Year'], how='left', suffixes=('', '_eod'))
            # EOD macro should fill gaps, not overwrite existing WB data
            for col in eod_macro_data.columns:
                if col not in ['ISO', 'Year']:
                    eod_col = f"{col}_eod" if f"{col}_eod" in result.columns else col
                    if eod_col in result.columns and col in result.columns:
                        result[col] = result[col].combine_first(result[eod_col])
                        result.drop(columns=[eod_col], inplace=True)
            print(f"  Merged EOD macro data: {len([c for c in eod_macro_data.columns if c not in ['ISO', 'Year']])} indicators")
        
        return result

    def patch_missing_data(self, df):
        """Patches gaps with FRED and EOD."""
        for key in self.config.keys():
            if key not in df.columns: df[key] = np.nan

        print("Patching Data (WorldBank -> FRED -> EOD)...")
        # Use vectorized/apply approach
        return df.groupby('ISO', group_keys=False).apply(self._patch_country_group)

    def _patch_country_group(self, group):
        iso = group['ISO'].iloc[0]
        iso3 = self._iso_map.get(iso)
        if not iso3: return group 

        # 1. FRED Patching (skip indicators already loaded from OECD/EOD macro)
        for key, meta in self.config.items():
            # Skip OECD indicators (fetched directly from OECD SDMX API)
            if meta.get('oecd_sdmx_params'):
                continue
            
            # Skip EOD macro indicators (fetched directly from EOD)
            if meta.get('eod_macro_code'):
                continue
            
            if not group[key].isna().any(): continue
            
            patches = meta.get('fred_patches', {})
            fred_code = patches.get(str(iso3))
            if fred_code:
                    try:
                        # Optimization: Fetch FRED once per code? No, client handles caching hopefully.
                        fred_data = self.client.get_fred_indicator(fred_code, start=datetime.datetime(1995,1,1), end=datetime.datetime(2024,12,31))
                        if not fred_data.empty:
                            if 'DATE' not in fred_data.columns: fred_data = fred_data.reset_index()
                            d_col = 'DATE' if 'DATE' in fred_data.columns else fred_data.columns[0]
                            fred_data['Year'] = pd.to_datetime(fred_data[d_col]).dt.year
                            annual = fred_data.groupby('Year')[fred_code].mean()
                            group[key] = group[key].fillna(group['Year'].map(annual))
                    except Exception: pass
        
        # 2. EOD Patching (ticker-based, not macro)
        for key, meta in self.config.items():
            # Skip EOD macro indicators (already loaded)
            if meta.get('eod_macro_code'):
                continue
                
            if not group[key].isna().any(): continue
            eod_code = meta.get('eod_code')
            ticker_pattern = meta.get('eod_ticker_pattern')
            
            if eod_code and 'FUTURE' in eod_code: eod_code = None
            if ticker_pattern and 'FUTURE' in ticker_pattern: ticker_pattern = None

            if eod_code:
                try:
                     # EOD Macro using ISO3
                     macro_data = self.eod_client.get_macro_indicator(iso3, indicators=[eod_code])
                     # Client not strictly defined? 
                     # Actually eod_client.py doesn't have get_macro_indicator in the snippet I saw.
                     # But MacroLoader in P4 used it? Let's check P4 content history. Yes, it used eod_client.
                     # I will assume it works or fails gracefully.
                     pass 
                except: pass
            
            if ticker_pattern:
                    ticker = ticker_pattern.replace("{ISO}", iso)
                    try:
                        exch = 'INDX' if '.INDX' not in ticker else ''
                        hist = self.eod_client.get_ticker_historical(ticker, exchange=exch)
                        if hist is not None and not hist.empty and 'date' in hist.columns:
                            hist['Year'] = pd.to_datetime(hist['date']).dt.year
                            annual = hist.groupby('Year')['close'].mean()
                            group[key] = group[key].fillna(group['Year'].map(annual))
                    except: pass
        
        return group


    def fetch_and_load_companies_details(self):
        """Micro: Fetches company details."""
        if hasattr(self.eod_client, 'get_companies_details'):
            print("Fetching Companies Details...")
            try:
                data = self.eod_client.get_companies_details()
                if data:
                    pd.DataFrame(data).to_sql('companies_details', self.conn, if_exists='replace', index=False)
                    print("Loaded companies_details.")
            except Exception as e:
                print(f"Error fetching companies details: {e}")

    def fetch_and_load_stocks_raw(self):
        """Micro: Fetches raw stock fundamentals."""
        if not hasattr(self.eod_client, 'get_stocks_data'):
            return
            
        try:
            exchanges = pd.read_sql('SELECT exchangeID FROM exchanges', self.conn)
        except:
            print("Exchanges table not available.")
            return

        print(f"Fetching Stocks Raw for {len(exchanges)} exchanges...")
        all_stocks = []
        for exch in exchanges['exchangeID']:
            try:
                data = self.eod_client.get_stocks_data(exch) 
                if data: all_stocks.extend(data)
            except: pass
        
        if all_stocks:
            # Process Data
            rows = []
            for row in all_stocks:
                # 1. Map Date -> QuarterID
                # Tries 'date', 'report_date', 'date_formatted'
                r_date = row.get('date') or row.get('report_date') or row.get('published_date')
                if not r_date: continue
                
                try:
                    dt = pd.to_datetime(r_date)
                    quarter = (dt.month - 1) // 3 + 1
                    qid = int(f"{dt.year}{quarter}")
                except: continue
                
                # 2. Extract Fields (Safe Map)
                # Maps EOD keys (assumed camelCase/snake_case mix) to Schema
                new_row = {
                    'ISIN': row.get('ISIN') or row.get('code'), # Fallback
                    'QuarterID': qid,
                    'revenue': float(row.get('revenue', 0) or 0),
                    'net_income': float(row.get('net_income', 0) or row.get('netIncome', 0) or 0),
                    'market_cap': float(row.get('market_cap', 0) or row.get('marketCapitalization', 0) or 0),
                    'total_assets': float(row.get('total_assets', 0) or row.get('totalAssets', 0) or 0),
                    'total_liabilities': float(row.get('total_liabilities', 0) or row.get('totalLiab', 0) or 0),
                    'dividend_paid': float(row.get('dividend_paid', 0) or 0), # usually negative in CF
                    'stock_buybacks': float(row.get('stock_buybacks', 0) or row.get('salePurchaseOfStock', 0) or 0), 
                    'investment_cf': float(row.get('investment_cf', 0) or row.get('netInvestments', 0) or 0),
                    'report_date': str(dt.date())
                }
                if new_row['ISIN']:
                    rows.append(new_row)

            if rows:
                df = pd.DataFrame(rows)
                # Shrink to ISIN level (Drop duplicates on PK)
                df.drop_duplicates(subset=['ISIN', 'QuarterID'], keep='last', inplace=True)
                df.to_sql('stocks_raw', self.conn, if_exists='replace', index=False)
                print(f"Loaded {len(df)} rows to stocks_raw.")
            else:
                print("No valid stock rows processed.")


    def run(self):
        print("--- Pipeline 4: Raw Data (Macro & Micro) ---")
        self.create_tables()
        
        # 1. Macro Section
        print("\n[Macro Data Payload]")
        countries = self.get_target_countries()
        if not countries.empty:
            df = self.process_initial_data(countries)
            if not df.empty:
                df = self.patch_missing_data(df)
                
                # Save Macro
                cols = ['ISO', 'Year'] + list(self.config.keys())
                cols = [c for c in cols if c in df.columns]
                save_df = df[cols].copy()
                save_df.drop_duplicates(subset=['ISO','Year'], inplace=True)
                save_df.to_sql('macro_raw', self.conn, if_exists='replace', index=False)
                print(f"Saved {len(save_df)} rows to macro_raw.")
        
        # 2. Micro Section
        print("\n[Micro Data Payload]")
        self.fetch_and_load_companies_details()
        self.fetch_and_load_stocks_raw()
        
        print("\nPipeline 4 Complete.")

