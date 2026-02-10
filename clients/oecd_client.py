"""
OECD Client - Fetches data from OECD SDMX 3.0 API
Handles Annual Sector Accounts and other OECD datasets
"""

from global_import import requests, pd, Optional, Dict, List, warnings, json, time, datetime

# Suppress SSL warnings for OECD API
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class OECDClient:
    def __init__(self):
        self.base_url = "https://sdmx.oecd.org/public/rest/data"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'ETF-Analyzer/1.0'
        })
    
    def fetch_sdmx_data(self, dataflow: str, key: str, start_period: Optional[str] = None, 
                        end_period: Optional[str] = None) -> pd.DataFrame:
        """
        Fetches data from OECD SDMX 3.0 API.
        
        Args:
            dataflow: Full dataflow identifier (e.g., "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.0")
            key: SDMX key with dimensions (e.g., "A....S1...A..F511+F521..USD......")
            start_period: Start period (e.g., "2010")
            end_period: End period (e.g., "2024")
            
        Returns:
            DataFrame with columns: REF_AREA (country), TIME_PERIOD (year), OBS_VALUE (value)
        """
        try:
            # Build URL - Note: comma after dataflow, then slash before key
            url = f"{self.base_url}/{dataflow},/{key}"
            
            # Add parameters
            params = {}
            if start_period:
                params['startPeriod'] = start_period
            if end_period:
                params['endPeriod'] = end_period
            
            # Build full URL with parameters for debugging
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{url}?{param_str}" if param_str else url
            
            print(f"    Fetching from OECD: {dataflow}")
            print(f"    Key: {key}")
            print(f"    Full URL: {full_url}")
            
            # Make request with explicit JSON Accept header
            # Note: OECD API returns XML by default, must explicitly request JSON
            headers = {
                'Accept': 'application/vnd.sdmx.data+json;version=2.0;urn=true',
                'Accept-Language': 'en',
                'Accept-Encoding': 'gzip, deflate, br, zstd'
            }
            response = self.session.get(url, params=params, headers=headers, timeout=60, verify=False)
            
            if response.status_code != 200:
                print(f"    ✗ OECD API error {response.status_code}: {response.text[:200]}")
                return pd.DataFrame()
            
            # Parse SDMX-JSON response
            data = response.json()
            
            if 'errors' in data:
                print(f"    ✗ OECD API reported errors: {data['errors']}")
            
            # Extract observations from SDMX structure
            observations = self._parse_sdmx_json(data)
            
            if observations:
                df = pd.DataFrame(observations)
                print(f"    ✓ Fetched {len(df)} records from OECD")
                return df
            else:
                print(f"    ✗ No observations found in OECD response")
                return pd.DataFrame()
                
        except requests.exceptions.Timeout:
            print(f"    ✗ OECD API timeout")
            return pd.DataFrame()
        except Exception as e:
            print(f"    ✗ Error fetching OECD data: {e}")
            return pd.DataFrame()
    
    def _parse_sdmx_json(self, data: Dict) -> List[Dict]:
        """
        Parses SDMX-JSON 2.0 format (series-keyed) used by OECD.
        
        Returns list of dicts with keys: REF_AREA, TIME_PERIOD, OBS_VALUE
        """
        observations = []
        
        try:
            # SDMX-JSON structure: data.dataSets[0].series
            if 'data' not in data:
                return observations
            
            data_msg = data['data']
            
            # Get structure (dimensions split into series and observation)
            structure = data_msg.get('structure', {})
            dimensions = structure.get('dimensions', {})
            
            # Parse series dimensions (e.g., REF_AREA at index 2)
            series_dims = dimensions.get('series', [])
            ref_area_idx = None
            ref_area_values = {}
            
            for idx, dim in enumerate(series_dims):
                if dim.get('id') == 'REF_AREA':
                    ref_area_idx = idx
                    ref_area_values = {i: v['id'] for i, v in enumerate(dim.get('values', []))}
                    break
            
            # Parse observation dimensions (e.g., TIME_PERIOD at index 0)
            obs_dims = dimensions.get('observation', [])
            time_period_values = {}
            
            for dim in obs_dims:
                if dim.get('id') == 'TIME_PERIOD':
                    time_period_values = {i: v['id'] for i, v in enumerate(dim.get('values', []))}
                    break
            
            # Extract data from series
            datasets = data_msg.get('dataSets', [])
            
            for dataset in datasets:
                series_data = dataset.get('series', {})
                
                for series_key, series_value in series_data.items():
                    # Parse series key (e.g., "0:0:2:0:0:0:0:0:0:1:0:0:0:0:0:0:0:0")
                    key_parts = series_key.split(':')
                    
                    # Extract country from series dimensions
                    country = None
                    if ref_area_idx is not None and len(key_parts) > ref_area_idx:
                        country_idx = int(key_parts[ref_area_idx])
                        country = ref_area_values.get(country_idx)
                    
                    # Get observations within this series
                    obs_data = series_value.get('observations', {})
                    
                    for obs_key, obs_value in obs_data.items():
                        # obs_key is index into observation dimensions (just TIME_PERIOD in this case)
                        time_idx = int(obs_key)
                        time_period = time_period_values.get(time_idx)
                        
                        # obs_value is array: [value, status, attributes...]
                        value = obs_value[0] if isinstance(obs_value, list) and len(obs_value) > 0 else obs_value
                        
                        if country and time_period and value is not None:
                            try:
                                observations.append({
                                    'REF_AREA': country,
                                    'TIME_PERIOD': time_period,
                                    'OBS_VALUE': float(value)
                                })
                            except (ValueError, TypeError):
                                # Skip invalid values
                                pass
            
            return observations
            
        except Exception as e:
            print(f"    ✗ Error parsing SDMX response: {e}")
            import traceback
            traceback.print_exc()
            return observations
    
    def fetch_indicator(self, params: Dict, start_year: int = 1995, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Convenience method to fetch indicator with config params.
        
        Args:
            params: Dict with 'dataflow' and 'key' from macro_indicators.json
            start_year: Start year (default 1995)
            end_year: End year (default current year)
            
        Returns:
            DataFrame with columns: ISO3, Year, Value
        """
        if end_year is None:
            end_year = datetime.now().year
        
        dataflow = params.get('dataflow')
        key = params.get('key')
        
        if not dataflow or not key:
            print(f"    ✗ Missing OECD parameters")
            return pd.DataFrame()
        
        # Fetch data in 5-year batches to avoid timeout/size issues
        all_dfs = []
        batch_size = 5
        max_retries_rate_limit = 3
        ssl_retry_delay = 10  # seconds
        
        for batch_start in range(start_year, end_year + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, end_year)
            
            print(f"    Fetching batch: {batch_start}-{batch_end}")
            
            # Retry logic with unlimited SSL retries and limited rate limit retries
            rate_limit_attempts = 0
            ssl_attempts = 0
            
            while True:
                try:
                    df_batch = self.fetch_sdmx_data(
                        dataflow=dataflow,
                        key=key,
                        start_period=str(batch_start),
                        end_period=str(batch_end)
                    )
                    
                    if not df_batch.empty:
                        all_dfs.append(df_batch)
                    
                    # Add delay between successful batches to avoid rate limiting
                    time.sleep(2)
                    break  # Success, move to next batch
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # Handle SSL errors - retry indefinitely with 10s delay
                    if 'SSL' in error_str or 'DECRYPTION_FAILED' in error_str or 'BAD_RECORD_MAC' in error_str:
                        ssl_attempts += 1
                        print(f"      ⚠ SSL error (attempt {ssl_attempts}). Waiting {ssl_retry_delay}s before retry...")
                        time.sleep(ssl_retry_delay)
                        continue  # Retry indefinitely
                    
                    # Handle rate limiting - retry up to max_retries_rate_limit times
                    elif '429' in error_str:
                        rate_limit_attempts += 1
                        if rate_limit_attempts < max_retries_rate_limit:
                            wait_time = 10 * rate_limit_attempts
                            print(f"      ⚠ Rate limited. Waiting {wait_time}s before retry {rate_limit_attempts}/{max_retries_rate_limit}...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"      ✗ Max rate limit retries reached. Skipping batch.")
                            break  # Give up on this batch
                    
                    # Other errors - give up on this batch
                    else:
                        break
        
        # Concatenate all batches
        if not all_dfs:
            print(f"    ✗ No data retrieved from any batch")
            return pd.DataFrame()
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"    ✓ Combined {len(all_dfs)} batches into {len(df)} total records")
        
        # Standardize columns
        df = df.rename(columns={
            'REF_AREA': 'ISO3',
            'TIME_PERIOD': 'Year',
            'OBS_VALUE': 'Value'
        })
        
        # Convert Year to int (handle formats like "2020" or "2020-Q1")
        def extract_year(time_str):
            try:
                if '-' in str(time_str):
                    return int(str(time_str).split('-')[0])
                return int(str(time_str)[:4])
            except:
                return None
        
        df['Year'] = df['Year'].apply(extract_year)
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        
        # Group by country and year (in case of quarterly data)
        df = df.groupby(['ISO3', 'Year'])['Value'].sum().reset_index()
        
        return df
