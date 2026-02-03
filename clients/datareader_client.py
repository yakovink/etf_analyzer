
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
        

