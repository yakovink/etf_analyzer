"""
in this pipeline we will setup core scheme from PIPELINES.md
at first, we will ensure that the db file exists and infastructre is loaded,
then we will create the tables if they do not exist and set their relations,
after that, we will load the data from EOD and apply the pipeline logic. Finally,
we will check the pipeline integrity
"""

from global_import import os, pd, time

from clients.eod_client import EODClient
from clients.database_manager import DatabaseManager

DATA_PATH = os.path.join( 'data')
DB_PATH = os.path.join(DATA_PATH, 'etf_analyzer.db')

class CoreLoader:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.database_manager = DatabaseManager()
        self.conn = self.database_manager.get_connection()
        self.cursor = self.conn.cursor()
        self.eod_client = EODClient()

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS currency_rates (currencyID TEXT, QuarterID INTEGER, RateToILS REAL, RateToUSD REAL, PRIMARY KEY(currencyID, QuarterID))''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS companies (ISIN TEXT PRIMARY KEY, CompanyName TEXT, ISO TEXT, FOREIGN KEY(ISO) REFERENCES countries(ISO))''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS TradePipelines (PipelineID INTEGER PRIMARY KEY AUTOINCREMENT, ISIN TEXT, ExchangeID TEXT, StockTicker TEXT, FOREIGN KEY(ISIN) REFERENCES companies(ISIN), FOREIGN KEY(ExchangeID) REFERENCES exchanges(code))''')
        self.conn.commit()




    def fetch_and_load_traded_stocks(self):
        # Use EODClient to fetch all traded stocks for all exchanges
        exchanges = pd.read_sql('SELECT code FROM exchanges', self.conn)
        all_stocks = []
        def build_exchange_stocks(exchange_row):
            exchange_stocks = self.eod_client.get_exchange_stocks_list(exchange_row['code'])
            if exchange_stocks is not None:
                exchange_stocks['ExchangeID'] = exchange_row['code']
                all_stocks.append(exchange_stocks)
            else:
                raise Exception(f"Failed to fetch stocks for exchange {exchange_row['code']}")
        exchanges.apply(build_exchange_stocks, axis=1)
        # Save to TradePipelines (mapping exchange code to ExchangeID if needed)
        # For demo, assume exchange code is ExchangeID
        df = pd.concat(all_stocks, ignore_index=True)
        df.dropna(subset=['Isin'], inplace=True)
        df['CodeLength'] = df['Code'].str.len()
        
        df.sort_values(by = 'CodeLength', inplace=True)
        df.drop_duplicates(subset=['Isin','ExchangeID'], keep='first', inplace=True)


        df = df.loc[df['Type']=='Common Stock', ['Code','Name', 'ExchangeID', 'Isin','Country']]
        companies_df = df[['Isin', 'Name']].drop_duplicates(subset=['Isin'],keep='first').rename(columns={'Isin':'ISIN', 'Name':'CompanyName'})
        companies_df['ISO'] = companies_df['ISIN'].str[:2]  # Simplistic ISO extraction from ISIN
        


        companies_df.to_sql('companies', self.conn, if_exists='replace', index=False)

        pipelines_df = df.reset_index().rename(columns={'Code': 'StockTicker', 'Isin': 'ISIN','index':'PipelineID'})
        
        pipelines_df[['PipelineID','ISIN', 'ExchangeID', 'StockTicker']].to_sql('TradePipelines', self.conn, if_exists='replace', index=False)

        print(f"Loaded TradePipelines from EOD.")

    

    def fetch_and_load_currencies_rates(self):
        all_rates = []
        def fetch_rates_for_currency(row: pd.Series):
            
            rates = self.eod_client.get_currencies_historical_rates(row['Ticker'], start_date='1995-01-01')
            
            if rates is None:
                raise Exception(f"Failed to fetch rates for currency {row['currencyID']}")
                
            rates['currencyID'] = row['currencyID']
            all_rates.append(rates)

        # Fetch currency rates from EOD and load into currency_rates table
        currencies = pd.read_sql('SELECT currencyID, Ticker FROM currencies', self.conn)
        quarters = pd.read_sql('SELECT QuarterID, Year, Quarter FROM quarters', self.conn)
        
        currencies.apply(fetch_rates_for_currency, axis=1)
        final_rates_df = pd.concat(all_rates, ignore_index=True)
        
        final_rates_df['year'] = pd.to_datetime(final_rates_df['date']).dt.year
        final_rates_df['quarter'] = pd.to_datetime(final_rates_df['date']).dt.quarter
        final_rates_df = final_rates_df.merge(quarters, on=['year','quarter'], how='left')
        final_rates_df = final_rates_df.merge(currencies, left_on='currencyID', right_on='currencyID', how='left')
        ils_rates = final_rates_df[final_rates_df['currencyID']=='ILS'][['date','close']].rename(columns={'close':'Close_ILS'})
        final_rates_df = final_rates_df.merge(ils_rates[['date','Close_ILS']],on='date', how='left')
        final_rates_df['RateToILS'] = final_rates_df['close'] / final_rates_df['Close_ILS']
        grouped = final_rates_df.groupby(['currencyID','QuarterID'])[['close','RateToILS']].mean().reset_index()
        grouped.rename(columns={'close':'RateToUSD'}, inplace=True)

        grouped.to_sql('currency_rates', self.conn, if_exists='replace', index=False)
        

    def validate_isin_iso(self):
        # Check for ISIN/ISO mismatches
        companies = pd.read_sql('SELECT ISIN, ISO FROM companies', self.conn)
        countries = pd.read_sql('SELECT ISO FROM countries', self.conn)
        missing_iso = set(companies['ISO']) - set(countries['ISO'])
        if missing_iso:
            print(f"Warning: Missing ISO codes in countries table: {missing_iso}")
            # Optionally, add missing ISO to countries.json/db
        else:
            print("All ISIN/ISO codes are valid.")

    def check_integrity(self):
        for table in ['currency_rates', 'companies', 'TradePipelines']:
            try:
                count = self.cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"Error checking {table}: {e}")

def run():
    loader = CoreLoader()
    loader.create_tables()

    loader.fetch_and_load_traded_stocks()
    loader.fetch_and_load_currencies_rates()
    loader.validate_isin_iso()
    loader.check_integrity()


