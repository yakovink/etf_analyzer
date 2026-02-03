"""
in this pipeline we will setup infastructre scheme from PIPELINES.md
at first, we will ensure that the db file exists
then we will create the tables if they do not exist and set their relations,
after that, we will load the json files from data/infastructre as tables and finally,
we will check the pipeline integrity
"""

from global_import import os, pd, pycountry, wb

from clients.database_manager import DatabaseManager
from clients.eod_client import EODClient

INFRA_PATH = os.path.join( 'data', 'infastructre')
DB_PATH = os.path.join(INFRA_PATH,'..', 'etf_analyzer.db')

TABLES = {
    'countries': 'countries.json',
    'exchanges': 'exchanges_new.json',
    'quarters': 'quarters.json'
}

class InfraLoader:
    def __init__(self, db_path=DB_PATH, infra_path=INFRA_PATH):
        self.db_path = db_path
        self.infra_path = infra_path
        self.database_manager = DatabaseManager()
        self.conn = self.database_manager.get_connection()
        self.cursor = self.conn.cursor()
        self.eod_client = EODClient()

    def create_tables(self):
        # Create tables as described in PIPELINES.md
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS countries (ISO TEXT PRIMARY KEY, ISO3 TEXT, country_name TEXT, region TEXT, incomeLevel TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS currencies (currencyID TEXT PRIMARY KEY, currencyName TEXT, Ticker TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS exchanges (code TEXT PRIMARY KEY, name TEXT, ISO TEXT, Currency TEXT, MIC TEXT, FOREIGN KEY(ISO) REFERENCES countries(ISO), FOREIGN KEY(Currency) REFERENCES currencies(currencyID))''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS quarters (QuarterID INTEGER PRIMARY KEY, Year INTEGER, Quarter INTEGER)''')
        self.conn.commit()


    def load_exchanges(self):
        exchanges_df = self.eod_client.get_exchanges_list()
        if exchanges_df is None:
            return
        exchanges_df.rename(
            columns = {"Name":"name","Code":"code","OperatingMIC":"MIC","CountryISO2":"ISO"}
        ,inplace=True)
        
        exchanges_df[['name','code','ISO','Currency','MIC']].to_sql("exchanges",self.conn,if_exists='replace',index=False)

    def get_iso_table(self):
        TAX_HAVEN_ISO_CODES = [    # --- ה"כבדות" (Tech, Pharma, Holdings) ---
    'KY',  # Cayman Islands (עליבאבא, טנסנט, המון טק סיני ואמריקאי)
    'BM',  # Bermuda (חברות שבבים כמו Marvell, חברות ביטוח)
    'VG',  # British Virgin Islands (חברות אחזקה פרטיות וציבוריות)
    'BS',  # Bahamas (פיננסים, קריפטו, תיירות)

    # --- איי התעלה (UK/US Dependencies) ---
    'JE',  # Jersey (אפטיב, אקספריאן, חברות הקשורות ללונדון/ארה"ב)
    'GG',  # Guernsey (קרנות גידור, פיננסים)
    'IM',  # Isle of Man (הימורים, פיננסים)

    # --- ספנות ודגלי נוחות (Shipping) ---
    'PA',  # Panama (קרניבל קרוז, אוניות משא)
    'LR',  # Liberia (ספנות - רויאל קריביאן וכד')
    'MH',  # Marshall Islands (ספנות ותובלה ימית)

    # --- אירופה הקטנה (Holding Hubs) ---
    # אלו מדינות אמיתיות, אבל משמשות לעיתים קרובות כ"צינור" לחברות רוסיות/ישראליות/אחרות
    'CY',  # Cyprus (נפוץ לחברות רוסיות, נדל"ן ופורקס)
    'MT',  # Malta (הימורים, בלוקצ'יין)
    'GI',  # Gibraltar (ביטוח, הימורים)
    'LI',  # Liechtenstein (בנקאות פרטית)

    # --- מרכז אמריקה/אחרים ---
    'BZ',  # Belize
    'SC',  # Seychelles

    # --- היסטוריים/משניים (Legacy) ---
    'AN',  # Netherlands Antilles (פורקה, עדיין מופיעה באג"ח ישנות)
    'CW',  # Curacao (היורשת של AN)
    'SX'   # Sint Maarten (היורשת של AN)
        ]

        iso_list = []
        for country in pycountry.countries:
            iso_list.append({
                'ISO': country.alpha_2,
                'ISO3': country.alpha_3,
                'country_name': country.name,
            })
        iso_list.append({
                'ISO': 'XS',
                'ISO3': 'XAA', 
                'country_name': 'International (Euroclear)'
            })
            
        # אופציונלי: הוספת האיחוד האירופי כ"מדינה" אם נתקלים ב-EU (נדיר ב-ISIN, קיים במאקרו)
        iso_list.append({
            'ISO': 'EU', 
            'ISO3': 'EUR', 
            'country_name': 'European Union'
        })
        df = pd.DataFrame(iso_list)
        df['is_tax_haven'] = df['ISO'].isin(TAX_HAVEN_ISO_CODES)

        wb_data = wb.get_countries()[['name','iso2c','region','incomeLevel']].set_index('iso2c')
        df['country_name'] = df['ISO'].map(wb_data['name']).fillna(df['country_name'])
        df['region'] = df['ISO'].map(wb_data['region'])
        df['incomeLevel'] = df['ISO'].map(wb_data['incomeLevel'])

        
        df.loc[df['ISO']=='TW','region'] = 'East Asia & Pacific'  # תיקון ידני ל-TW (לא קיים ב-WB)
        df.loc[df['ISO']=='TW','incomeLevel'] = 'High income'  # תיקון ידני ל-TW (לא קיים ב-WB)
        
        # לבסוף נגדיר מטבע ייחודי עבור כל מדינה לפי המטבע של הבורסאות שלה
        currency_map = self.database_manager.fetch_df('''
            SELECT DISTINCT ISO, Currency from exchanges where ISO!=''
                                                      ''').set_index('ISO')['Currency']
        df['currency'] = df['ISO'].map(currency_map)


        df.to_sql('countries', self.conn, if_exists='replace', index=False)
        
        

    def fetch_and_load_currencies(self):
        currencies_df = self.eod_client.get_currencies_list()
        if currencies_df is not None:
            clean_df = currencies_df.loc[(currencies_df['Code'].str[:3]=='USD')|(currencies_df['Code'].str.len() == 3)].copy()
            format3 = clean_df['Code'].str.len()==3
            clean_df['target_code'] = clean_df['Code']
            clean_df.loc[~format3, 'target_code'] = clean_df.loc[~format3, 'Code'].str[3:]
            clean_df.drop_duplicates(subset=['target_code'], inplace=True)
            delete_words = ['FX Cross Rate','FX Spot Rate','US Dollar/','USD/','US Dollar','USD','/']
            for word in delete_words:
                clean_df['Name'] = clean_df['Name'].str.replace(word,'')
            clean_df['Name'] = clean_df['Name'].str.strip()
            pd.options.display.max_rows = 200
            clean_df = clean_df[['target_code','Name','Code']].rename(columns={'target_code':'currencyID','Name':'currencyName','Code':'Ticker'})
            clean_df.to_sql('currencies', self.conn, if_exists='replace', index=False)

            print("Loaded currencies from EOD.")
        else:
            print("Failed to load currencies from EOD.")

    def load_quarters(self):
        quarters_df = pd.DataFrame([{"quarter": q, "year": y} for y in range(1996,2026) for q in range(1,5)]).reset_index().rename(columns={"index":"QuarterID"})
        quarters_df.to_sql("quarters",self.conn,if_exists='replace',index=False)

    def check_integrity(self):
        # Simple check: count rows in each table
        for table in TABLES:
            try:
                count = self.cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"Error checking {table}: {e}")

def run():
    loader = InfraLoader()
    loader.create_tables()
    loader.load_quarters()
    loader.fetch_and_load_currencies()
    loader.load_exchanges()
    loader.get_iso_table()
    
    loader.check_integrity()


