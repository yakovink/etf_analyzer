"""
Pipeline 3: Portfolio Loader
Strictly follows PIPELINES.md for managers, funds, holdings.
- Managers from JSON, funds from EOD, holdings via Edgar API.
"""


from global_import import pd
from clients.database_manager import DatabaseManager
import clients.edgar_client as edgar_client


class PortfolioLoader:
    def __init__(self):
        self.db = DatabaseManager()
        self.conn = self.db.get_connection()
        self.cursor = self.conn.cursor()

    def create_tables(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Managers (
            ManagerID INTEGER PRIMARY KEY, ManagerName TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS funds (
             SeriesID TEXT PRIMARY KEY, fund_ticker TEXT , fund_name TEXT, ManagerID INTEGER, net_assets REAL)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS holdings (
            HoldingID TEXT PRIMARY KEY,SeriesID TEXT, ISIN TEXT, weight REAL)''')
        self.conn.commit()

    def load_funds_data(self):
        funds_data : pd.DataFrame = edgar_client.get_fund_table(2025, 4)
        funds_table = funds_data[["Entity Name","Class Name","Class Ticker","NET_ASSETS","SERIES_ID"]].drop_duplicates()
        funds_table.rename(columns={
            "Class Ticker": "fund_ticker",
            "Entity Name": "managerName",
            "NET_ASSETS": "net_assets",
            "Class Name": "fund_name",
            "SERIES_ID": "SeriesID"
        }, inplace=True
        )
        managers_df = funds_table[["managerName"]].drop_duplicates().reset_index().rename(columns={"index":"ManagerID"})
        funds_table = funds_table.merge(managers_df, on="managerName", how="left")
        holdings_table = funds_data[['FinalISIN','PERCENTAGE',"SERIES_ID","HOLDING_ID"]].rename(columns={
            'FinalISIN':'ISIN',
            'PERCENTAGE':'weight',
            "SERIES_ID":"SeriesID",
            "HOLDING_ID":"HoldingID"
        }).drop_duplicates(subset=['ISIN','SeriesID'],keep='last').reset_index(drop=True)

        managers_df.to_sql('Managers', self.conn, if_exists='replace', index=False)
        funds_table[['SeriesID','fund_ticker','fund_name','ManagerID','net_assets']].to_sql('funds', self.conn, if_exists='replace', index=False)
        holdings_table.to_sql('holdings', self.conn, if_exists='replace', index=False)

    def check_integrity(self):
        # Simple integrity checks
        self.cursor.execute("SELECT COUNT(*) FROM funds")
        funds_count = self.cursor.fetchone()[0]
        self.cursor.execute("SELECT COUNT(*) FROM holdings")
        holdings_count = self.cursor.fetchone()[0]
        print(f"Funds loaded: {funds_count}, Holdings loaded: {holdings_count}")



def run():
    loader = PortfolioLoader()
    loader.create_tables()
    loader.load_funds_data()
    loader.check_integrity()
