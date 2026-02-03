
from global_import import pd, sqlite3, os, Any, List, Dict
class DatabaseManager:
    def __init__(self):

        self.db_path = os.path.join( 'data', 'etf_analyzer.db')
        #print current full path
        print(os.getcwd())
        print(f"DatabaseManager initialized with DB path: {self.db_path}")
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Any]:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchall()
            conn.commit()
            return result
        finally:
            conn.close()

    def fetch_df(self, query: str, params: tuple = ()) -> pd.DataFrame:
        conn = self.get_connection()
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()

    def save_df(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """
        Saves DataFrame to DB. 
        if_exists: 'fail', 'replace', 'append'. 
        Note: 'replace' drops the table! For updates on PK, we need custom logic.
        """
        conn = self.get_connection()
        try:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        finally:
            conn.close()
            
    def upsert_dict(self, table_name: str, data: List[Dict], pk_cols: List[str]):
        """
        Performs INSERT OR REPLACE / UPSERT logic for a list of records.
        Using SQLite INSERT OR REPLACE is simple but overwrites non-specified columns with default/null if replace happens.
        For true partial update (UPSERT), we need ON CONFLICT DO UPDATE.
        
        Args:
            table_name: Table name
            data: List of dictionaries matching column names
            pk_cols: List of Primary Key columns for conflict resolution
        """
        if not data: return
        
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            
            # Assuming all dicts have same keys
            keys = list(data[0].keys())
            columns = ', '.join(keys)
            placeholders = ', '.join(['?'] * len(keys))
            
            # Construct ON CONFLICT clause for true upsert (PostgreSQL/SQLite 3.24+)
            # INSERT INTO table (col1, col2) VALUES (?, ?) ON CONFLICT(pk) DO UPDATE SET col1=excluded.col1, ...
            
            pk_str = ', '.join(pk_cols)
            update_set = ', '.join([f"{k}=excluded.{k}" for k in keys if k not in pk_cols])
            
            if not update_set:
                # If only PK or no cols to update (ignore)
                sql = f"INSERT OR IGNORE INTO {table_name} ({columns}) VALUES ({placeholders})"
            else:
                sql = f"""
                    INSERT INTO {table_name} ({columns}) 
                    VALUES ({placeholders}) 
                    ON CONFLICT({pk_str}) 
                    DO UPDATE SET {update_set}
                """
            
            values = [tuple(d[k] for k in keys) for d in data]
            cursor.executemany(sql, values)
            conn.commit()
        except Exception as e:
            print(f"Upsert Error on {table_name}: {e}")
        finally:
            conn.close()
