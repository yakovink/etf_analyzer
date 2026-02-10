"""
Pipeline 6: AI Analysis
- Module: Macro Analysis
- Model: LSTM-BiLSTM
- Task: Fit macro indicators and predict future trends/scores.
"""

from global_import import pd, np, Literal, Tuple, torch
from clients.database_manager import DatabaseManager
from MLmodules.macro_model import MacroLSTM
from MLmodules.preprocessing import MacroTimeSeriesPreprocessor
import torch.nn as nn
import torch.optim as optim

class AnalysisLoader:
    def __init__(self):
        self.db = DatabaseManager()
        self.conn = self.db.get_connection()
        self.cursor = self.conn.cursor()
        
    def create_tables(self):
        # 17. Macro Analysis
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS macro_analysis (
            ISO TEXT, 
            QuarterID INTEGER, 
            liquidity_score REAL, 
            predicted_local_cash_stream_to_stocks REAL, 
            predicted_forign_cash_stream_to_stocks REAL, 
            PRIMARY KEY (ISO, QuarterID))''')

        # 18. Stocks Analysis
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS stocks_analysis (
            ISIN TEXT, 
            CycleID INTEGER, 
            PredictedLength REAL, 
            PredictedSuccesfullProb REAL, 
            PRIMARY KEY (ISIN, CycleID))''')

        # 19. ETF Analysis
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS etf_analysis (
            etf_ticker TEXT PRIMARY KEY, 
            weighted_prob REAL, 
            weighted_cycle_length REAL, 
            weighted_pe_deriv REAL, 
            coverage_pct REAL, 
            last_updated TEXT)''')
            
        self.conn.commit()
        print("Pipeline 6 Tables initialized.")

    def load_data(self):
        print("Loading macro data...")
        # Join macro_calculated with countries info
        query = """
        SELECT t1.*, t2.region, t2.income_level
        FROM macro_calculated t1
        LEFT JOIN countries t2 ON t1.ISO = t2.ISO
        ORDER BY t1.ISO, t1.Year
        """
        df = self.db.fetch_df(query)
        if df.empty:
            print("No data found in macro_calculated.")
            return None
        return df

    def run_ml_analysis(self):
        print("--- Pipeline 6: ML Analysis ---")
        self.create_tables()
        
        # 1. Macro Analysis
        df = self.load_data()
        if df is not None:
            # 2. Config Preprocessor
            # Define columns
            cat_cols = ['ISO', 'region', 'income_level']
            exclude_cols = ['Year'] + cat_cols
            cont_cols = [c for c in df.columns if c not in exclude_cols]
            
            prep = MacroTimeSeriesPreprocessor(
                continuous_cols=cont_cols,
                categorical_cols=cat_cols,
                id_col='ISO',
                time_col='Year',
                target_col=None, # Reconstruction task
                test_size=0.2,
                pca_variance=0.99
            )
            
            # 3. Fit & Transform
            X_train, _, X_test, _ = prep.fit_transform(df)
            
            # Extract Dimensions for Model
            # PCA might reduce features, so we check the shape of x_cont
            n_continuous = X_train['x_cont'].shape[2] 
            n_isos = len(prep.encoders['ISO'].classes_)
            n_regions = len(prep.encoders['region'].classes_)
            n_incomes = len(prep.encoders['income_level'].classes_)
            
            print(f"Training Model with: Cont={n_continuous}, ISOs={n_isos}, Regs={n_regions}, Incs={n_incomes}")
            
            # 4. Initialize Model
            model = MacroLSTM(
                n_isos=n_isos,
                n_regions=n_regions,
                n_incomes=n_incomes,
                n_continuous=n_continuous,
                device='cpu' # Or 'cuda' if available
            )
            
            # 5. Train
            # Unpack X_train dict for fit args
            model.fit(
                x_iso=X_train['x_ISO'], 
                x_reg=X_train['x_region'], 
                x_inc=X_train['x_income_level'], 
                x_cont=X_train['x_cont'], 
                epochs=20
            )
            
            # TODO: Evaluation on X_test?
            # model.eval()
            # ...
            
        print("ML Analysis Module Executed.")

    def run(self):
        self.run_ml_analysis()

    def check_integrity(self):
        for table in ['macro_analysis', 'stocks_analysis', 'etf_analysis']:
            try:
                count = self.cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"Error checking {table}: {e}")

if __name__ == '__main__':
    loader = AnalysisLoader()
    loader.run()
