import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from typing import List, Union, Dict, Tuple, Optional

# Attempt relative import, fallback to absolute
try:
    from . import toolbox
except ImportError:
    import MLmodules.toolbox as toolbox

class MacroTimeSeriesPreprocessor:
    def __init__(self, 
                 continuous_cols: List[str], 
                 categorical_cols: List[str] = [], 
                 id_col: str = 'ISO', 
                 time_col: str = 'Year',
                 target_col: Union[str, List[str]] = None, 
                 test_size: float = 0.2, 
                 random_state: int = 42,
                 corr_threshold: float = 0.95,
                 pca_variance: float = 0.99):
        """
        Generic Preprocessor for Time Series / Sequence Data.
        
        Args:
            continuous_cols: List of numerical feature column names.
            categorical_cols: List of categorical feature column names.
            id_col: Column name identifying the sequence entity (e.g., 'ISO', 'StockID').
            time_col: Column name for sorting the sequence (e.g., 'Year').
            target_col: Column name(s) for the target variable (y). If None, y will be None.
            the target col list:
                continues:
                    percent_foreign_holdings REAL,
                    percent_local_holdings REAL,
                    percent_world_in_country REAL,
                    percent_local_investment_on_foreign_from_global REAL,
                    fluidity_growth_1y REAL,
                    fluidity_growth_5y REAL,
                    central_bank_rate REAL,
                    central_bank_rate_growth_1y REAL, 
                    gov_bond_yeald_10y REAL,
                    gov_bond_yeald_3y REAL,
                    gov_bond_risk_rate REAL, 
                    revenue_debt_rate REAL,
                    military_revenue_rate REAL,

                    cap_rate_sector_from_country_<Sector>, for each sector
                    profit_rate_sector_from_country_<Sector>, for each sector
                    growth_beta_sector_from_country_<Sector>, for each sector

                    cap_rate_industrial_from_country_<Industrial>, for each industrial
                    profit_rate_industrial_from_country_<Industrial>, for each industrial
                    growth_beta_industrial_from_country_<Industrial>, for each industrial

                    cap_rate_sector_from_global_<Sector>, for each sector
                    profit_rate_sector_from_global_<Sector>, for each sector
                    growth_beta_sector_from_global_<Sector>, for each sector

                    cap_rate_industrial_from_global_<Industrial>, for each industrial
                    profit_rate_industrial_from_global_<Industrial>, for each industrial
                    growth_beta_industrial_from_global_<Industrial>, for each industrial

                    exchange_yield,
                    companies_growth,
                    PE,
                    PE_growth,
                    country_from_global_market_cap,
                    country_from_global_net_profit,
                    country_growth_beta
                embeddings:
                    ISO (Country Code)
                    region (World Bank Region)
                    incomeLevel (World Bank Income Level)

                there are 11 sectors and 69 industrials,
                there are 60 ISOs, 7 regions and 4 income levels, so total features are:
            21 (base continuous) + (11*3)*2 (sector features) + (69*3)*2 (industrial features) + 60**1.5 (ISO emb - 465) + 7**1.5 (region emb - 19) + 4**1.5 (income emb - 8) = 21 + 66 + 414 + 465 + 19 + 8 = 993 features in total. 

            Records: a Year for each ISO from 1996 to 2025 (30 years) -> 60*30 = 1800 records.
            So the input tensor will be of shape (1800, 993) for the MacroLSTM model.
            The train-test split will be randomal, and be balanced on ISOs and Years.
            The split ratio will be 0.2 (20% test, 80% train).
            
            train_size: Fraction of entities to use for training.
            test_size: Fraction of entities to set aside for testing.
            random_state: Random seed for splitting.

            The preprocessor will handle:
                - Filling NaNs (0 for continuous, 'Unknown' for categorical)
                - Encoding categoricals with LabelEncoder
                - Creating padded tensors grouped by id_col
                - Splitting into train/test sets by entities
                - drop columns that train have low variance.
                - drop columns that have high correlation with other columns (threshold can be set here).
                - sign anomalies and fit logistic regression (with train only) with categorials as dummies with for interactions to find if there are significant interactions between categorials and anomalies in continuous features.
                - drop records from categorials that have high risk to anomaly, according to the logistic regression model.
                - Some features may have log scaling applied (not implemented here, but can be added).
                - Scaling continuous features with StandardScaler, fitted on training data.
                - apply PCA to continuous features to reduce dimensionality (implment here). PCA will save componenets with cumelative explained variance of 99%.
                - note in dict all columns and rows that were dropped and why (low variance, high correlation, anomaly risk).
                - build a dictionary to map column names to their number after pca.
                - convert the data into tensors for model input (x_continuous, x_categorical), and target tensor y if target_col is provided.

        """
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.corr_threshold = corr_threshold
        self.pca_variance = pca_variance
        
        # State Objects
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance)
        self.encoders = {col: LabelEncoder() for col in categorical_cols}
        
        # Metadata
        self.dropped_info = {
            'low_variance': [],
            'high_correlation': [],
            'anomaly_risk_categories': []
        }
        self.pca_map = {} 
        self.active_continuous_cols = [] 

    def fit_transform(self, df: pd.DataFrame) -> Tuple[Dict, Optional[torch.Tensor], Dict, Optional[torch.Tensor]]:
        """
        Fits scalers/encoders on training portion, optimizes features, and transforms data into split tensors.
        """
        # 1. Initial Cleanup
        df_clean = self._clean_and_sort(df)
        
        # 2. Split Entities (Train/Test)
        train_entities, test_entities = self._split_entities(df_clean)
        
        df_train = df_clean[df_clean[self.id_col].isin(train_entities)].copy()
        df_test = df_clean[df_clean[self.id_col].isin(test_entities)].copy()
        
        # 3. Fit Logic (Feature Selection, Scaler, PCA) on TRAIN ONLY
        self._fit_logic(df_train)
        
        # 4. Filter Anomaly Risk if needed
        risky_cats = self.dropped_info['anomaly_risk_categories']
        if risky_cats:
            print(f"Dropping {len(risky_cats)} risky entities from training: {risky_cats}")
            df_train = df_train[~df_train[self.id_col].isin(risky_cats)]
            # We do NOT drop them from test, usually we want to see if we fail on them or not.
            # But the logic above was dropping from both. Let's keep it consistent: drop from train only.
            
        # 5. Fit Encoders (Currently on both to handle test labels safely, 
        # or we fit on train and handle unknown in transform - simplified to fit on train here)
        # Note: If test has new categories, LabelEncoder will fail unless we handle it.
        # We will fit on Train, and use a safe transform method later.
        for col in self.categorical_cols:
            self.encoders[col].fit(df_train[col].astype(str))
            
        # 6. Transform both
        X_train, y_train = self.transform(df_train, is_training_data=True)
        X_test, y_test = self.transform(df_test, is_training_data=False)
        
        return X_train, y_train, X_test, y_test

    def transform(self, df: pd.DataFrame, is_training_data=False) -> Tuple[Dict, Optional[torch.Tensor]]:
        """
        Applies fitted transformations to a new dataset.
        """
        if not self.active_continuous_cols and not hasattr(self, 'active_continuous_cols'):
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
            
        df = self._clean_and_sort(df)
        
        # 1. Scale
        # Filter to active columns only (feature selection applied)
        # Note: if df is missing columns that were selected, this will error (expected).
        df_cont = df[self.active_continuous_cols].copy()
        df_cont_scaled = self.scaler.transform(df_cont)
        
        # 2. PCA
        pca_data = self.pca.transform(df_cont_scaled)
        
        # 3. Rebuild with PCA
        n_components = self.pca.n_components_
        pca_cols = [f'pca_comp_{i}' for i in range(n_components)]
        
        cols_to_drop = set(self.continuous_cols)
        meta_df = df.drop(columns=[c for c in df.columns if c in cols_to_drop], errors='ignore')
        pca_df = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
        df_pca = pd.concat([meta_df, pca_df], axis=1)
        
        # 4. Encode Categoricals
        for col in self.categorical_cols:
            # Handle unknown categories safely
            known_classes = set(self.encoders[col].classes_)
            # Map unknowns to a safe value or existing mode? 
            # Scikit LabelEncoder doesn't support 'unknown'.
            # Strategy: If 'Unknown' is in classes, use it. Else use first class (0) or mode.
            # We assume 'Unknown' was added during clean_and_sort and MIGHT be in classes if it existed in train.
            
            curr_vals = df_pca[col].astype(str)
            unknown_mask = ~curr_vals.isin(known_classes)
            
            if unknown_mask.any():
                # print(f"Warning: {unknown_mask.sum()} unknown labels in {col}")
                if 'Unknown' in known_classes:
                    fill_val = 'Unknown'
                else:
                    # Fallback
                    fill_val = str(known_classes.copy().pop()) 
                
                curr_vals[unknown_mask] = fill_val
                
            df_pca[f'{col}_idx'] = self.encoders[col].transform(curr_vals)

        # 5. Create Tensors
        X, y = self._create_tensors(df_pca, pca_cols)
        return X, y

    def _fit_logic(self, df_train: pd.DataFrame):
        """
        Internal method to fit scalers, PCA, and feature selection logic.
        """
        # A. Low Variance
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(df_train[self.continuous_cols])
        support = selector.get_support()
        low_var_cols = [c for c, keep in zip(self.continuous_cols, support) if not keep]
        self.dropped_info['low_variance'] = low_var_cols
        self.active_continuous_cols = [c for c, keep in zip(self.continuous_cols, support) if keep]
        
        # B. High Correlation
        to_drop = self._find_high_correlation(df_train[self.active_continuous_cols])
        self.dropped_info['high_correlation'] = to_drop
        self.active_continuous_cols = [c for c in self.active_continuous_cols if c not in to_drop]
        
        # C. Anomaly Risk
        risky_cats = self._detect_anomaly_risk(df_train)
        self.dropped_info['anomaly_risk_categories'] = risky_cats
        
        # Note: We do NOT remove risky cats here for fitting the Scaler/PCA necessarily, 
        # but the prompt implied using 'train only' and 'drop info' suggests we might want to clean train first.
        # If we drop risky cats, we should do it before scaling fit.
        if risky_cats:
            mask = ~df_train[self.id_col].isin(risky_cats) # Local mask
            df_for_fit = df_train[mask]
        else:
            df_for_fit = df_train
            
        # D. Fit Scaler
        self.scaler.fit(df_for_fit[self.active_continuous_cols])
        
        # E. Fit PCA (on scaled data)
        scaled_data = self.scaler.transform(df_for_fit[self.active_continuous_cols])
        self.pca.fit(scaled_data)
        
        n_components = self.pca.n_components_
        self.pca_map = {f'pca_comp_{i}': var for i, var in enumerate(self.pca.explained_variance_ratio_)}

    def _create_tensors(self, df: pd.DataFrame, feature_cols: List[str]):
        # X Continuous
        X_cont = toolbox.groupby_to_tensor_pad(
            df, self.id_col, feature_cols, padding_value=0, padding_mode='constant'
        )
        
        # X Categorical
        X_cats = {}
        for col in self.categorical_cols:
            X_cats[f'x_{col}'] = toolbox.groupby_to_tensor_pad(
                df, self.id_col, f'{col}_idx', padding_mode='edge'
            )
            
        # Combine X
        X = {'x_cont': X_cont, **X_cats}
        
        # Y Target
        y = None
        if self.target_col:
            if isinstance(self.target_col, str) and self.target_col in df.columns:
                 y = toolbox.groupby_to_tensor_pad(
                    df, self.id_col, self.target_col, padding_value=0, padding_mode='constant'
                )
            
        return X, y

    def _clean_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.time_col:
            df = df.sort_values([self.id_col, self.time_col])
        else:
            df = df.sort_values([self.id_col])
            
        for c in self.continuous_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        for c in self.categorical_cols:
            if c in df.columns:
                df[c] = df[c].fillna('Unknown')
                
        return df

    def _split_entities(self, df: pd.DataFrame):
        entities = df[self.id_col].unique()
        train_e, test_e = train_test_split(entities, test_size=self.test_size, random_state=self.random_state)
        return train_e, test_e

    def _find_high_correlation(self, df: pd.DataFrame) -> List[str]:
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
        return to_drop

    def _detect_anomaly_risk(self, df: pd.DataFrame) -> List[str]:
        if not self.active_continuous_cols: return []
        means = df[self.active_continuous_cols].mean()
        stds = df[self.active_continuous_cols].std()
        z_scores = ((df[self.active_continuous_cols] - means) / (stds + 1e-8)).abs()
        anomalies = (z_scores > 3).sum(axis=1) > 2
        
        if anomalies.sum() < 5: return []

        try:
            X_log = pd.get_dummies(df[self.categorical_cols], drop_first=True)
            y_log = anomalies.astype(int)
            lr = LogisticRegression(max_iter=1000, solver='lbfgs')
            lr.fit(X_log, y_log)
            
            df_scored = df[[self.id_col]].copy()
            df_scored['anomaly_prob'] = lr.predict_proba(X_log)[:, 1]
            entity_risk = df_scored.groupby(self.id_col)['anomaly_prob'].mean()
            return entity_risk[entity_risk > 0.4].index.tolist()
        except:
            return []

