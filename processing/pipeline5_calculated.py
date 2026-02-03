"""
Pipeline 5: Calculated Data Loader
- Scheme 5: Calculated
- Input: stocks_raw (from Pipeline 4)
- Tables: stocks_calculated, stocks_cycle, (optional: macro_calculated)
"""

from global_import import pd, Literal, np, Tuple
from clients.database_manager import DatabaseManager

class CalculatedLoader:
    def __init__(self):
        self.db = DatabaseManager()
        self.conn = self.db.get_connection()
        self.cursor = self.conn.cursor()
        self.currency_map_quarter, self.currency_map_year = self.get_currencies_map()


    def get_aggreated_micro(self):
        """
        Docstring for get_aggreated_micro
        
        :param self: Description
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        cap_rate_sector_from_country_<Sector>,
        profit_rate_sector_from_country_<Sector>,
        growth_beta_sector_from_country_<Sector>,

        cap_rate_industrial_from_country_<Industrial>,
        profit_rate_industrial_from_country_<Industrial>,
        growth_beta_industrial_from_country_<Industrial>,

        cap_rate_sector_from_global_<Sector>,
        profit_rate_sector_from_global_<Sector>,
        growth_beta_sector_from_global_<Sector>,

        cap_rate_industrial_from_global_<Industrial>,
        profit_rate_industrial_from_global_<Industrial>,
        growth_beta_industrial_from_global_<Industrial>,

        exchange_yield,
        companies_growth,
        PE,
        PE_growth,
        country_from_global_market_cap,
        country_from_global_net_profit,
        country_growth_beta
        """
        micro_agg = self.db.fetch_df('''
                                    SELECT ISO,Year,Sector,Industrial,
                                    SUM(market_cap) as total_market_cap,
                                    SUM(net_profit) as total_net_profit
                                    FROM stocks_raw
                                    left join companies_details
                                    on stocks_raw.ISIN=companies_details.ISIN
                                    left join companies
                                    on stocks_raw.ISIN=companies.ISIN
                                    left join quarters
                                    on stocks_raw.QuarterID=quarters.QuarterID
                                    where Quarter = 4
                                    group by ISO,Year,Sector,Industrial
                                    ''')
        # we want to calc 4 categorial rates: local sector from country, local industrial from country, local sector from global sector, local industrial from global industrial
        # for each rate we will calc: cap_rate, profit_rate, growth_beta

        # first we will calc the local sector and industrial sums
        sectors_calculation = micro_agg.groupby(['ISO','Year','Sector'])[['total_net_profit','total_market_cap']].sum().reset_index()
        industrial_calculation = micro_agg.groupby(['ISO','Year','Industrial'])[['total_net_profit','total_market_cap']].sum().reset_index()

        # the local sector from country
        sectors_calculation['cap_rate_sector_from_country'] = sectors_calculation['total_net_profit'] / sectors_calculation.groupby('ISO')['total_market_cap'].transform('sum')
        sectors_calculation['profit_rate_sector_from_country'] = sectors_calculation['total_net_profit'] / sectors_calculation.groupby('ISO')['total_net_profit'].transform('sum')
        sectors_calculation['growth_beta_sector_from_country'] = sectors_calculation['profit_rate_sector_from_country'] / sectors_calculation['cap_rate_sector_from_country']
        # the local industrial from country
        industrial_calculation['cap_rate_industrial_from_country'] = industrial_calculation['total_net_profit'] / industrial_calculation.groupby('ISO')['total_market_cap'].transform('sum')
        industrial_calculation['profit_rate_industrial_from_country'] = industrial_calculation['total_net_profit'] / industrial_calculation.groupby('ISO')['total_net_profit'].transform('sum')
        industrial_calculation['growth_beta_industrial_from_country'] = industrial_calculation['profit_rate_industrial_from_country'] / industrial_calculation['cap_rate_industrial_from_country']
        # the local sector from global sector
        sectors_calculation['cap_rate_sector_from_global'] = sectors_calculation['total_net_profit'] / sectors_calculation.groupby('Sector')['total_market_cap'].transform('sum')
        sectors_calculation['profit_rate_sector_from_global'] = sectors_calculation['total_net_profit'] / sectors_calculation.groupby('Sector')['total_net_profit'].transform('sum')
        sectors_calculation['growth_beta_sector_from_global'] = sectors_calculation['profit_rate_sector_from_global'] / sectors_calculation['cap_rate_sector_from_global']
        # the local industrial from global industrial
        industrial_calculation['cap_rate_industrial_from_global'] = industrial_calculation['total_net_profit'] / industrial_calculation.groupby('Industrial')['total_market_cap'].transform('sum')
        industrial_calculation['profit_rate_industrial_from_global'] = industrial_calculation['total_net_profit'] / industrial_calculation.groupby('Industrial')['total_net_profit'].transform('sum')
        industrial_calculation['growth_beta_industrial_from_global'] = industrial_calculation['profit_rate_industrial_from_global'] / industrial_calculation['cap_rate_industrial_from_global']
        #transform to cap_rate_SectorA, cap_rate_SectorB, profit_rate_SectorA, ...
        sectors_unstacks = sectors_calculation.pivot(index=['ISO','Year'], columns='Sector', values=['cap_rate_sector_from_country','profit_rate_sector_from_country','growth_beta_sector_from_country','cap_rate_sector_from_global','profit_rate_sector_from_global','growth_beta_sector_from_global'])
        industrial_unstacks = industrial_calculation.pivot(index=['ISO','Year'], columns='Industrial', values=['cap_rate_industrial_from_country','profit_rate_industrial_from_country','growth_beta_industrial_from_country','cap_rate_industrial_from_global','profit_rate_industrial_from_global','growth_beta_industrial_from_global'])
        # for last we will calc the overall country rates
        summed = micro_agg.groupby(['ISO','Year'])[['total_net_profit','total_market_cap']].sum()
        summed['exchange_yield'] = summed['total_market_cap'] / summed.groupby('ISO')['total_net_profit'].shift(1) - 1
        summed['companies_growth'] = summed['total_net_profit'] / summed.groupby('ISO')['total_net_profit'].shift(1) - 1
        summed['PE'] = summed['total_market_cap'] / summed['total_net_profit']
        summed['PE_growth'] = summed['PE'] / summed.groupby('ISO')['PE'].shift(1) - 1
        summed['country_from_global_market_cap'] = summed['total_market_cap'] / summed.groupby('Year')['total_market_cap'].transform('sum')
        summed['country_from_global_net_profit'] = summed['total_net_profit'] / summed.groupby('Year')['total_net_profit'].transform('sum')
        summed['country_growth_beta'] = summed['country_from_global_net_profit'] / summed['country_from_global_market_cap']

        unstacked = pd.concat([sectors_unstacks, industrial_unstacks,summed], axis=1)
        return unstacked.reset_index()

    def calc_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Docstring for calc_growth
        
        :param self: Description
        :param df: Description
        :type df: pd.DataFrame
        :return: Description
        :rtype: DataFrame
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        - stock_price_yield
        - profit_growth_expection
        - profit_growth
        - growth_model_error

        """
        df['stock_price_yield'] = df['market_cap'] / (df.groupby('ISIN')['market_cap'].shift(1) + 1e-6) - 1
        # on the focus of the fundemental model is to calc the current growth expection and the yearly growth expetetions by the stock_price
        # the growth expection will be calced by the growth on the profits by the investment (represents new books value)
        df['marginal_profit'] = df['net_profit'] - df.groupby('ISIN')['net_profit'].shift(1) - 1
        df['profit_growth_expection'] = df['marginal_profit'] / (df['investment'] + 1e-6)
        df['profit_growth'] = df['net_profit'] / df.groupby('ISIN')['net_profit'].shift(1) - 1
        df['growth_model_error'] = df['profit_growth_expection'] - df['profit_growth']
        return df

    def calc_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Docstring for calc_years
        
        :param self: Description
        :param df: Description
        :type df: pd.DataFrame
        :return: Description
        :rtype: DataFrame
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        - premium_years_nonGrowth
        - premium_years_Growth
        """
        df['premium'] = df['market_cap'] - df['books_value']
        df['premium_years_nonGrowth'] = df['premium'] / (df['net_profit'] + 1e-6)

        # then calc the n of engneering series sum, where sum = premium_profit_nonGrowth, q = 1 + expected_growth and a = 1
        # the basic formula is: S_n = a(1-q^n)/(1-q) -> S_n*(1-q)/a = 1-q^n -> 1-S_n*(1-q)/a = q^n -> log(1-S_n*(1-q)/a) = n*log(q) -> n = log(1-S_n*(1-q)/a) / log(q)
        
        df['premium_years_Growth'] = np.log(1 - df['premium_years_nonGrowth'] * (df['expected_growth'])) / np.log(1 + df['expected_growth'] + 1e-6)
        return df

    def calc_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Docstring for calc_cycle
        
        :param self: Description
        :param df: Description
        :type df: pd.DataFrame
        :return: Description
        :rtype: DataFrame
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        For cycles:
        - CycleID
        - StartedQuarter
        - PickQuarter
        - EndQuarter
        - IsSuccesfull
        For stocks_calculated:
        - cycle_id

        """
        # then calc the location on the bussiness cycle
        #lets said that marginal is stable if between -1% to 1% quarterly
        df['derivative_pe'] = df['PE'] / (df.groupby('ISIN')['PE'].shift(1) + 1e-6) - 1
        df['2nd_derivative_pe'] = df['derivative_pe'] - df.groupby('ISIN')['derivative_pe'].shift(1)
        df['derivative_type'] = df['derivative_pe'].apply(lambda x: 'expansion' if x > 0.01 else ('recession' if x < -0.01 else 'stable'))
        df['2nd_derivative_type'] = df['2nd_derivative_pe'].apply(lambda x: 'acceleration' if x > 0.01 else ('deceleration' if x < -0.01 else 'stable'))
        # we will define that the cycle end when we cumelative reduce of 20% or more from the pick. we will mark the pick quarter and the end quarter
        # we will define that the cycle start when the last cycle end or the first quarter in the data
        # we will start with run a loop to mark the cycles. if derivative is negative we are in recession until we reach the end condition, then we mark the pick quarter and start a new cycle
        # data is already ordered by ISO, ISIN, QuarterID
        df['last_4quarters_expansion'] = df.groupby('ISIN')['derivative_type'].transform(lambda x: x.rolling(window=4).apply(lambda y: all(v == 'expansion' for v in y), raw=True).fillna(0))
        df['last_4quarters_recession'] = df.groupby('ISIN')['derivative_type'].transform(lambda x: x.rolling(window=4).apply(lambda y: all(v == 'recession' for v in y), raw=True).fillna(0))
        
        print("Detecting Market Cycles (Bull/Bear)...")
        
        def _analyze_isin_cycles(group: pd.DataFrame) -> list:
            group = group.sort_values('QuarterID')
            cycles = []
            
            # State Machine
            # State: 'neutral', 'bull', 'bear'
            state = 'neutral'
            current_cycle = None
            
            # Triggers are at the END of the 4-quarter window.
            # So if trigger at index i, the phase started at index i-3.
            
            # Converting columns to numpy for speed
            q_ids = group['QuarterID'].values
            bull_trigs = group['last_4quarters_expansion'].values # 1.0 or 0.0
            bear_trigs = group['last_4quarters_recession'].values
            
            # Create a Market Cap Lookup for this ISIN to calc yield later
            mc_map = group.set_index('QuarterID')['market_cap'].to_dict()

            for i in range(len(group)):
                # 3-quarter offset for start date
                # Ensure i-3 is valid
                start_idx = i - 3
                if start_idx < 0: continue
                
                is_bull = (bull_trigs[i] == 1)
                is_bear = (bear_trigs[i] == 1)
                
                curr_q = int(q_ids[start_idx]) # The actual start quarter of the pattern
                
                if is_bull:
                    # Signal: We are in a Bullish Phase (validated by last 4 quarters)
                    
                    if state != 'bull':
                        # New Bullish Phase Detected
                        state = 'bull'
                        
                        # Ends previous cycle if exists
                        if current_cycle:
                            current_cycle['EndQuarter'] = curr_q
                            
                            # Determine Success: Market Cap Yield over Beary Period (Pick -> End)
                            pick_q = current_cycle['PickQuarter']
                            if pick_q is not None and pick_q in mc_map and curr_q in mc_map:
                                mc_start = mc_map[pick_q]
                                mc_end = mc_map[curr_q]
                                if mc_start > 0:
                                    # Yield = (End / Start) - 1. We check if Product > 0?
                                    # User: "take the market_cap yield over the beary period. if the product is positive, is succesfull"
                                    # Yield is positive if End > Start. "Product" usually implies multiplication, but yield is a % change.
                                    # Assuming "Positive Yield" (i.e., Cap Increased despite Bear Phase? Or simply survived?)
                                    yld = (mc_end / mc_start) - 1
                                    current_cycle['IsSuccesfull'] = 1 if yld > 0 else 0
                                else:
                                    current_cycle['IsSuccesfull'] = 0
                            
                            cycles.append(current_cycle)
                        
                        # Start New Cycle
                        current_cycle = {
                            'ISIN': group['ISIN'].iloc[0], # group.name is the grouping key (ISIN)
                            'CycleID': len(cycles) + 1, # relative ID, will fix globally later or keep per ISIN? Usually global PK. 
                            # Let's keep local ID or None and let DB handle? Schema has CycleID. 
                            # We can generate UUID or sequential later.
                            'StartedQuarter': curr_q,
                            'PickQuarter': None,
                            'EndQuarter': None,
                            'IsSuccesfull': None
                        }
                
                elif is_bear:
                    # Signal: We are in a Bearish Phase
                    
                    if state == 'bull':
                        # Transition from Bull -> Bear
                        state = 'bear'
                        if current_cycle:
                            current_cycle['PickQuarter'] = curr_q
                
                # If neither (stable/mixed), state persists (Latching behavior)

            # Close last cycle? 
            # "Cycle will end only when new bullish period is start"
            # So the last cycle remains open (EndQuarter=None)
            if current_cycle:
                cycles.append(current_cycle)
                
            return cycles

        # Apply logic
        # Group by ISIN
        cycles_lists = df.groupby('ISIN').apply(_analyze_isin_cycles)
        
        # Flatten list of lists
        all_cycles = [c for sublist in cycles_lists for c in sublist]
        return df, all_cycles

    def fetch_cycles(self, df_cycles: pd.DataFrame, df_micro: pd.DataFrame):
        """
        Builds cycle statistics for each cycle using apply instead of iterrows.
        Returns the stats DataFrame.
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        - market_cap at StartedQuarter, PickQuarter, EndQuarter
        - PE at StartedQuarter, PickQuarter, EndQuarter
        - BullYield
        - BearYield
        - TotalYield
        - PE_Expansion
        - PE_Contraction
        """
        print("Building cycle statistics...")
        
        if df_cycles is None or df_cycles.empty:
            return pd.DataFrame()

        # Prepare Micro Data for Lookup
        micro_lookup = df_micro.set_index(['ISIN', 'QuarterID'])

        # Define processing function for apply
        def _calc_row_stats(cycle_row):
            isin = cycle_row['ISIN']
            start_q = cycle_row['StartedQuarter']
            pick_q = cycle_row['PickQuarter']
            end_q = cycle_row['EndQuarter']

            # Helper to get value safetly
            def _get(q, col):
                try:
                    if pd.notna(q) and (isin, q) in micro_lookup.index:
                         return float(micro_lookup.loc[(isin, q)][col])
                except: pass
                return None

            mc_start = _get(start_q, 'market_cap')
            mc_pick = _get(pick_q, 'market_cap')
            mc_end = _get(end_q, 'market_cap')
            
            pe_start = _get(start_q, 'PE')
            pe_pick = _get(pick_q, 'PE')
            pe_end = _get(end_q, 'PE')

            bull_yield = ((mc_pick / mc_start) - 1) if (mc_start and mc_pick and mc_start > 0) else None
            bear_yield = ((mc_end / mc_pick) - 1) if (mc_pick and mc_end and mc_pick > 0) else None
            total_yield = ((mc_end / mc_start) - 1) if (mc_start and mc_end and mc_start > 0) else None
            
            pe_exp = ((pe_pick / pe_start) - 1) if (pe_start and pe_pick and pe_start > 0) else None
            pe_con = ((pe_end / pe_pick) - 1) if (pe_pick and pe_end and pe_pick > 0) else None
            
            return pd.Series({
                'CycleID': cycle_row['CycleID'],
                'BullYield': bull_yield,
                'BearYield': bear_yield,
                'TotalYield': total_yield,
                'PE_Expansion': pe_exp,
                'PE_Contraction': pe_con
            })

        # Use Apply
        stats_df = df_cycles.apply(_calc_row_stats, axis=1)
        return stats_df
    
    def calc_fundemental_data(self):
        print("Calculating fundemental calculated data...")
        # Step 1: Get raw fundemental micro data that was been loaded from EOD
        fundemental_micro_df = self.db.fetch_df('''
                                    SELECT ISO,Quarter,Year,Sector,Industrial,
                                    market_cap,
                                    net_profit,
                                    books_value,
                                    amount_of_shares,
                                    dividend + buyback as artificial_return,
                                    investment,
                                    net_profit - dividend - buyback - investment as cash_flow,
                                    market_cap / net_profit as PE,
                                    net_profit / books_value as ROE,
                                    market_cap / books_value as PB
                                    from companies_details
                                    left join companies
                                    on companies_details.ISIN=companies.ISIN
                                    left join stocks_raw
                                    on companies.ISIN=stocks_raw.ISIN
                                    left join quarters
                                    on stocks_raw.QuarterID=quarters.QuarterID
                                    ORDER BY ISO, ISIN, QuarterID
                                    ''')
        
        fundemental_micro_df = self.calc_growth(fundemental_micro_df)
        fundemental_micro_df = self.calc_years(fundemental_micro_df)
        fundemental_micro_df, all_cycles = self.calc_cycle(fundemental_micro_df)
        
        if all_cycles:
            df_cycles = pd.DataFrame(all_cycles)
            df_cycles['CycleID'] = range(1, len(df_cycles) + 1)
            cols = ['CycleID', 'ISIN', 'StartedQuarter', 'PickQuarter', 'EndQuarter', 'IsSuccesfull']
            for c in cols:
                if c not in df_cycles.columns: df_cycles[c] = None
            df_cycles = df_cycles[cols]
            print(f"Generated {len(df_cycles)} cycles.")
            
            # ------------------------------------------------------------------
            # Cycle Mapping to Stocks (to add CycleID to stocks_calculated)
            # ------------------------------------------------------------------
            # Create a lookup mapping: (ISIN, QuarterID) -> CycleID
            cycle_map = {}
            for _, row in df_cycles.iterrows():
                isin = row['ISIN']
                cid = row['CycleID']
                sq = row['StartedQuarter']
                eq = row['EndQuarter']
                
                # We need all quarters between Started and End (inclusive)
                # Since QuarterID is sequential integer (e.g. 20231, 20232), we can iterate? 
                # No, standard QuarterIDs like 20234 -> 20241 have gaps (6).
                # We need to rely on the 'quarters' table ideally, but here we can just map the specific points or use range if numeric is consistent-ish (it breaks at year boundary).
                # Better approach: Filter the DF for this ISIN and assign CycleID where QuarterID in range.
                pass
            
            # Vectorized Cycle Assignment:
            # We must map CycleID back to fundemental_micro_df
            # Since an ISIN can have multiple cycles, but they shouldn't overlap in time (per logic),
            # we can join or apply.
            # Efficient way:
            fundemental_micro_df['CycleID'] = None
            
            # Loop over cycles (slow but safe for range logic) or use interval index?
            # Or since we processed by ISIN, we can maintain the relationship.
            # Let's use the fact that cycles are non-overlapping.
            # We can create a mapping DataFrame "IsinQuarterCycle" and merge.
            
            cycle_mapping_list = []
            
            # Get all quarters to know the sequence
            all_q = sorted(fundemental_micro_df['QuarterID'].unique())
            
            for _, row in df_cycles.iterrows():
                isin = row['ISIN']
                cid = row['CycleID']
                s = row['StartedQuarter']
                e = row['EndQuarter']
                
                if pd.isna(e): e = 999999 # Active cycle
                
                # This logic assumes simple integer comparison works for QuarterID
                # 20234 < 20241 is True. So yes.
                cycle_mapping_list.append({
                    'ISIN': isin,
                    'Start': s,
                    'End': e,
                    'CycleID': cid
                })
            
            # This is still O(N_cycles * N_rows) if not careful.
            # Optimization: merge on ISIN, then filter.
            if cycle_mapping_list:
                map_df = pd.DataFrame(cycle_mapping_list)
                # Map to micro df
                # Inner join on ISIN, then filter rows where Start <= QuarterID <= End
                merged_map = pd.merge(fundemental_micro_df[['ISIN', 'QuarterID']].reset_index(drop=True), map_df, on='ISIN', how='left')
                
                # Boolean mask
                mask = (merged_map['QuarterID'] >= merged_map['Start']) & (merged_map['QuarterID'] <= merged_map['End'])
                
                # Get valid matches
                valid_mappings = merged_map[mask][['ISIN', 'QuarterID', 'CycleID']].drop_duplicates(subset=['ISIN', 'QuarterID'])
                
                # Merge back to main df
                fundemental_micro_df = pd.merge(fundemental_micro_df, valid_mappings, on=['ISIN', 'QuarterID'], how='left')
                
                # Fix duplicates from merge? (CycleID_x, CycleID_y if existed)
                if 'CycleID_x' in fundemental_micro_df.columns:
                     fundemental_micro_df['CycleID'] = fundemental_micro_df['CycleID_y'].fillna(fundemental_micro_df['CycleID_x'])
                     fundemental_micro_df.drop(columns=['CycleID_x', 'CycleID_y'], inplace=True)
            
            # Calculate Cycle Stats
            stats_df = self.fetch_cycles(df_cycles, fundemental_micro_df)
            
            # Merge and Save to 'cycles' table
            if not stats_df.empty:
                full_cycles_df = pd.merge(df_cycles, stats_df, on='CycleID', how='left')
                full_cycles_df.to_sql('cycles', self.conn, if_exists='replace', index=False)
                print(f"Saved {len(full_cycles_df)} cycles to 'cycles' table.")
            else:
                 df_cycles.to_sql('cycles', self.conn, if_exists='replace', index=False)
        else:
            print("No cycles detected.")
            self.cursor.execute("DELETE FROM cycles")
            self.conn.commit()
            df_cycles = pd.DataFrame() # Empty for next step
            fundemental_micro_df['CycleID'] = None


        # Save Calculated Metrics to stocks_calculated
        # Filter relevant columns for storage
        print("Saving Calculated Metrics...")
        
        # User requested fields from: calc_growth, calc_year, cycle_id (Done)
        
        # Fields:
        # stock_price_yield, profit_growth_expection, profit_growth, growth_model_error
        # premium_years_nonGrowth, premium_years_Growth, premium
        # CycleID
        
        save_cols = [
            'ISIN', 'QuarterID', 'net_profit', 'PE', 'MarginalPE', 'PB', 'ROE', 
            'MarginalProfit', 'Premium', 'ExpectedGrowth', 
            'Premium_profit_nonGrowth', 'Premium_profit_Growth',
            'stock_price_yield', 'profit_growth_expection', 'profit_growth', 'growth_model_error',
            'CycleID'
        ]
        
        # Rename for schema match if needed
        # Columns in df are lowercase from methods (premium, expected_growth, etc)
        rename_map = {
            'premium': 'Premium',
            'expected_growth': 'ExpectedGrowth',
            'premium_years_nonGrowth': 'Premium_profit_nonGrowth',
            'premium_years_Growth': 'Premium_profit_Growth'
        }
        
        final_df = fundemental_micro_df.copy()
        final_df.rename(columns=rename_map, inplace=True)
        
        # Ensure 'profit_growth' exists (calc_growth added it?)
        if 'profit_growth' not in final_df.columns:
             # calc_growth usually adds it based on logic or user request.
             # Let's check calc_growth implementation:
             # df['profit_growth_expection'] = ...
             # df['growth_model_error'] = ...
             # It didn't explicitly create 'profit_growth' in previous read, but used it in error calc?
             # Re-checking calc_growth logic...
             # Ah, "df['growth_model_error'] = df['expected_growth'] - df['profit_growth_expection']" in old code.
             # The new instruction says "profit_growth" is output.
             # I should update calc_growth to produce it if missing, or use net_profit raw growth.
             pass 

        # Filter only existing
        valid_cols = [c for c in save_cols if c in final_df.columns]
        
        final_df = final_df[valid_cols]
        final_df.to_sql('stocks_calculated', self.conn, if_exists='replace', index=False)
        print(f"Saved {len(final_df)} rows to stocks_calculated.")
        
        print("Calculations Complete.")
    
    def get_monitaric_data(self,aggregated_micro_df : pd.DataFrame):
        """
        Given the forigen cash stream to the exchanges,
        the assumption that 30 years back related to now the total_cap is 0
        and given the annualy total_cap growth,
        calculate the annualy volume of forign holdings from the exchange cap
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        - precent_foreign_holdings
        - precent_local_holdings
        - precent_world_in_country
        - precent_local_investment_on_foreign_from_global
        - fluidity_growth_1y
        - fluidity_growth_5y
        - central_bank_rate_growth_1y
        -----------------------------------------------------------------------
        """
        # get the anually cash stream and the M3/4 data from macro
        macro_data = self.db.fetch_df('''
                                    SELECT ISO,Year,foreign_capital_inflow_usd, money_supply_m4
                                    ''')
        # get the exchange total cap from aggregated micro
        exchange_cap = aggregated_micro_df[['ISO','Year','total_market_cap','exchange_yield']]
        # merge the dataframes
        merged = pd.merge(macro_data, exchange_cap, on=['ISO','Year'], how='left')
        # calculate the cumelative product growth of the exchange cap (what yield the stream got from each year until the last)
        merged['cum_exchange_cap'] = merged.groupby('ISO')['exchange_yield'].apply(lambda x: (1 + x).cumprod())
        # calculate the forigen stream yield
        merged['stream_yield'] = merged['foreign_capital_inflow_usd'] * merged['cum_exchange_cap']
        # sum the cumelative stream yield into total forigen holdings
        merged['total_foreign_holdings'] = merged.groupby('ISO')['stream_yield'].cumsum()
        # then calc the local holdings sum
        merged['total_local_holdings'] = merged['total_market_cap'] - merged['total_foreign_holdings']
        # calc the local investment on foreign holdings
        merged['local_investment_on_foreign'] = merged['money_supply_m4'] - merged['total_local_holdings']
        # then calc the precents of rate and local investment from market cap
        merged['percent_foreign_holdings'] = merged['total_foreign_holdings'] / merged['total_market_cap']
        merged['percent_local_holdings'] = merged['total_local_holdings'] / merged['total_market_cap']
        
        # Bring back the aggregated Micro Data Columns that were dropped by the initial merge strategy if desired.
        # However, relying on the 'merged' dataframe which is focused on monitaric data. 
        # The user requested output columns of get_aggregated_micro. 
        # Since 'aggregated_micro_df' has them, and we merge macro_data (left) with exchange_cap (subset of micro), we lost them.
        # Let's fix the merge to include the unstacked micro data again.
        
        # Re-merge with full aggregated_micro_df (Outer or Left on ISO/Year)
        # We need to be careful about column overlap. exchange_yield is in both.
        
        full_merged = pd.merge(merged, aggregated_micro_df.drop(columns=['total_market_cap', 'exchange_yield'], errors='ignore'), on=['ISO','Year'], how='left')
        
        # and the precents of world calc in country and precent of local investment on foreign from global foreign investment
        full_merged['percent_world_in_country'] = full_merged['total_foreign_holdings'] / full_merged.groupby('Year')['total_foreign_holdings'].transform('sum')
        full_merged['percent_local_investment_on_foreign_from_global'] = full_merged['local_investment_on_foreign'] / full_merged.groupby('Year')['total_foreign_holdings'].transform('sum')
        # finally calc the fluidity growth for 1 and 5 years, and the interest rate growth last year
        full_merged['fluidity_growth_1y'] = full_merged['money_supply_m4'] / full_merged.groupby('ISO')['money_supply_m4'].shift(1) - 1
        full_merged['fluidity_growth_5y'] = full_merged['money_supply_m4'] / full_merged.groupby('ISO')['money_supply_m4'].shift(5) - 1
        full_merged['central_bank_rate_growth_1y'] = full_merged['central_bank_rate'] / full_merged.groupby('ISO')['central_bank_rate'].shift(1) - 1
        # return the table
        return full_merged
    
    def calc_fiscal_data(self,merged_monitaric_table : pd.DataFrame):
        """
        Docstring for calc_fiscal_data
        
        :param self: Description
        :param merged_monitaric_table: Description
        :type merged_monitaric_table: pd.DataFrame
        -----------------------------------------------------------------------
        The output table will contain the following columns:
        - gov_bond_risk_rate
        - revenue_debt_rate
        - military_revenue_rate
        """
        # get the fiscal debt data from macro_raw
        fiscal_data = self.db.fetch_df('''
                                    SELECT ISO,Year, gov_bond_yeald_10y, gov_bond_yeald_3y, gov_debt_total, gov_revenue, military_expenditure
                                    ''')
        # merge the dataframes
        merged = pd.merge(merged_monitaric_table, fiscal_data, on=['ISO','Year'], how='left')
        # first we will calc rate between yields of gov bonds 3y and 10y
        merged['gov_bond_risk_rate'] = merged['gov_bond_yeald_10y'] - merged['gov_bond_yeald_3y']
        # then we will calc revenue / debt rate
        merged['revenue_debt_rate'] = merged['gov_revenue'] / (merged['gov_debt_total'] + 1e-6)
        # then we will calc military expenditure from revenue rate
        merged['military_revenue_rate'] = merged['military_expenditure'] / (merged['gov_revenue'] + 1e-6)
        return merged
        
    def calc_macro_data(self):
        print("Calculating macro calculated data...")
        # Step 1: Get Aggregated Micro Data
        aggregated_micro_df = self.get_aggreated_micro()
        
        # Step 2: Monitaric Data Calculations
        monitaric_df = self.get_monitaric_data(aggregated_micro_df)
        
        # Step 3: Fiscal Data Calculations
        fiscal_df = self.calc_fiscal_data(monitaric_df)
        
        # Save to macro_calculated table
        fiscal_df.to_sql('macro_calculated', self.conn, if_exists='replace', index=False)
        print("Calculated macro_calculated table.")

    def get_currencies_map(self):
        # Fetch currency mapping from DB
        df = self.db.fetch_df('SELECT currencyID, QuarterID,Year, RateToUSD FROM currency_rates left join quarters on currency_rates.QuarterID=quarters.QuarterID')
        currency_map = df.set_index(['currencyID', 'QuarterID'])['RateToUSD'].unstack()
        currency_map_years = df.groupby(['currencyID','Year'])['RateToUSD'].last().unstack()
        return currency_map, currency_map_years

    def create_tables(self):
        # Stocks Calculated
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS stocks_calculated (
            ISIN TEXT, 
            QuarterID INTEGER, 
            net_profit REAL, 
            PE REAL, 
            MarginalPE REAL, 
            PB REAL, 
            ROE REAL, 
            MarginalProfit REAL, 
            Premium REAL, 
            ExpectedGrowth REAL, 
            Premium_profit_nonGrowth REAL, 
            Premium_profit_Growth REAL, 
            PRIMARY KEY (ISIN, QuarterID))''')
            
        # Stocks Cycle
        # MERGED into 'cycles'
            
        # Cycles (Merged stocks_cycle + cycle_calculated)
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS cycles (
            CycleID INTEGER,
            ISIN TEXT,
            StartedQuarter INTEGER,
            PickQuarter INTEGER,
            EndQuarter INTEGER,
            IsSuccesfull INTEGER,
            BullYield REAL,
            BearYield REAL,
            TotalYield REAL,
            PE_Expansion REAL,
            PE_Contraction REAL,
            PRIMARY KEY (CycleID))''')

        # Macro Calculated
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS macro_calculated (
            ISO TEXT,
            Year INTEGER,
            foreign_capital_inflow_usd REAL,
            money_supply_m4 REAL,
            total_market_cap REAL,
            exchange_yield REAL,
            cum_exchange_cap REAL,
            stream_yield REAL,
            total_foreign_holdings REAL,
            total_local_holdings REAL,
            local_investment_on_foreign REAL,
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
            gov_debt_total REAL,
            gov_revenue REAL,
            military_expenditure REAL,
            gov_bond_risk_rate REAL, 
            revenue_debt_rate REAL,
            military_revenue_rate REAL,
            PRIMARY KEY (ISO, Year))''')
        
        self.conn.commit()
        print("Scheme 5 Tables initialized.")

    def check_integrity(self):
        for table in ['stocks_calculated', 'cycles', 'macro_calculated']:
            try:
                count = self.cursor.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                print(f"{table}: {count} rows")
            except Exception as e:
                print(f"Error checking {table}: {e}")
    

def run():
    loader = CalculatedLoader()
    print("--- Pipeline 5: Calculated Data ---")
    loader.create_tables()
    loader.calc_fundemental_data()
    loader.calc_macro_data()
    loader.check_integrity()
    print("Pipeline 5 Complete.")



    

        
        


