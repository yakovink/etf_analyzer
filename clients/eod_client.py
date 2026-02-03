

from global_import import requests, Optional, Dict, Any, pd, eodhd, time, logging, ssl, List
from clients.database_manager import DatabaseManager
class EODClient:

    def get_exchange_stocks_list(self, exchange_code: str) -> Optional[pd.DataFrame]:
        """
        Fetches the list of traded stocks for a given exchange code.
        Returns a DataFrame with columns: ticker, exchange, ISIN.
        """

        while True:
            try:
                response = self.api.get_list_of_tickers(exchange_code)
                data_df = pd.DataFrame(response)
                return data_df
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_exchange_stocks_list for {exchange_code}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching stocks for {exchange_code}: {e}")
                return None
        
    def get_exchanges_list(self) -> Optional[pd.DataFrame]:
        """
        Fetches the list of supported exchanges.
        Returns a DataFrame with exchange details.
        """
        while True:
            try:
                response = self.api.get_exchanges()
                data_df = pd.DataFrame(response)
                return data_df
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_exchanges_list: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching exchanges list: {e}")
                return None
    
    def get_currencies_list(self) -> Optional[pd.DataFrame]:
        """
        Fetches the list of supported currencies.
        Returns a DataFrame with currency details.
        """
        while True:
            try:
                response = self.api.get_exchange_symbols("FOREX")
                data_df = pd.DataFrame(response)
                return data_df
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_currencies_list: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching currencies list: {e}")
                return None

    def get_currencies_historical_rates(self, currency_code: str, start_date: str, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetches historical exchange rates for a given currency code between start_date and end_date.
        Returns a DataFrame with historical rates.
        """
        if end_date is None:
            end_date = ''
        while True:
            try:
                currency_ticker = f'{currency_code}.FOREX'
                return self.api.get_historical_data(currency_ticker,interval='m', iso8601_start=start_date, iso8601_end=end_date,results=400).reset_index()
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_currencies_historical_rates for {currency_code}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching historical rates for {currency_code}: {e}")
                return None
        
    def get_etf_list(self) -> Optional[pd.DataFrame]:
        """
        Fetches the list of ETFs.
        Returns a DataFrame with ETF details.
        """
        while True:
            try:
                response: pd.DataFrame = self.api.get_exchange_symbols("US")
                if response is None:
                    return pd.DataFrame()
                response = response[response['type'] == 'ETF']
                response['Manager'] = response['name'].str.split(' ').str[0]
                data_df = response[['symbol', 'name', 'Manager']].rename(columns={'symbol': 'Ticker', 'name': 'Name'})
                return data_df
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_etf_list: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching ETF list: {e}")
                return None

    def __init__(self):
        self.api_token = 'API'
        self.base_url = 'https://eodhd.com/api'
        self.api = eodhd.APIClient(self.api_token)
        self.db = DatabaseManager()

    def get_macro_indicator(self, country_code: str, indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fetches specific macro indicators for a country.
        If indicators is None, fetches a default broad set.
        """
        if indicators is None:
            wanted_indicators = [
                'real_interest_rate',
                'population_total',
                'population_growth_annual',
                'inflation_consumer_prices_annual',
                'consumer_price_index',
                'gdp_current_usd',
                'gdp_per_capita_usd',
                'gdp_growth_annual',
                'debt_percent_gdp',
                'net_trades_goods_services',
                'inflation_gdp_deflator_annual',
                'agriculture_value_added_percent_gdp',
                'industry_value_added_percent_gdp',
                'services_value_added_percent_gdp',
                'exports_of_goods_services_percent_gdp',
                'imports_of_goods_services_percent_gdp',
                'gross_capital_formation_percent_gdp',
                'net_migration',
                'gni_usd',
                'gni_per_capita_usd',
                'gni_ppp_usd',
                'gni_per_capita_ppp_usd',
                'income_share_lowest_twenty',
                'life_expectancy',
                'fertility_rate',
                'prevalence_hiv_total',
                'co2_emissions_tons_per_capita',
                'surface_area_km',
                'poverty_poverty_lines_percent_population',
                'revenue_excluding_grants_percent_gdp',
                'cash_surplus_deficit_percent_gdp',
                'startup_procedures_register',
                'market_cap_domestic_companies_percent_gdp',
                'mobile_subscriptions_per_hundred',
                'internet_users_per_hundred',
                'high_technology_exports_percent_total',
                'merchandise_trade_percent_gdp',
                'total_debt_service_percent_gni',
                'unemployment_total_percent'
            ]
        else:
            wanted_indicators = indicators

        quarters_map = self.db.fetch_df("SELECT QuarterID, Year, Quarter FROM quarters")

        country_indicators = self.api.get_macro_indicators_data(country_code, wanted_indicators)
        if not country_indicators or country_indicators.empty:
            return pd.DataFrame()
        country_indicators = pd.DataFrame(country_indicators)

        # select only quarter end dates
        country_indicators['Date'] = pd.to_datetime(country_indicators['Date'], errors='coerce').dt.date
        country_indicators['Month'] = country_indicators['Date'].dt.month
        country_indicators['Year'] = country_indicators['Date'].dt.year
        country_indicators = country_indicators.merge(quarters_map, on=['Year'], how='left')
        country_indicators = country_indicators.sort_values(by=['Date'], ascending=False)
        country_indicators.drop_duplicates(subset = ['QuarterID','Indicator'], keep='last', inplace=True)

        # unstack to have one row per date with all indicators as columns
        country_indicators = country_indicators.pivot(index=['Date','Year','Quarter','QuarterID'], columns='Indicator', values='Value').reset_index()
        return country_indicators

    def get_ticker_historical(self, ticker: str, exchange: str = 'US', start_date: str = '1995-01-01') -> pd.DataFrame:
        """
        Fetches historical data for any ticker/exchange.
        """
        full_ticker = f"{ticker}.{exchange}" if exchange and '.' not in ticker else ticker
        try:
             # EODHD API uses 'period' or 'interval'. 'd' is default.
             # We want annual average likely? Pipeline can aggregate.
             data = self.api.get_historical_data(full_ticker, iso8601_start=start_date)
             if isinstance(data, list):
                 return pd.DataFrame(data)
             return pd.DataFrame(data)
        except Exception as e:
            # print(f"EODClient Error {ticker}: {e}")
            return pd.DataFrame()

    def get_fundamentals(self, ticker: str, exchange: str = 'US') -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a single specific ticker.
        Endpoint: /fundamentals/{Ticker}.{Exchange}
        """
        symbol = f"{ticker}.{exchange}"
        url = f"{self.base_url}/fundamentals/{symbol}?api_token={self.api_token}&fmt=json"
        
        while True:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"EODClient: Failed to fetch fundamentals for {symbol}. Status: {response.status_code}")
                    return None
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_fundamentals for {symbol}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching fundamentals for {symbol}: {e}")
                return None

    def search_instrument(self, query: str) -> Optional[list]:
        """
        Search for an instrument (Ticker, ISIN, SEDOL, etc.)
        Endpoint: /search/{query_string}
        """
        url = f"{self.base_url}/search/{query}?api_token={self.api_token}&fmt=json"
        
        while True:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"EODClient: Search failed for {query}. Status: {response.status_code}")
                    return []
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in search_instrument for {query}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error searching for {query}: {e}")
                return []

    def get_live_price(self, ticker: str, exchange: str = 'US') -> Optional[float]:
        """
        Get live (delayed) price for a single ticker.
        Endpoint: /real-time/{Ticker}.{Exchange}
        """
        symbol = f"{ticker}.{exchange}"
        url = f"{self.base_url}/real-time/{symbol}?api_token={self.api_token}&fmt=json"
        
        while True:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('close')
                else:
                    return self._get_eod_price(ticker, exchange)
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_live_price for {symbol}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching price for {symbol}: {e}")
                return None

    def _get_eod_price(self, ticker: str, exchange: str) -> Optional[float]:
        """
        Fallback: Get end-of-day price from /eod endpoint.
        """
        symbol = f"{ticker}.{exchange}"
        url = f"{self.base_url}/eod/{symbol}?api_token={self.api_token}&fmt=json&limit=1"
        try:
             response = requests.get(url, timeout=5)
             if response.status_code == 200:
                 data = response.json()
                 if isinstance(data, list) and len(data) > 0:
                     return data[0].get('close')
        except:
            pass
        return None

    def get_historical_market_cap(self, ticker: str, exchange: str = 'US') -> Optional[Any]:
        """
        Get historical market capitalization data.
        """
        symbol = f"{ticker}.{exchange}"
        url = f"{self.base_url}/historical-market-cap/{symbol}?api_token={self.api_token}&fmt=json"
        
        while True:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.json()
                return None
            except ssl.SSLError as ssl_err:
                logging.error(f"SSL error in get_historical_market_cap for {symbol}: {ssl_err}")
                time.sleep(10)
            except Exception as e:
                print(f"EODClient: Error fetching historical MC for {symbol}: {e}")
                return None

    def get_full_history_quarterly(self, ticker: str, exchange: str = 'US') -> Optional[list]:
        """
        Fetches >30 years of Quarterly History merging Net Profit, Market Cap, and Book Value.
        Returns a list of dicts: [{'date': '...', 'net_profit': ..., 'market_cap': ..., 'book_value': ...}, ...]
        """
        # 1. Fundamentals for Profit & Book Value
        funds = self.get_fundamentals(ticker, exchange)
        if not funds: return None
        
        financials = funds.get('Financials', {})
        income_q = financials.get('Income_Statement', {}).get('quarterly', {})
        balance_q = financials.get('Balance_Sheet', {}).get('quarterly', {})
        
        if not income_q: return None

        # 2. Historical Market Cap
        hist_mc_data = self.get_historical_market_cap(ticker, exchange) 
        
        mc_lookup = {}
        if hist_mc_data and isinstance(hist_mc_data, dict):
            mc_lookup = hist_mc_data
        
        # Merge
        result = []
        sorted_dates = sorted(income_q.keys(), reverse=True)
        
        for date_str in sorted_dates:
            # Income
            data = income_q[date_str]
            np_val = data.get('netIncome')
            if np_val is None: continue
            
            # Market Cap
            mc_val = mc_lookup.get(date_str)
            
            # Book Value
            bv_val = None
            if balance_q and date_str in balance_q:
                bv_item = balance_q[date_str]
                # 'totalStockholderEquity' is preferred
                bv_val = bv_item.get('totalStockholderEquity') or bv_item.get('netTangibleAssets')
            
            result.append({
                "date": date_str,
                "net_profit": float(np_val),
                "market_cap": float(mc_val) if mc_val is not None else None,
                "book_value": float(bv_val) if bv_val is not None else None
            })
            
        return result

    def get_stock_analysis_data(self, ticker: str, exchange: str = 'US') -> Dict[str, Any]:
        """
        Retrieves specific micro-analysis data points for a stock.
        - Historical Quarterly Net Profit
        - Market Cap / Price
        - Shares Amount
        - Book Value
        - Growth Expectation (Next Year)
        """
        # Fetch full fundamentals
        data = self.get_fundamentals(ticker, exchange)
        result = {
            'quarterly_net_profit': [],
            'market_cap': None,
            'price': None,
            'shares': None,
            'book_value': None,
            'growth_expectation': None
        }
        
        if not data:
            return result
            
        try:
            # 1. Historical Quarterly Net Profit
            # Financials -> Income_Statement -> quarterly
            financials = data.get('Financials', {})
            income_stmt = financials.get('Income_Statement', {})
            quarterly = income_stmt.get('quarterly', {})
            
            # Convert dict {date: val} to list of tuples or dicts
            net_profits = []
            for date_str, info in quarterly.items():
                np = info.get('netIncome')
                # Ensure it's not None/str 'NA'
                if np and str(np).replace('.','',1).isdigit() or (isinstance(np, (int, float))):
                     try:
                         net_profits.append({'date': date_str, 'value': float(np)})
                     except: pass
            
            # Sort by date
            result['quarterly_net_profit'] = sorted(net_profits, key=lambda x: x['date'], reverse=True)

            # 2. Market Cap / Price / Shares
            highlights = data.get('Highlights', {})
            shares_stats = data.get('SharesStats', {})
            
            result['market_cap'] = highlights.get('MarketCapitalization')
            result['shares'] = shares_stats.get('SharesOutstanding')
            
            # Price: Often in RealTime or calculated
            if result['market_cap'] and result['shares']:
                 try:
                     result['price'] = float(result['market_cap']) / float(result['shares'])
                 except: pass

            # 3. Book Value
            # Balance_Sheet -> quarterly -> latest -> totalStockholderEquity
            balance_sheet = financials.get('Balance_Sheet', {})
            bs_quarterly = balance_sheet.get('quarterly', {})
            
            # Get latest date
            if bs_quarterly:
                latest_bs_date = max(bs_quarterly.keys())
                latest_bs = bs_quarterly[latest_bs_date]
                # 'totalStockholderEquity' is standard Book Value
                result['book_value'] = latest_bs.get('totalStockholderEquity') or latest_bs.get('netTangibleAssets')

            # 4. Growth Expectation (Next Year)
            # Earnings -> Trend
            earnings = data.get('Earnings', {})
            trend = earnings.get('Trend', {})
            
            if trend:
                # Get latest entry
                latest_k = max(trend.keys())
                latest_t = trend[latest_k]
                result['growth_expectation'] = latest_t.get('growth') 

        except Exception as e:
            print(f"EODClient: Error parsing analysis data for {ticker}: {e}")
            
        return result



