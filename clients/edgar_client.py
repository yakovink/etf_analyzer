
from global_import import requests, pd, io, np, zipfile

def get_nport_dataframes(year: int, quarter: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Downloads N-PORT zip, extracts specific TSV files into memory,
    and returns them as a tuple of pandas DataFrames.
    """
    headers = {
        "User-Agent": "YourName YourEmail@domain.com",  # חובה לפי דרישות ה-SEC
        "Accept-Encoding": "gzip, deflate"
    }
    
    url = f"https://www.sec.gov/files/dera/data/form-n-port-data-sets/{year}q{quarter}_nport.zip"
    
    print(f"Downloading data for {year} Q{quarter}...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # שימוש ב-BytesIO כדי להתייחס לתוכן שהורד כקובץ בזיכרון
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # הגדרת הקבצים שברצוננו לשלוף
        target_files = {
            "FUND_REPORTED_INFO.tsv":["ACCESSION_NUMBER","SERIES_ID","NET_ASSETS"],
              "FUND_REPORTED_HOLDING.tsv":['ACCESSION_NUMBER', 'HOLDING_ID','ISSUER_LEI','ISSUER_CUSIP','ASSET_CAT','INVESTMENT_COUNTRY','PERCENTAGE'],
                "IDENTIFIERS.tsv":["HOLDING_ID","IDENTIFIER_ISIN",'OTHER_IDENTIFIER', 'OTHER_IDENTIFIER_DESC']
                }

        dfs = {}
        
        for file_name, columns in target_files.items():
            if file_name in z.namelist():
                print(f"Reading {file_name} into DataFrame...")
                with z.open(file_name) as f:
                    # low_memory=False מונע אזהרות לגבי סוגי נתונים מעורבים בעמודות
                    dfs[file_name] = pd.read_csv(f, sep='\t', low_memory=False)
                    print(dfs[file_name].columns,"picked columns: ",columns)
                    dfs[file_name] = dfs[file_name][columns]
            else:
                raise FileNotFoundError(f"{file_name} not found in the zip archive.")
        return dfs["FUND_REPORTED_HOLDING.tsv"], dfs["FUND_REPORTED_INFO.tsv"], dfs["IDENTIFIERS.tsv"]

def get_updated_lei_mapping() -> pd.DataFrame:
    """
    Downloads the latest LEI to Company Name mapping from GLEIF.
    Returns a DataFrame with 'LEI' and 'Entity Legal Name' columns.
    """
    table = pd.read_csv("data/funds_data/lei-isin.csv", dtype=str)
    table.drop_duplicates(subset=['LEI'], keep='first', inplace=True)
    return table

def get_fund_table(year: int, quarter: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    lei_table = get_updated_lei_mapping()
    df_holdings, df_info, df_identifiers = get_nport_dataframes(year, quarter)
    df_etf_tickers = get_funds_tickers(year)

    # Clean data - drop rows with critical missing values
    df_holdings.dropna(subset = ['HOLDING_ID'], inplace=True)
    df_info.dropna(subset = ['ACCESSION_NUMBER','SERIES_ID'], inplace=True)
    df_identifiers.dropna(subset = ['HOLDING_ID'], inplace=True)
    lei_table.dropna(inplace=True)
    df_etf_tickers.dropna(subset = ['Series ID'], inplace=True)



    # for identifiers, keep first occurrence of each HOLDING_ID
    df_identifiers = df_identifiers.groupby(['HOLDING_ID']).agg({
        "IDENTIFIER_ISIN": 'first',
        "OTHER_IDENTIFIER": 'first',
        "OTHER_IDENTIFIER_DESC": 'first'
    }).reset_index()
    # for etf tickers, keep shortest ticker for each Series ID
    df_etf_tickers['Ticker Length'] = df_etf_tickers['Class Ticker'].str.len()
    df_etf_tickers = df_etf_tickers.sort_values(by = 'Ticker Length').drop_duplicates(subset=['Series ID'], keep='first')


    #first merge holdings with info on ACCESSION_NUMBER - get holding list for reported funds
    merged = pd.merge(df_info, df_holdings, on='ACCESSION_NUMBER', how='left')
    # then merge with identifiers on HOLDING_ID to get ISINs, LEIs and CUSIPs
    print(merged.loc[merged['SERIES_ID']=='S000004354'].shape," before merging identifiers")
    merged = merged.merge(df_identifiers, on="HOLDING_ID", how="left")
    # then merge with lei_table on ISSUER_LEI to get issuer names
    print(merged.loc[merged['SERIES_ID']=='S000004354'].shape," before merging lei_table")
    merged = merged.merge(lei_table, left_on="ISSUER_LEI", right_on="LEI", how="left")
    # finally merge with etf tickers on SERIES_ID to get fund tickers and names
    print(merged.loc[merged['SERIES_ID']=='S000004354'].shape," before merging etf tickers")
    merged = merged.merge(df_etf_tickers, left_on="SERIES_ID", right_on="Series ID", how="left")

    have_original_cusip = (~merged['ISSUER_CUSIP'].isnull())&(merged['INVESTMENT_COUNTRY'].isin(['US','CA']))
    cusip_nicknames = [
    "CG Symbol",
    "CGSymbol",
    "INTERNAL CUSIP",
    "CINS",
    "CUSIP"
    ]
    have_side_cusip = (merged['OTHER_IDENTIFIER_DESC'].isin(cusip_nicknames)&(merged['INVESTMENT_COUNTRY'].isin(['US','CA'])))
    merged.loc[have_side_cusip,'CUSIP'] = merged.loc[have_side_cusip,'OTHER_IDENTIFIER']
    merged.loc[have_original_cusip,'CUSIP'] = merged.loc[have_original_cusip,'ISSUER_CUSIP']
    merged.loc[have_side_cusip|have_original_cusip,'ISIN_by_cusip'] = merged.loc[have_side_cusip|have_original_cusip,["INVESTMENT_COUNTRY","CUSIP"]].astype(str).sum(axis=1)
    merged['FinalISIN'] = merged['IDENTIFIER_ISIN'].fillna(merged['ISIN_by_cusip']).fillna(merged['ISIN'])
    


    

    print(merged.loc[merged['SERIES_ID']=='S000004354'].shape," after all merging")
    final = merged.loc[
        (~merged['FinalISIN'].isnull())&(merged['ASSET_CAT']=='EC')&(~merged['FinalISIN'].isin([None,np.nan])),
        ["Entity Name","Class Name","Class Ticker",
         "NET_ASSETS",'INVESTMENT_COUNTRY','PERCENTAGE',
         "FinalISIN","SERIES_ID","HOLDING_ID"]
         ].copy()
    # keep only last occurrence of each FinalISIN per SERIES_ID (latest ACCESSION_NUMBER)
    print(final.loc[final['SERIES_ID']=='S000004354'].shape," before dropping duplicates")
    merged.sort_values(by="ACCESSION_NUMBER", inplace=True)
    final.drop_duplicates(subset=['FinalISIN','SERIES_ID'], keep='last', inplace=True)
    print(final.loc[final['SERIES_ID']=='S000004354'].shape," final shape")
    return final
    



def get_funds_tickers(year: int) -> pd.DataFrame:
    """
    Download SEC file mapping SERIES_ID to TICKER and NAME.
    """
    url = f"https://www.sec.gov/files/investment/data/other/investment-company-series-class-information/investment-company-series-class-{year}.csv"
    headers = {
        "User-Agent": "YourName YourEmail@domain.com",  # חובה לפי דרישות ה-SEC
        "Accept-Encoding": "gzip, deflate"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tickers = pd.read_csv(io.StringIO(response.text))
    return tickers[["Entity Name","Series ID","Class Name","Class Ticker"]]
                
