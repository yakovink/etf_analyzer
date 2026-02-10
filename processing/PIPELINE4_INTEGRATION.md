# Pipeline 4: Multi-Source Data Integration

## Overview
Pipeline 4 has been updated to fetch macro indicators from **4 data sources** in a prioritized waterfall:

1. **World Bank** (primary source for most indicators)
2. **OECD SDMX 3.0 API** (sector financial accounts: S1/S2 domestic/foreign equity)
3. **EOD Macro Indicators** (GNI, gross savings)
4. **FRED** (patches for specific country-indicator combinations)
5. **EOD Ticker-based** (fallback for missing data using tickers/futures)

## Changes Implemented

### 1. New OECDClient (`clients/oecd_client.py`)
- **Purpose**: Dedicated client for OECD SDMX 3.0 API
- **Key Methods**:
  - `fetch_sdmx_data(dataflow, key, start_period, end_period)` - Makes HTTPS requests to OECD SDMX endpoint
  - `_parse_sdmx_json(data)` - Parses complex SDMX-JSON structure with dimensions/observations
  - `fetch_indicator(params, start_year, end_year)` - Wrapper for config-driven fetching

- **OECD SDMX Format**:
  ```python
  {
    "dataflow": "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.0",  # Annual Sector Accounts
    "key": "A....S1...A..F511+F521..USD......"               # Dimension key pattern
  }
  ```

- **SSL Handling**: Uses `verify=False` due to OECD SSL certificate issues

### 2. Pipeline 4 Updates (`processing/pipeline4_raw.py`)

#### New Methods:

**`get_oecd_data(countries_df)`**
- Fetches OECD indicators using `oecd_sdmx_params` from config
- Converts ISO3 → ISO2 for consistency with other sources
- Returns DataFrame with columns: `[ISO, Year, s1_domestic_equity_usd, s2_foreign_equity_usd, ...]`

**`fetch_eod_macro_data(countries_df)`**
- Fetches EOD macro indicators using `eod_macro_code` from config
- Aggregates quarterly data to annual (using mean)
- Returns DataFrame with columns: `[ISO, Year, national_income_usd, national_savings_usd, ...]`

#### Updated Methods:

**`process_initial_data(countries_df)`**
- **Before**: Only merged World Bank data
- **After**: Merges WB → OECD → EOD macro in sequence
- **Merge Strategy**:
  - Create skeleton (all ISO×Year combinations)
  - Merge WB data (primary source)
  - Merge OECD data (overwrites WB if both exist - OECD is more specific for S1/S2)
  - Merge EOD macro data (fills gaps, preserves WB data)
  
**`_patch_country_group(group)`**
- **Before**: Tried to patch all indicators with FRED/EOD
- **After**: Skips indicators already loaded from OECD/EOD macro
- **Logic**:
  ```python
  if meta.get('oecd_sdmx_params'):
      continue  # Skip OECD indicators
  if meta.get('eod_macro_code'):
      continue  # Skip EOD macro indicators
  ```

### 3. Configuration Updates (`data/infrastructure/macro_indicators.json`)

#### New OECD Indicators:
```json
"s1_domestic_equity_usd": {
  "oecd_sdmx_params": {
    "dataflow": "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.0",
    "key": "A....S1...A..F511+F521..USD......"
  },
  "description": "Household equity holdings (domestic)"
}
```

```json
"s2_foreign_equity_usd": {
  "oecd_sdmx_params": {
    "dataflow": "OECD.SDD.NAD,DSD_NASEC20@DF_T720R_A,1.0",
    "key": "A....S2...A..F511+F521..USD......"
  },
  "description": "Institutional equity holdings (foreign)"
}
```

#### New EOD Macro Indicators:
```json
"national_income_usd": {
  "eod_macro_code": "gni_current_us",
  "wb_code": "NY.GNP.MKTP.CD",
  "description": "Gross National Income"
}
```

```json
"national_savings_usd": {
  "eod_macro_code": "gross_savings_current_us",
  "wb_code": "NY.GNS.ICTR.CD",
  "description": "Gross National Savings"
}
```

## Data Flow

```
┌─────────────┐
│  Countries  │ (from exchanges table)
└──────┬──────┘
       │
       ├───────────────┐
       │               │
       ▼               ▼
  ┌─────────┐   ┌──────────┐   ┌─────────────┐
  │World Bank│   │   OECD   │   │  EOD Macro  │
  │ (WB API) │   │(SDMX 3.0)│   │(get_macro)  │
  └────┬────┘   └─────┬────┘   └──────┬──────┘
       │              │                │
       └──────────────┴────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Merge onto   │
              │  Skeleton    │
              │ (ISO × Year) │
              └──────┬───────┘
                     │
                     ▼
              ┌─────────────┐
              │   Patch     │
              │ with FRED   │
              │  & EOD      │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ macro_raw   │
              │   (table)   │
              └─────────────┘
```

## Testing

Run the exploration notebook to test the integration:

```bash
# Open notebook
code ipynb/pipeline4_exploration.ipynb

# Execute test cells (at bottom of notebook):
# - Test 1: OECD S1 domestic equity fetch
# - Test 2: OECD S2 foreign equity fetch
# - Test 3: EOD GNI macro fetch
# - Test 4: Full pipeline integration test
```

Expected results:
- OECD S1/S2 data: ~38 countries (OECD members), 2010-2023
- EOD GNI/Savings: Available for major economies (USA confirmed)
- Full integration: All 4 indicators merged successfully for test countries

## Known Issues & Limitations

1. **OECD SSL Certificates**: 
   - OECD SDMX API has SSL certificate issues
   - Workaround: `verify=False` in requests (acceptable for read-only public API)

2. **OECD Data Coverage**:
   - Limited to OECD member countries (~38 countries)
   - Annual data only (quarterly not available for sector accounts)
   - Historical coverage varies by country (typically 2010+)

3. **EOD Macro Aggregation**:
   - EOD returns quarterly data, pipeline aggregates to annual using mean
   - May not match official annual figures for stock variables
   - Coverage varies by country and indicator

4. **SDMX Dataflow Complexity**:
   - OECD SDMX keys are complex (14+ dimensions)
   - Documentation: https://sdmx.oecd.org/
   - Use OECD Data Explorer to find dataflow IDs: https://data.oecd.org/

## Future Enhancements

1. **Caching**: Add local caching for OECD SDMX responses (API is slow)
2. **Error Handling**: Better retry logic for OECD API failures
3. **Validation**: Add data quality checks for OECD/EOD merges
4. **Logging**: Enhanced logging for multi-source fetching
5. **Documentation**: Document SDMX dimension keys for each indicator

## References

- **OECD SDMX API**: https://sdmx.oecd.org/public/rest/data/
- **OECD Data Explorer**: https://data.oecd.org/
- **EOD Macro API**: https://eodhistoricaldata.com/financial-apis/macroeconomics-data-and-macro-indicators-api/
- **SDMX-JSON Format**: https://github.com/sdmx-twg/sdmx-json
