# Database Tables Reference

This document describes the schema and relationships for the ETF Analyzer database.
Architecture: **ELT** (Extract-Load-Transform).

## Table Order (Creation Sequence)

## Scheme 1: Infrastructure

1. **countries** - from pycountry/wb
   - `ISO` (PK, CHAR-2), `ISO3` (CHAR-3), `country_name`, `region`, `income_level`

2. **currencies** - from EOD
   - `currencyID` (PK), `currencyName`, `Ticker`

3. **exchanges** - from EOD
   - `code` (PK), `name`, `ISO` (FK->countries), `Currency`, `MIC`

4. **quarters** - self build
   - `QuarterID` (PK, INT e.g., 20234), `Year`, `Quarter`

## Scheme 2: Core

5. **currency_rates** - from EOD
   - `currencyID` (FK->currencies), `QuarterID` (FK->quarters), `RateToILS`, `RateToUSD`
   - PK: (`currencyID`, `QuarterID`)

6. **companies** - from EOD
   - `ISIN` (PK), `CompanyName`, `ISO` (FK->countries), `Currency`

7. **TradePipelines** - self build
   - `PipelineID` (PK), `ISIN` (FK->companies), `StockTicker`, `Exchange` (FK->exchanges)

## Scheme 3: Portfolio

8. **Managers** - from EDGAR
   - `ManagerID` (PK), `ManagerName`

9. **funds** - from EODGAR
   - `SeriesID` (PK), `fund_ticker`, `fund_name`, `ManagerID` (FK->Managers), `net_assets`

10. **holdings** - from EDGAR
    - `HoldingID` (PK), `SeriesID` (FK->funds), `ISIN` (FK->TradePipelines), `weight`

## Scheme 4: Raw Data (The Data Lake)

11. **macro_raw** - from World Bank / FRED / EOD (OECD Patch)
    - **Note:** Data is stored AS IS (no currency conversion).
    - `ISO` (FK -> countries), `Year` (FK -> quarters)
    - **Economic:** `gdp_nominal` (LCU/USD for TWN)
    - **Liquidity:** `money_supply_m4` (LCU), `central_bank_rate` (%)
    - **Bonds (EOD):** `gov_bond_yield_10y` (%), `gov_bond_yield_3y` (%)
    - **Fiscal:** `gov_debt_total` (LCU), `gov_revenue_lcu` (LCU), `military_exp_lcu` (LCU)
    - **External:** `foreign_capital_inflow_usd` (USD), `reserves_total_usd` (USD)
    - PK: (`ISO`, `Year`)

12. **companies_details** - from EOD Fundamentals
    - `ISIN` (FK->companies), `Industry`, `Sector`, `Description`
    - PK: (`ISIN`)

13. **stocks_raw** - from EOD Fundamentals
    - `ISIN` (FK -> companies), `QuarterID` (FK -> quarters)
    - Data: `revenue`, `net_income`, `market_cap`, `total_assets`, `total_liabilities`, `dividend_paid`, `stock_buybacks`, `investment_cf`, `report_date`
    - PK: (`ISIN`, `QuarterID`)

## Scheme 5: Calculated (The Engine)

14. **stocks_calculated** - self build (Python Script)
    - `ISIN` (FK->companies), `QuarterID` (FK->quarters), `CycleID` (FK->cycles)
    - **Valuation:** `PE`, `MarginalPE`, `PB`, `ROE`
    - **Growth (Marginals):** `MarginalIncome`, `MarginalProfit`, `MarginalBookValue`, `MarginalROE`
    - **Model:** `Premium`, `ExpectedGrowth`, `Premium_profit_nonGrowth`, `Premium_profit_Growth`
    - **Analysis:** `stock_price_yield`, `profit_growth_expection`, `profit_growth`, `growth_model_error`
    - PK: (`ISIN`, `QuarterID`)

15. **cycles** - self build
    - `CycleID` (PK), `ISIN` (FK->companies)
    - `StartedQuarter` (FK->quarters), `PickQuarter` (FK->quarters), `EndQuarter` (FK->quarters)
    - `IsSuccesfull` (Boolean)
    - **Stats:** `BullYield`, `BearYield`, `TotalYield`, `PE_Expansion`, `PE_Contraction`
    - PK: (`CycleID`)

16. **macro_calculated** - self build
    - `ISO` (FK -> countries), `Year` (FK -> quarters)
    - **Aggregated Micro:** `PE`, `PE_growth`, `companies_growth`, `country_growth_beta`, `country_from_global_market_cap`, `country_from_global_net_profit`
    - **Sector/Industry Rates:** `cap_rate_*`, `profit_rate_*`, `growth_beta_*` (broken down by local/global sources for sectors/industrials)
    - **Monetary:** `money_supply_m4`, `exchange_yield`, `cum_exchange_cap`, `stream_yield`, `total_foreign_holdings`, `percent_foreign_holdings`, `central_bank_rate`, `fluidity_growth_*`
    - **Fiscal:** `gov_bond_risk_rate`, `revenue_debt_rate`, `military_revenue_rate`, `gov_debt_total`, `gov_revenue` 
    - PK: (`ISO`, `Year`)

## Scheme 6: Analysis

17. **macro_analysis** - self build
    - `ISO` (FK -> countries), `QuarterID` (FK->quarters)
    - `liquidity_score`
    - `predicted_local_cash_stream_to_stocks`
    - `predicted_forign_cash_stream_to_stocks`
    - PK: (`ISO`, `QuarterID`)

18. **stocks_analysis** - self build
    - `ISIN` (FK->companies), `CycleID` (FK->cycles)
    - `PredictedLength`
    - `PredictedSuccesfullProb`
    - PK: (`ISIN`, `CycleID`)

19. **etf_analysis**
    - `etf_ticker` (FK -> funds), `weighted_prob`, `weighted_cycle_length`, `weighted_pe_deriv`, `coverage_pct`, `last_updated`
    - PK: (`etf_ticker`)

---
For any schema/process changes, update this file and the `DatabaseInitializer` logic accordingly.