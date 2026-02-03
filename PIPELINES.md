# Data Pipelines & Orchestration

This document outlines the 6 distinct pipelines required to populate the ETF Analyzer database.
**Architecture:** ELT (Extract -> Load -> Transform).

---

## Pipeline 1: Infrastructure Initialization
**Goal:** Load static and quasi-static reference data.

### Tables Populated
1. **`countries`** (Source: JSON Config / ISO Standards)
2. **`currencies`** (Source: EOD API)
3. **`exchanges`** (Source: JSON Config / EOD API)
4. **`quarters`** (Source: Generated Logic)

### Execution Logic
1. Load `countries` from pycountry, and fetch region and income level from WB.
2. Fetch supported currencies and exchanges from EOD API to populate `currencies` and `exchanges`.
3. Generate `quarters` table programmatically (e.g., 1990 to 2030).

---

## Pipeline 2: Core Data Ingestion
**Goal:** Establish the universe of tradeable assets and FX rates.

### Tables Populated
5. **`currency_rates`** (Source: EOD API)
6. **`companies`** (Source: EOD Bulk Exchange API)
7. **`trade_pipelines`** (Source: Derived from Companies)

### Execution Logic
* **Currency Rates:**
    * Download historical FX rates for all currencies in `currencies` table.
    * **Processing:** Map daily rates to the start/end of each `QuarterID`. Store rates to ILS and USD.
* **Companies & Pipelines:**
    * Download Ticker Lists **Exchange by Exchange** (Bulk).
    * **Processing:**
        * Aggregate by `ISIN` to remove duplicates (same company on multiple exchanges).
        * Extract the first 2 letters of the `ISIN` to determine `ISO`.
        * *Validation:* If extracted ISO does not exist in `countries`, raise a warning to update `master_country_list.csv`.
        * Populate `companies`.
        * Populate `trade_pipelines` with specific Ticker + Exchange combinations.

---

## Pipeline 3: Portfolio Construction
**Goal:** Map ETF Holdings (The Target Layer).

### Tables Populated
8. **`managers`** (Source: EDGAR)
9. **`funds`** (Source: EDGAR)
10. **`holdings`** (Source: SEC EDGAR Filings)

### Execution Logic
1.  **Scraping:** Draw data from SEC EDGAR system (13F Filings).
2.  **Merging:** Merge raw filings into one large temporary table/dataframe.
3.  **Normalization:**
    * Extract unique Managers -> `managers`.
    * Extract Funds/Series -> `funds`.
    * Map CUSIP/ISIN/LEI from filings to `companies(ISIN)`.
    * Insert linked data into `holdings`.

---

## Pipeline 4: Raw Data Ingestion (The Data Lake)
**Goal:** Fetch Macro and Micro fundamental data *without* heavy processing.

### Tables Populated
11. **`macro_raw`** (Source: World Bank / FRED / EOD Patch)
    * Dynamic loading based on `macro_config.json`.
12. **`companies_details`** (Source: EOD Fundamentals)
13. **`stocks_raw`** (Source: EOD Fundamentals)

### Execution Logic
* **Macro Data:**
    * Iterate `ISO` list.
    * Fetch EOD Data (Bond Yields / GOV debts - Downsample to Annual).
    * Fetch WB Data (Annual).
    * Fetch FRED Data (Patches for TWN/CAN/CHE).
    * map WB and FRED country names into iso.
    * **Store AS IS** (No currency conversion yet).
* **Micro Data (Stocks):**
    * Download Fundamentals (Bulk) via EOD.
    * **Processing:**
        * Shrink pipeline data to `ISIN` level.
        * Map `ReportDate` to `QuarterID`.
        * Store raw values (`revenue`, `net_income`, `market_cap`) in original currency.
* **Company Details:**
    * Update `Industry`, `Sector`, and `Description` (Requires EOD subscription).

---

## Pipeline 5: Calculation Engine (Transformation)
**Goal:** Convert Raw Data into standardized Financial Metrics.

### Tables Populated
14. **`stocks_calculated`** (Source: Derived)
15. **`stocks_cycle`** (Source: Derived)

### Execution Logic (Python Script)
1.  **Load:** Read `stocks_raw`, `companies_details`, `macro_raw`, and `quarters`.
2.  **Calculated Stocks:**
    * Compute `PE`, `PB`, `ROE`, `stock_price_yield`.
    * Compute Marginals and Growth Models (`profit_growth`, `expected_growth`, `growth_model_error`).
    * Compute Valuation Models (`Premium`, `Premium_profit_Growth`).
3.  **Cycle Detection:**
    * Analyze historical PE derivatives (Expansion/Recession).
    * Identify Business Cycles (Start/Pick/End) via State Machine.
    * Aggregate cycle statistics (Yields, PE Expansion) and store in `cycles`.
    * **Map:** Assign `CycleID` back to `stocks_calculated` based on cycle dates.
    * Store complete `stocks_calculated` table.
4.  **Calculated Macro:**
    * Compute aggregated micro-stats (Sector Rates, Exchange Yields, Country PE).
    * Merge with `macro_raw` to calculate Monetary flows (Foreign vs Local holdings).
    * Calculate Fiscal ratios (Revenue/Debt).
    * Store in `macro_calculated` conserving all aggregated micro metrics.

---

## Pipeline 6: AI Analysis & Prediction
**Goal:** Run Machine Learning models and store outputs.

### Tables Populated
17. **`macro_analysis`** (Source: ML Model)
18. **`stocks_analysis`** (Source: ML Model)
19. **`etf_analysis`** (Source: ML Model / Aggregation)

### Execution Logic
1.  **Macro Analysis:**
    * Input: `macro_calculated`.
    * Output: `liquidity_score`, `predicted_local_cash_stream`.
2.  **Stock Analysis:**
    * Input: `stocks_calculated` + `cycles`.
    * Output: `PredictedLength` (of cycle), `PredictedSuccessProb`.
3.  **ETF Analysis:**
    * Aggregate Stock Analysis based on `holdings` weights.
    * Calculate `weighted_prob`, `coverage_pct`.
    * Store Betas and final scores.

---

## Pipeline 7: Business Intelligence (BI)
**Goal:** Load gold data and create visualizations/statistical tests to validate hypotheses.

### Output Artifacts
- **Visualizations:** PNG charts saved to `data/` folder
- **Statistical Reports:** Console output with hypothesis test results

### Hypotheses Tested

#### 1. Central Bank Rate Impact (NEW)
**Hypothesis:** High central bank rates significantly increase the probability of cycles being successful and ending with market increase.

**Tests:**
- Chi-square test for independence (rate category vs cycle success)
- T-test for yield differences (high rate vs low rate environments)
- Correlation analysis (central bank rate vs total cycle yield)

**Outputs:**
- Success rates by rate category
- Yield distributions
- Scatter plots with trend lines
- Statistical significance (p-values)

#### 2. PE Derivatives Predict Cycles
**Hypothesis:** 2nd derivative of PE ratio signals cycle turning points (peaks/troughs).

**Tests:**
- ANOVA test (2nd derivative across cycle phases)
- Mean derivative analysis by phase

**Outputs:**
- Derivatives by cycle phase (Start/Bull/Peak/Bear/End)
- Distribution analysis
- Time series examples

#### 3. Growth Model Validation
**Hypothesis:** P/B Premium = Expected Growth Rate (Premium - 1 ≈ Marginal Book Value).

**Tests:**
- Linear regression (premium growth vs actual growth)
- Correlation analysis
- Error distribution analysis

**Outputs:**
- Scatter plots (expected vs actual)
- Model accuracy distribution (Excellent/Good/Fair/Poor)
- Residual plots

#### 4. Liquidity Leads Cycles
**Hypothesis:** Macro liquidity scores predict future PE changes by 2-4 quarters.

**Tests:**
- Lagged correlation analysis (1-4 quarters)
- Time series analysis
- Quartile analysis (liquidity level vs future returns)

**Outputs:**
- Predictive power by time lag
- Best lag identification
- Country-specific time series
- Liquidity quartile performance

### Execution Logic
1. **Load Gold Data:** Fetch all calculated and analysis tables from database
2. **Run Statistical Tests:** Execute hypothesis-specific tests with proper statistical methods
3. **Generate Visualizations:** Create comprehensive 2x2 subplot charts for each hypothesis
4. **Executive Summary:** Consolidate findings into actionable summary table

### Usage
```python
from processing.pipeline7_bi import BIAnalyzer

bi = BIAnalyzer()
bi.run()  # Run all hypothesis tests

# Or run individual tests:
bi.load_gold_data()
bi.test_hypothesis_central_bank_rate()
bi.test_hypothesis_pe_derivatives_predict_cycles()
bi.test_hypothesis_growth_model()
bi.test_hypothesis_liquidity_leads_cycles()
```