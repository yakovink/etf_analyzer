# Pipeline 7: Business Intelligence (BI) Analysis

## Overview

Pipeline 7 is the final analytical layer that validates the core hypotheses of the ETF Analyzer project through statistical testing and visualization. It loads "gold" data from all previous pipelines and generates comprehensive reports to answer key research questions.

## Purpose

This pipeline bridges the gap between data engineering (Pipelines 1-5) and machine learning (Pipeline 6) by providing statistical evidence for investment strategies and model validation.

## Hypotheses Tested

### 1. Central Bank Rate Impact ⭐ NEW
**Question:** Do high central bank rates increase the probability of successful market cycles?

**Methodology:**
- Chi-square test for independence between rate levels and cycle outcomes
- T-test for yield differences across rate environments
- Median split analysis (high vs low rates)

**Key Metrics:**
- Success rate by rate category
- Average total yield by rate environment
- Statistical significance (p-values)

**Expected Insight:** Tight monetary policy may force market discipline, leading to higher quality bull markets with better risk/reward profiles.

---

### 2. PE Derivatives Predict Cycles
**Question:** Can the 2nd derivative of PE ratio predict cycle turning points?

**Methodology:**
- ANOVA test across cycle phases (Start/Bull/Peak/Bear/End)
- Time series analysis of PE acceleration/deceleration

**Key Metrics:**
- Mean 1st and 2nd derivatives by phase
- Distribution analysis at peaks vs troughs

**Expected Insight:** Negative 2nd derivative (deceleration) at peaks signals exhaustion; positive 2nd derivative at troughs signals recovery.

---

### 3. Growth Model Validation
**Question:** Does P/B Premium accurately reflect expected growth (Premium - 1 ≈ Marginal Book Value)?

**Methodology:**
- Linear regression: Premium Growth vs Actual Growth
- Correlation analysis
- Error distribution categorization (Excellent/Good/Fair/Poor)

**Key Metrics:**
- R-squared of regression model
- Mean/median growth model error
- % of stocks with error < 10%

**Expected Insight:** Efficient markets should price growth accurately; large errors indicate mispricing opportunities.

---

### 4. Liquidity Leads Cycles
**Question:** Do macro liquidity scores predict future PE changes by 2-4 quarters?

**Methodology:**
- Lagged correlation analysis (1-4 quarters ahead)
- Quartile analysis (high liquidity → future returns)

**Key Metrics:**
- Correlation by lag period
- Best predictive lag identification
- Country-specific predictive power

**Expected Insight:** Liquidity is a leading indicator; capital flows precede valuation changes.

---

## Output Artifacts

### Visualizations (PNG)
All saved to `data/` folder:

1. **`bi_hypothesis_central_bank_rate.png`** - 4 subplots:
   - Success rate by rate bins
   - Distribution by outcome
   - Rate vs yield scatter
   - Yield box plots

2. **`bi_hypothesis_pe_derivatives.png`** - 4 subplots:
   - Derivatives by phase (bar chart)
   - Distribution by phase (box plot)
   - Time series examples
   - Overall distribution

3. **`bi_hypothesis_growth_model.png`** - 4 subplots:
   - Scatter: Expected vs Actual
   - Error distribution
   - Accuracy pie chart
   - Residuals plot

4. **`bi_hypothesis_liquidity_leads.png`** - 4 subplots:
   - Correlation by lag
   - Liquidity vs future PE scatter
   - Country time series
   - Quartile performance

### Console Reports
- **Statistical test results** (Chi-square, t-tests, ANOVA, regression)
- **P-values and significance indicators**
- **Summary tables** (success rates, correlations, error stats)

---

## Usage

### Python Script
```python
from processing.pipeline7_bi import BIAnalyzer

# Run full pipeline
bi = BIAnalyzer()
bi.run()

# Or run individual tests
bi.load_gold_data()
bi.test_hypothesis_central_bank_rate()
bi.test_hypothesis_pe_derivatives_predict_cycles()
bi.test_hypothesis_growth_model()
bi.test_hypothesis_liquidity_leads_cycles()
```

### Jupyter Notebook
Open `ipynb/pipeline7_bi.ipynb` and run cells interactively for:
- Step-by-step hypothesis testing
- Custom exploratory analysis
- Data quality validation

---

## Dependencies

Required packages (already in `requirements.txt`):
```
matplotlib>=3.5.0
seaborn>=0.12.0
scipy>=1.9.0
pandas
numpy
```

---

## Prerequisites

Before running Pipeline 7, ensure these pipelines are complete:

1. ✅ **Pipeline 5** - `stocks_calculated`, `cycles`, `macro_calculated`
2. ✅ **Pipeline 6** (Optional) - `macro_analysis` (for liquidity hypothesis)

Missing data will result in skipped tests with warnings.

---

## Interpretation Guide

### Statistical Significance
- **p < 0.01**: Very strong evidence (⭐⭐⭐)
- **p < 0.05**: Significant (⭐⭐)
- **p < 0.10**: Marginally significant (⭐)
- **p >= 0.10**: Not significant (✗)

### Correlation Strength
- **|r| > 0.7**: Strong
- **|r| > 0.5**: Moderate
- **|r| > 0.3**: Weak
- **|r| < 0.3**: Very weak

### Model Fit (R²)
- **R² > 0.7**: Excellent explanatory power
- **R² > 0.5**: Good
- **R² > 0.3**: Fair
- **R² < 0.3**: Poor

---

## Expected Runtimes

- **Data Loading**: 5-15 seconds (depends on database size)
- **Central Bank Hypothesis**: 10-20 seconds
- **PE Derivatives**: 15-30 seconds (processes all stock-quarter pairs)
- **Growth Model**: 10-15 seconds
- **Liquidity Leads**: 10-20 seconds
- **Total Pipeline**: ~2-3 minutes

---

## Troubleshooting

### "No data available" warnings
**Cause:** Missing tables from previous pipelines.
**Solution:** Run Pipelines 1-5 first. Pipeline 6 is optional (only needed for liquidity hypothesis).

### Memory errors
**Cause:** Large dataset (>1M records in stocks_calculated).
**Solution:** Add filtering in `load_gold_data()` method (e.g., date range, specific ISINs).

### Visualization issues
**Cause:** Matplotlib backend not configured.
**Solution:** 
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

---

## Extension Ideas

### Additional Hypotheses
1. **Sector Rotation**: Do cycle phases favor specific sectors?
2. **Size Effect**: Do small-cap cycles differ from large-cap?
3. **Regional Differences**: Compare US vs Europe vs Asia cycles
4. **Dividend Impact**: Do high-dividend stocks have different cycle characteristics?

### Enhanced Visualizations
- Interactive Plotly dashboards
- Streamlit web app for real-time exploration
- Tableau/Power BI integration

### Automated Reporting
- PDF report generation
- Email alerts for significant findings
- Scheduled runs with change detection

---

## Integration with Trading Strategy

The BI pipeline informs strategy decisions:

| Finding | Action |
|---------|--------|
| High central bank rate + Early bull phase | **BUY** - High probability of success |
| Negative 2nd derivative + Late bull phase | **SELL** - Peak approaching |
| Low growth model error + High premium | **HOLD** - Fairly valued growth stock |
| Rising liquidity score | **ACCUMULATE** - Bull market ahead (2Q lag) |

---

## Contributing

To add new hypothesis tests:

1. Create method `test_hypothesis_[name]()` in `BIAnalyzer` class
2. Follow existing pattern:
   - Load/merge required data
   - Run statistical tests
   - Print results to console
   - Create 2x2 subplot visualization
   - Save PNG to `data/`
3. Update `run()` method to include new test
4. Document in this README

---

## References

- **Statistical Methods**: Scipy documentation
- **Visualization**: Seaborn gallery, Matplotlib docs
- **Financial Theory**: "Quantitative Momentum" by Wesley Gray, "Valuation" by McKinsey

---

## License

Part of the ETF Analyzer project. See main README for license details.
