"""
Pipeline 7: Business Intelligence (BI)
- Purpose: Load gold data and create visualizations to test hypotheses
- Input: All calculated and analysis tables from database
- Output: Interactive plots and statistical tables
"""

from global_import import pd, np, Literal, Tuple
from clients.database_manager import DatabaseManager
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class BIAnalyzer:
    def __init__(self):
        self.db = DatabaseManager()
        self.conn = self.db.get_connection()
        self.cursor = self.conn.cursor()
        
        # Load core datasets
        self.cycles_df = None
        self.stocks_calc_df = None
        self.macro_calc_df = None
        self.macro_analysis_df = None
        
    def load_gold_data(self):
        """Load all calculated and analysis data from database"""
        print("Loading gold data from database...")
        
        # Load cycles with success indicators
        self.cycles_df = self.db.fetch_df('''
            SELECT c.*, 
                   co.ISO,
                   co.CompanyName,
                   (c.EndQuarter - c.StartedQuarter) as CycleLengthQuarters,
                   CASE WHEN c.IsSuccesfull = 1 THEN 'Successful' ELSE 'Failed' END as CycleOutcome
            FROM cycles c
            LEFT JOIN companies co ON c.ISIN = co.ISIN
            WHERE c.EndQuarter IS NOT NULL
        ''')
        
        # Load stocks calculated data
        self.stocks_calc_df = self.db.fetch_df('''
            SELECT sc.*, 
                   co.ISO,
                   co.CompanyName,
                   cd.Sector,
                   cd.Industry
            FROM stocks_calculated sc
            LEFT JOIN companies co ON sc.ISIN = co.ISIN
            LEFT JOIN companies_details cd ON sc.ISIN = cd.ISIN
        ''')
        
        # Load macro calculated with central bank rates
        self.macro_calc_df = self.db.fetch_df('''
            SELECT mc.*,
                   c.country_name,
                   c.region,
                   c.income_level
            FROM macro_calculated mc
            LEFT JOIN countries c ON mc.ISO = c.ISO
        ''')
        
        # Load macro analysis (liquidity scores)
        self.macro_analysis_df = self.db.fetch_df('''
            SELECT ma.*,
                   c.country_name
            FROM macro_analysis ma
            LEFT JOIN countries c ON ma.ISO = c.ISO
        ''')
        
        print(f"✓ Loaded {len(self.cycles_df)} cycles")
        print(f"✓ Loaded {len(self.stocks_calc_df)} stock-quarter records")
        print(f"✓ Loaded {len(self.macro_calc_df)} macro-quarter records")
        print(f"✓ Loaded {len(self.macro_analysis_df)} macro analysis records")
        
    def test_hypothesis_central_bank_rate(self):
        """
        Hypothesis: High central bank rate significantly increases probability 
        of cycle being successful and ending with market increase.
        """
        print("\n" + "="*80)
        print("HYPOTHESIS TEST: Central Bank Rate Impact on Cycle Success")
        print("="*80)
        
        # Merge cycles with macro data at cycle start
        cycles_with_macro = self.cycles_df.merge(
            self.macro_calc_df[['ISO', 'Year', 'central_bank_rate', 'money_supply_m4']],
            left_on=['ISO', 'StartedQuarter'],
            right_on=['ISO', 'Year'],
            how='left',
            suffixes=('', '_start')
        )
        
        # Filter out missing data
        cycles_with_macro = cycles_with_macro.dropna(subset=['central_bank_rate', 'IsSuccesfull'])
        
        if len(cycles_with_macro) == 0:
            print("⚠ No data available with central bank rates. Cannot test hypothesis.")
            return None
        
        # Define high/low central bank rate (median split)
        median_rate = cycles_with_macro['central_bank_rate'].median()
        cycles_with_macro['RateCategory'] = cycles_with_macro['central_bank_rate'].apply(
            lambda x: 'High Rate' if x > median_rate else 'Low Rate'
        )
        
        # Calculate success rates by rate category
        success_by_rate = cycles_with_macro.groupby('RateCategory').agg({
            'IsSuccesfull': ['sum', 'count', 'mean'],
            'TotalYield': 'mean',
            'CycleLengthQuarters': 'mean',
            'central_bank_rate': 'mean'
        }).round(4)
        
        print("\n📊 Success Rates by Central Bank Rate Category")
        print("-" * 80)
        print(success_by_rate)
        
        # Statistical test: Chi-square for independence
        contingency_table = pd.crosstab(
            cycles_with_macro['RateCategory'], 
            cycles_with_macro['IsSuccesfull']
        )
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        print(f"\n📈 Chi-Square Test Results:")
        print(f"   Chi-square statistic: {chi2:.4f}")
        print(f"   P-value: {p_value:.4f}")
        print(f"   Degrees of freedom: {dof}")
        
        if p_value < 0.05:
            print(f"   ✓ SIGNIFICANT relationship (p < 0.05)")
        else:
            print(f"   ✗ NOT significant (p >= 0.05)")
        
        # T-test for yields
        high_rate_yields = cycles_with_macro[cycles_with_macro['RateCategory'] == 'High Rate']['TotalYield'].dropna()
        low_rate_yields = cycles_with_macro[cycles_with_macro['RateCategory'] == 'Low Rate']['TotalYield'].dropna()
        
        if len(high_rate_yields) > 0 and len(low_rate_yields) > 0:
            t_stat, t_pvalue = stats.ttest_ind(high_rate_yields, low_rate_yields)
            print(f"\n📊 T-Test for Total Yield Difference:")
            print(f"   T-statistic: {t_stat:.4f}")
            print(f"   P-value: {t_pvalue:.4f}")
            print(f"   Mean yield (High Rate): {high_rate_yields.mean():.2%}")
            print(f"   Mean yield (Low Rate): {low_rate_yields.mean():.2%}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Success rate by central bank rate bins
        cycles_with_macro['RateBin'] = pd.cut(cycles_with_macro['central_bank_rate'], bins=5)
        success_by_bin = cycles_with_macro.groupby('RateBin')['IsSuccesfull'].agg(['mean', 'count'])
        success_by_bin = success_by_bin[success_by_bin['count'] >= 5]  # Filter bins with <5 samples
        
        axes[0, 0].bar(range(len(success_by_bin)), success_by_bin['mean'], color='steelblue')
        axes[0, 0].set_xlabel('Central Bank Rate Range')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Cycle Success Rate by Central Bank Rate')
        axes[0, 0].set_xticks(range(len(success_by_bin)))
        axes[0, 0].set_xticklabels([f"{interval.left:.1f}-{interval.right:.1f}%" for interval in success_by_bin.index], 
                                    rotation=45, ha='right')
        axes[0, 0].axhline(y=cycles_with_macro['IsSuccesfull'].mean(), color='red', 
                           linestyle='--', label=f'Overall Mean: {cycles_with_macro["IsSuccesfull"].mean():.2%}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of rates by outcome
        successful = cycles_with_macro[cycles_with_macro['IsSuccesfull'] == 1]['central_bank_rate']
        failed = cycles_with_macro[cycles_with_macro['IsSuccesfull'] == 0]['central_bank_rate']
        
        axes[0, 1].hist([successful, failed], bins=20, label=['Successful', 'Failed'], 
                       color=['green', 'red'], alpha=0.6)
        axes[0, 1].set_xlabel('Central Bank Rate (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Central Bank Rates by Cycle Outcome')
        axes[0, 1].legend()
        axes[0, 1].axvline(median_rate, color='black', linestyle='--', label=f'Median: {median_rate:.2f}%')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter - Rate vs Total Yield
        colors = cycles_with_macro['IsSuccesfull'].map({1: 'green', 0: 'red'})
        axes[1, 0].scatter(cycles_with_macro['central_bank_rate'], 
                          cycles_with_macro['TotalYield'], 
                          c=colors, alpha=0.5, s=50)
        axes[1, 0].set_xlabel('Central Bank Rate (%)')
        axes[1, 0].set_ylabel('Total Cycle Yield')
        axes[1, 0].set_title('Central Bank Rate vs Cycle Yield')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add trendline
        valid_data = cycles_with_macro[['central_bank_rate', 'TotalYield']].dropna()
        if len(valid_data) > 2:
            z = np.polyfit(valid_data['central_bank_rate'], valid_data['TotalYield'], 1)
            p = np.poly1d(z)
            axes[1, 0].plot(valid_data['central_bank_rate'].sort_values(), 
                           p(valid_data['central_bank_rate'].sort_values()), 
                           "b--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
            axes[1, 0].legend()
        
        # Plot 4: Box plot - Yields by rate category
        data_to_plot = [high_rate_yields, low_rate_yields]
        axes[1, 1].boxplot(data_to_plot, labels=['High Rate', 'Low Rate'])
        axes[1, 1].set_ylabel('Total Cycle Yield')
        axes[1, 1].set_title('Yield Distribution by Central Bank Rate Category')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/bi_hypothesis_central_bank_rate.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved visualization: data/bi_hypothesis_central_bank_rate.png")
        plt.show()
        
        return cycles_with_macro
    
    def test_hypothesis_pe_derivatives_predict_cycles(self):
        """
        Hypothesis: 2nd derivative of PE ratio signals cycle turning points
        """
        print("\n" + "="*80)
        print("HYPOTHESIS TEST: PE Derivatives Predict Cycle Turning Points")
        print("="*80)
        
        # Merge stocks calculated with cycle information
        stocks_with_cycles = self.stocks_calc_df.merge(
            self.cycles_df[['CycleID', 'ISIN', 'StartedQuarter', 'PickQuarter', 'EndQuarter', 'IsSuccesfull']],
            on=['ISIN', 'CycleID'],
            how='left'
        )
        
        # Identify phase of each quarter within cycle
        def get_cycle_phase(row):
            if pd.isna(row['StartedQuarter']) or pd.isna(row['QuarterID']):
                return 'No Cycle'
            if row['QuarterID'] == row['StartedQuarter']:
                return 'Start (Trough)'
            elif row['QuarterID'] == row['PickQuarter']:
                return 'Peak'
            elif row['QuarterID'] == row['EndQuarter']:
                return 'End'
            elif row['QuarterID'] < row['PickQuarter']:
                return 'Bull Phase'
            elif row['QuarterID'] > row['PickQuarter']:
                return 'Bear Phase'
            else:
                return 'No Cycle'
        
        stocks_with_cycles['CyclePhase'] = stocks_with_cycles.apply(get_cycle_phase, axis=1)
        
        # Calculate derivatives
        stocks_with_cycles = stocks_with_cycles.sort_values(['ISIN', 'QuarterID'])
        stocks_with_cycles['derivative_pe'] = stocks_with_cycles.groupby('ISIN')['PE'].diff()
        stocks_with_cycles['2nd_derivative_pe'] = stocks_with_cycles.groupby('ISIN')['derivative_pe'].diff()
        
        # Analyze derivatives by cycle phase
        phase_analysis = stocks_with_cycles.groupby('CyclePhase').agg({
            'PE': ['mean', 'std'],
            'derivative_pe': ['mean', 'std'],
            '2nd_derivative_pe': ['mean', 'std', 'count']
        }).round(4)
        
        print("\n📊 PE Derivatives by Cycle Phase")
        print("-" * 80)
        print(phase_analysis)
        
        # Statistical test: ANOVA for 2nd derivative across phases
        phases = ['Start (Trough)', 'Bull Phase', 'Peak', 'Bear Phase']
        phase_groups = [stocks_with_cycles[stocks_with_cycles['CyclePhase'] == phase]['2nd_derivative_pe'].dropna() 
                       for phase in phases if phase in stocks_with_cycles['CyclePhase'].values]
        
        if len(phase_groups) >= 2:
            f_stat, p_value = stats.f_oneway(*phase_groups)
            print(f"\n📈 ANOVA Test (2nd Derivative across Phases):")
            print(f"   F-statistic: {f_stat:.4f}")
            print(f"   P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"   ✓ SIGNIFICANT differences between phases (p < 0.05)")
            else:
                print(f"   ✗ NOT significant (p >= 0.05)")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Mean derivatives by phase
        phase_means = stocks_with_cycles.groupby('CyclePhase')[['derivative_pe', '2nd_derivative_pe']].mean()
        phase_means = phase_means.loc[[p for p in phases if p in phase_means.index]]
        
        x = np.arange(len(phase_means))
        width = 0.35
        axes[0, 0].bar(x - width/2, phase_means['derivative_pe'], width, label='1st Derivative', alpha=0.8)
        axes[0, 0].bar(x + width/2, phase_means['2nd_derivative_pe'], width, label='2nd Derivative', alpha=0.8)
        axes[0, 0].set_xlabel('Cycle Phase')
        axes[0, 0].set_ylabel('Mean Derivative Value')
        axes[0, 0].set_title('PE Derivatives by Cycle Phase')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(phase_means.index, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot of 2nd derivative by phase
        phase_data = [stocks_with_cycles[stocks_with_cycles['CyclePhase'] == phase]['2nd_derivative_pe'].dropna() 
                     for phase in phases if phase in stocks_with_cycles['CyclePhase'].values]
        phase_labels = [phase for phase in phases if phase in stocks_with_cycles['CyclePhase'].values]
        
        axes[0, 1].boxplot(phase_data, labels=phase_labels)
        axes[0, 1].set_ylabel('2nd Derivative of PE')
        axes[0, 1].set_title('Distribution of PE 2nd Derivative by Cycle Phase')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Time series example (first 3 stocks)
        example_isins = stocks_with_cycles['ISIN'].dropna().unique()[:3]
        for isin in example_isins:
            stock_data = stocks_with_cycles[stocks_with_cycles['ISIN'] == isin].sort_values('QuarterID')
            if len(stock_data) > 5:
                axes[1, 0].plot(stock_data['QuarterID'], stock_data['PE'], marker='o', label=f'{isin[:8]}...', alpha=0.7)
        
        axes[1, 0].set_xlabel('Quarter ID')
        axes[1, 0].set_ylabel('PE Ratio')
        axes[1, 0].set_title('PE Ratio Time Series (Sample Stocks)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: 2nd derivative distribution
        axes[1, 1].hist(stocks_with_cycles['2nd_derivative_pe'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('2nd Derivative of PE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of PE 2nd Derivative (All Data)')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/bi_hypothesis_pe_derivatives.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved visualization: data/bi_hypothesis_pe_derivatives.png")
        plt.show()
        
        return stocks_with_cycles
    
    def test_hypothesis_growth_model(self):
        """
        Hypothesis: P/B Premium = Expected Growth Rate (Premium-1 ≈ Marginal Book Value)
        """
        print("\n" + "="*80)
        print("HYPOTHESIS TEST: Growth Model Validation (P/B Premium = Expected Growth)")
        print("="*80)
        
        # Filter data with valid growth model components
        model_data = self.stocks_calc_df[
            (self.stocks_calc_df['Premium'].notna()) & 
            (self.stocks_calc_df['MarginalBookValue'].notna()) &
            (self.stocks_calc_df['growth_model_error'].notna())
        ].copy()
        
        if len(model_data) == 0:
            print("⚠ No data available with growth model metrics.")
            return None
        
        # Calculate expected vs actual
        model_data['PremiumGrowth'] = model_data['Premium'] - 1
        model_data['ActualGrowth'] = model_data['MarginalBookValue']
        
        # Correlation analysis
        correlation = model_data[['PremiumGrowth', 'ActualGrowth']].corr().iloc[0, 1]
        print(f"\n📊 Correlation between Premium Growth and Actual Growth: {correlation:.4f}")
        
        # Error analysis
        error_stats = model_data['growth_model_error'].describe()
        print(f"\n📈 Growth Model Error Statistics:")
        print(error_stats)
        
        # Categorize model accuracy
        model_data['ModelAccuracy'] = pd.cut(
            model_data['growth_model_error'], 
            bins=[0, 0.1, 0.25, 0.5, float('inf')],
            labels=['Excellent (<10%)', 'Good (10-25%)', 'Fair (25-50%)', 'Poor (>50%)']
        )
        
        accuracy_distribution = model_data['ModelAccuracy'].value_counts(normalize=True).sort_index()
        print(f"\n📊 Model Accuracy Distribution:")
        print(accuracy_distribution)
        
        # Statistical test: Are premium and growth significantly correlated?
        valid_data = model_data[['PremiumGrowth', 'ActualGrowth']].dropna()
        if len(valid_data) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                valid_data['PremiumGrowth'], valid_data['ActualGrowth']
            )
            print(f"\n📈 Linear Regression Results:")
            print(f"   Slope: {slope:.4f}")
            print(f"   Intercept: {intercept:.4f}")
            print(f"   R-squared: {r_value**2:.4f}")
            print(f"   P-value: {p_value:.4f}")
            
            if p_value < 0.05 and abs(slope - 1.0) < 0.5:
                print(f"   ✓ Model is VALID (slope ≈ 1, significant correlation)")
            else:
                print(f"   ⚠ Model may need recalibration")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Scatter - Premium vs Actual Growth
        axes[0, 0].scatter(model_data['PremiumGrowth'], model_data['ActualGrowth'], 
                          alpha=0.3, s=20)
        axes[0, 0].plot([-1, 2], [-1, 2], 'r--', linewidth=2, label='Perfect Prediction (y=x)')
        
        # Add regression line
        if len(valid_data) > 2:
            axes[0, 0].plot(valid_data['PremiumGrowth'].sort_values(), 
                           slope * valid_data['PremiumGrowth'].sort_values() + intercept,
                           'b-', linewidth=2, label=f'Actual Fit (y={slope:.2f}x+{intercept:.2f})')
        
        axes[0, 0].set_xlabel('Premium Growth (P/B - 1)')
        axes[0, 0].set_ylabel('Actual Book Value Growth')
        axes[0, 0].set_title(f'Growth Model Validation (Correlation: {correlation:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(-1, 2)
        axes[0, 0].set_ylim(-1, 2)
        
        # Plot 2: Error distribution
        axes[0, 1].hist(model_data['growth_model_error'].clip(0, 1), bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Growth Model Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Growth Model Errors')
        axes[0, 1].axvline(x=0.1, color='green', linestyle='--', label='10% Threshold')
        axes[0, 1].axvline(x=0.25, color='orange', linestyle='--', label='25% Threshold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model accuracy pie chart
        accuracy_counts = model_data['ModelAccuracy'].value_counts()
        colors = ['green', 'lightgreen', 'orange', 'red']
        axes[1, 0].pie(accuracy_counts, labels=accuracy_counts.index, autopct='%1.1f%%',
                      colors=colors[:len(accuracy_counts)], startangle=90)
        axes[1, 0].set_title('Model Accuracy Distribution')
        
        # Plot 4: Residuals plot
        model_data['Residual'] = model_data['ActualGrowth'] - model_data['PremiumGrowth']
        axes[1, 1].scatter(model_data['PremiumGrowth'], model_data['Residual'], alpha=0.3, s=20)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Premium Growth (P/B - 1)')
        axes[1, 1].set_ylabel('Residual (Actual - Expected)')
        axes[1, 1].set_title('Model Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/bi_hypothesis_growth_model.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved visualization: data/bi_hypothesis_growth_model.png")
        plt.show()
        
        return model_data
    
    def test_hypothesis_liquidity_leads_cycles(self):
        """
        Hypothesis: Macro liquidity scores predict future PE changes
        """
        print("\n" + "="*80)
        print("HYPOTHESIS TEST: Liquidity Score Leads PE Changes")
        print("="*80)
        
        if self.macro_analysis_df is None or len(self.macro_analysis_df) == 0:
            print("⚠ No macro analysis data available. Run Pipeline 6 first.")
            return None
        
        # Merge liquidity scores with aggregate market PE from macro_calculated
        liquidity_pe = self.macro_analysis_df.merge(
            self.macro_calc_df[['ISO', 'Year', 'PE', 'central_bank_rate']],
            left_on=['ISO', 'QuarterID'],
            right_on=['ISO', 'Year'],
            how='inner'
        )
        
        # Calculate lagged liquidity (1-4 quarters ahead)
        liquidity_pe = liquidity_pe.sort_values(['ISO', 'QuarterID'])
        for lag in [1, 2, 3, 4]:
            liquidity_pe[f'liquidity_lag{lag}'] = liquidity_pe.groupby('ISO')['liquidity_score'].shift(lag)
            liquidity_pe[f'PE_future{lag}'] = liquidity_pe.groupby('ISO')['PE'].shift(-lag)
            liquidity_pe[f'PE_change_future{lag}'] = liquidity_pe.groupby('ISO')['PE'].pct_change(-lag)
        
        # Correlation analysis
        print(f"\n📊 Correlation: Liquidity Score vs Future PE Changes")
        print("-" * 80)
        for lag in [1, 2, 3, 4]:
            corr = liquidity_pe[['liquidity_score', f'PE_change_future{lag}']].corr().iloc[0, 1]
            count = liquidity_pe[[f'PE_change_future{lag}']].notna().sum().iloc[0]
            print(f"   {lag} quarter(s) ahead: {corr:.4f} (n={count})")
        
        # Find best predictive lag
        correlations = []
        for lag in [1, 2, 3, 4]:
            corr = liquidity_pe[['liquidity_score', f'PE_change_future{lag}']].dropna().corr().iloc[0, 1]
            correlations.append((lag, corr))
        
        best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))
        print(f"\n🎯 Best predictive lag: {best_lag} quarter(s) with correlation {best_corr:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Correlation by lag
        lags, corrs = zip(*correlations)
        axes[0, 0].bar(lags, corrs, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Lag (Quarters Ahead)')
        axes[0, 0].set_ylabel('Correlation with PE Change')
        axes[0, 0].set_title('Predictive Power of Liquidity Score by Time Lag')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter - Current liquidity vs future PE change (best lag)
        valid_data = liquidity_pe[['liquidity_score', f'PE_change_future{best_lag}']].dropna()
        axes[0, 1].scatter(valid_data['liquidity_score'], valid_data[f'PE_change_future{best_lag}'], 
                          alpha=0.4, s=30)
        
        if len(valid_data) > 2:
            z = np.polyfit(valid_data['liquidity_score'], valid_data[f'PE_change_future{best_lag}'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(valid_data['liquidity_score'].min(), valid_data['liquidity_score'].max(), 100)
            axes[0, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
                           label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        
        axes[0, 1].set_xlabel('Liquidity Score')
        axes[0, 1].set_ylabel(f'PE Change ({best_lag}Q Ahead)')
        axes[0, 1].set_title(f'Liquidity Score vs Future PE Change (Lag={best_lag}Q)')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Time series example (one country)
        if 'country_name' in liquidity_pe.columns:
            example_country = liquidity_pe['country_name'].value_counts().index[0]
            country_data = liquidity_pe[liquidity_pe['country_name'] == example_country].sort_values('QuarterID')
            
            ax3a = axes[1, 0]
            ax3b = ax3a.twinx()
            
            ax3a.plot(country_data['QuarterID'], country_data['liquidity_score'], 
                     'b-', marker='o', label='Liquidity Score', linewidth=2)
            ax3b.plot(country_data['QuarterID'], country_data['PE'], 
                     'r-', marker='s', label='PE Ratio', linewidth=2, alpha=0.7)
            
            ax3a.set_xlabel('Quarter ID')
            ax3a.set_ylabel('Liquidity Score', color='b')
            ax3b.set_ylabel('PE Ratio', color='r')
            ax3a.set_title(f'Liquidity vs PE Time Series: {example_country}')
            ax3a.tick_params(axis='y', labelcolor='b')
            ax3b.tick_params(axis='y', labelcolor='r')
            ax3a.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax3a.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3a.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 4: Liquidity quartiles vs average PE change
        liquidity_pe['LiquidityQuartile'] = pd.qcut(liquidity_pe['liquidity_score'], 
                                                     q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
        quartile_analysis = liquidity_pe.groupby('LiquidityQuartile')[f'PE_change_future{best_lag}'].mean()
        
        axes[1, 1].bar(range(len(quartile_analysis)), quartile_analysis.values, 
                      color=['red', 'orange', 'lightgreen', 'green'], alpha=0.7)
        axes[1, 1].set_xlabel('Liquidity Score Quartile')
        axes[1, 1].set_ylabel(f'Avg PE Change ({best_lag}Q Ahead)')
        axes[1, 1].set_title(f'Average Future PE Change by Current Liquidity Level')
        axes[1, 1].set_xticks(range(len(quartile_analysis)))
        axes[1, 1].set_xticklabels(quartile_analysis.index, rotation=45, ha='right')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/bi_hypothesis_liquidity_leads.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Saved visualization: data/bi_hypothesis_liquidity_leads.png")
        plt.show()
        
        return liquidity_pe
    
    def create_executive_summary(self):
        """
        Generate executive summary tables for all hypotheses
        """
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY - HYPOTHESIS TEST RESULTS")
        print("="*80)
        
        summary_data = {
            'Hypothesis': [],
            'Status': [],
            'Key Finding': [],
            'Statistical Significance': []
        }
        
        # You can extend this based on test results
        summary_df = pd.DataFrame(summary_data)
        
        print("\n📋 Summary Table")
        print("-" * 80)
        if len(summary_df) > 0:
            print(summary_df.to_string(index=False))
        else:
            print("Run individual hypothesis tests to populate summary.")
        
        return summary_df
    
    def run(self):
        """Execute full BI pipeline"""
        print("\n" + "="*80)
        print("PIPELINE 7: BUSINESS INTELLIGENCE ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_gold_data()
        
        # Test all hypotheses
        print("\n\n🔬 Running Hypothesis Tests...")
        
        # Test 1: Central bank rate impact (NEW HYPOTHESIS)
        cycles_macro = self.test_hypothesis_central_bank_rate()
        
        # Test 2: PE derivatives predict cycles
        stocks_cycles = self.test_hypothesis_pe_derivatives_predict_cycles()
        
        # Test 3: Growth model validation
        growth_model = self.test_hypothesis_growth_model()
        
        # Test 4: Liquidity leads cycles
        liquidity_analysis = self.test_hypothesis_liquidity_leads_cycles()
        
        # Executive summary
        summary = self.create_executive_summary()
        
        print("\n" + "="*80)
        print("✓ Pipeline 7 Complete!")
        print("="*80)
        print("\nGenerated artifacts:")
        print("  • data/bi_hypothesis_central_bank_rate.png")
        print("  • data/bi_hypothesis_pe_derivatives.png")
        print("  • data/bi_hypothesis_growth_model.png")
        print("  • data/bi_hypothesis_liquidity_leads.png")
        print("\n📊 Analysis complete. Review visualizations for insights.")

if __name__ == "__main__":
    bi = BIAnalyzer()
    bi.run()
