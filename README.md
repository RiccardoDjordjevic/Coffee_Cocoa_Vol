# Agricultural Commodity Volatility Forecasting with Machine Learning and Model Confidence Sets

## Background

This repository contains a comprehensive volatility forecasting framework designed for agricultural commodity price volatility analysis, with implementations for both **cocoa** and **coffee** markets. Volatility forecasting is crucial in commodity markets as it helps traders, risk managers, and agricultural stakeholders understand and predict price uncertainty, which directly impacts hedging strategies, option pricing, and investment decisions.

### Why Agricultural Commodity Volatility Matters

Agricultural commodities are critical global markets with prices influenced by various factors including:

**Cocoa Market Drivers:**
- Weather conditions in major producing regions (Côte d'Ivoire, Ghana, Indonesia)
- Currency fluctuations (XOF/USD, IDR/USD, GHS/USD)
- Chocolate industry demand and stock performance
- Global economic uncertainty indicators

**Coffee Market Drivers:**
- Weather conditions in key growing regions (Indonesia, Vietnam, Brazil)
- Currency fluctuations (IDR/USD, VND/USD, BRL/USD)
- Coffee industry dynamics and company performance
- Global commodity price correlations

This project implements and compares multiple volatility forecasting models to identify the most reliable approaches for predicting agricultural commodity price volatility across different forecasting horizons for both cocoa and coffee markets.

## Methodology

### Volatility Measures
The project implements two sophisticated volatility estimators:
- **Parkinson Volatility**: Uses high-low price ranges for more efficient volatility estimation
- **Garman-Klass Volatility**: Incorporates open, high, low, and close prices for enhanced accuracy

### Model Framework
We implement a **Heterogeneous Autoregressive (HAR)** framework extended with external predictors, comparing:
- **OLS Regression**: Linear baseline model
- **LASSO Regression**: L1 regularization for feature selection
- **Elastic Net**: Combined L1/L2 regularization
- **Bagging Ensemble**: Bootstrap aggregating for robust predictions

### Model Selection
The **Model Confidence Set (MCS)** procedure by Hansen, Lunde, and Nason (2011) is used to statistically identify the set of best-performing models at each forecasting horizon, providing robust model selection beyond simple point estimates.

## Repository Structure

### Data Files

**Cocoa Analysis:**
- `clean_merged_data.csv` - Raw merged dataset with all cocoa-related variables
- `clean_volatility.csv` - Processed dataset with volatility measures and features
- `Var_Desc.csv` - Variable descriptions and category mappings for cocoa

**Coffee Analysis:**
- `all_merged_data_clean.csv` - Raw merged dataset with all coffee-related variables  
- `coffee_volatility.csv` - Processed dataset with volatility measures and features
- `Variable_Desc.csv` - Variable descriptions and category mappings for coffee

### Core Analysis Scripts

#### 1. Feature Engineering

**Cocoa Analysis:**
- `features.py` - Comprehensive feature creation pipeline for cocoa
  - Calculates Parkinson and Garman-Klass volatility measures
  - Creates realized moments (variance, skewness, kurtosis)
  - Generates rolling averages and lagged features
  - Constructs first differences and squared returns
  - Creates chocolate stock index and categorical variables

**Coffee Analysis:**
- `features.py` (Coffee version) - Feature engineering pipeline for coffee
  - Implements identical volatility measures (Parkinson & Garman-Klass)
  - Processes coffee-specific weather data (Indonesia, Vietnam, Brazil)
  - Creates coffee company stock index (CCL, Dydo, ITO, KeyCoffee, Tata)
  - Handles coffee-specific currency pairs (IDR/USD, VND/USD, BRL/USD)

#### 2. Model Implementation

**Cocoa Models:**
- `OLS_MCS.py` - OLS regression with cross-validation and MCS analysis
- `LASSO_MCS.py` - LASSO regression with hyperparameter optimization
- `HAR_EN_MCS.py` - Elastic Net with automated parameter selection
- `HAR_BAG_MCS.py` - Bagging ensemble with optimized estimators

**Coffee Models:**
- `OLS_4.py` - OLS regression implementation for coffee volatility
- `LASSO_2.py` - LASSO regression for coffee markets
- `EN_2.py` - Elastic Net implementation for coffee analysis
- `BAG_2.py` - Bagging ensemble for coffee volatility forecasting

#### 3. Analysis and Evaluation

**Cocoa Analysis:**
- `MCS_final.py` - Comprehensive Model Confidence Set analysis across all models
- `VI_bygroup.py` - Variable importance analysis by category groups
- `Desc_Stat.py` - Descriptive statistics calculator

**Coffee Analysis:**
- `MCS_final.py` (Coffee version) - Model Confidence Set analysis for coffee models
- `VI_by_group.py` - Variable importance analysis for coffee by category groups  
- `Desc_Stat.py` (Coffee version) - Descriptive statistics for coffee data

## Features and Variables

The analysis incorporates multiple categories of predictors tailored to each commodity:

### Core Variables (Both Commodities)
- **Volatility Lags**: HAR components (daily, weekly, monthly)
- **Economic Indicators**: VIX, ADS Index, Economic Policy Uncertainty
- **Commodity Prices**: Related agricultural and energy commodities
- **Equity Markets**: Regional stock market indices
- **Realized Moments**: Higher-order moments of return distributions

### Commodity-Specific Variables

**Cocoa Analysis:**
- **Weather Data**: Temperature and precipitation from Côte d'Ivoire, Ghana, Indonesia
- **Currency Rates**: XOF/USD, IDR/USD, GHS/USD, EUR/USD, GBP/USD
- **Related Equities**: Chocolate company stocks (Hershey, Lindt, Fuji, Guan Chong, Barry Callebaut)
- **Regional Markets**: Netherlands, Germany, Malaysia stock indices

**Coffee Analysis:**
- **Weather Data**: Temperature and precipitation from Indonesia, Vietnam, Brazil
- **Currency Rates**: IDR/USD, VND/USD, BRL/USD, EUR/USD, CHF/USD
- **Related Equities**: Coffee company stocks (CCL, Dydo, ITO, KeyCoffee, Tata)
- **Regional Markets**: Germany, US, France stock indices

### Feature Sets
Each model tests multiple feature combinations:
- `base`: Core HAR volatility lags only
- `all`: Complete feature set
- `weather`: HAR + weather variables
- `uncertainty`: HAR + economic uncertainty measures
- `equity`: HAR + stock market variables
- `fx`: HAR + foreign exchange rates
- `commodity`: HAR + commodity prices
- `moments`: HAR + realized moments
- `pca`: HAR + principal components from variable groups

## Getting Started

### Prerequisites
```python
# Core packages
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
statsmodels>=0.12.0

# Specialized packages
polars>=0.15.0  # For fast data processing
numba>=0.56.0   # For accelerated volatility calculations
scipy>=1.7.0    # For statistical tests
```

### Installation
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Ensure data files are in the `data/` directory

### Workflow

#### Cocoa Volatility Analysis

**Step 1: Feature Engineering**
```bash
python features.py
```
- Processes `clean_merged_data.csv`
- Creates volatility measures and all features
- Outputs `clean_volatility.csv`

**Step 2: Run Individual Models**
```bash
# Run each model (can be parallelized)
python OLS_MCS.py
python LASSO_MCS.py  
python HAR_EN_MCS.py
python HAR_BAG_MCS.py
```

Each script generates:
- Best model results: `best_models_{model}_clean.csv`
- Variable importance: `var_imp_{model}_clean.csv`
- Individual MCS results: `mcs_results_{model}_clean.csv`
- Fold-level losses: `{model}_fold_losses_mcs.csv`

**Step 3: Comprehensive MCS Analysis**
```bash
python MCS_final.py
```
- Combines all model results
- Performs final Model Confidence Set analysis
- Generates summary tables and rankings

**Step 4: Additional Analysis**
```bash
# Variable importance by groups
python VI_bygroup.py

# Descriptive statistics
python Desc_Stat.py
```

#### Coffee Volatility Analysis

**Step 1: Feature Engineering**
```bash
python features.py  # Coffee version
```
- Processes `all_merged_data_clean.csv`
- Creates coffee-specific volatility measures and features
- Outputs `coffee_volatility.csv`

**Step 2: Run Individual Models**
```bash
# Run each model for coffee (can be parallelized)
python OLS_4.py
python LASSO_2.py  
python EN_2.py
python BAG_2.py
```

Each script generates:
- Best model results: `best_models_{model}.csv`
- Variable importance: `var_imp_{model}.csv`
- Individual MCS results: `mcs_results_{model}.csv`
- Fold-level losses: `{model}_fold_losses_mcs.csv`

**Step 3: Comprehensive MCS Analysis**
```bash
python MCS_final.py  # Coffee version
```
- Combines all coffee model results
- Performs final Model Confidence Set analysis
- Generates coffee-specific summary tables

**Step 4: Additional Analysis**
```bash
# Variable importance by groups
python VI_by_group.py

# Descriptive statistics  
python Desc_Stat.py  # Coffee version
```

## Coffee Volatility Analysis Implementation

### Coffee-Specific Features

The coffee analysis extends the cocoa framework with commodity-specific adaptations:

#### Geographic and Weather Variables
- **Indonesia**: Temperature and precipitation data for major coffee-growing regions
- **Vietnam**: Weather conditions affecting robusta coffee production  
- **Brazil**: Climate data from key arabica coffee areas
- **First Differences**: All weather variables are differenced to capture short-term changes

#### Coffee Industry Equity Variables
- **Asian Markets**: CCL, Dydo, ITO, KeyCoffee (Japanese coffee companies)
- **Global Players**: Tata (Indian conglomerate with coffee operations)
- **Coffee Index**: Equal-weighted index of all coffee company stocks
- **Squared Returns**: Volatility measures for coffee equity performance

#### Currency-Specific Variables
- **Producer Currencies**: 
  - IDR/USD (Indonesian Rupiah) - largest robusta producer
  - VND/USD (Vietnamese Dong) - second-largest coffee producer
  - BRL/USD (Brazilian Real) - largest arabica producer
- **Safe Haven Currencies**: CHF/USD alongside traditional EUR/USD
- **Developed Markets**: US, Germany, France equity indices representing major coffee-consuming regions

#### Coffee-Specific Data Processing

**Enhanced Feature Engineering (`features.py`):**
```python
# Coffee company index creation
coffee_columns = ["CCL", "Dydo", "ITO", "KeyCoffee", "Tata"]
coffee_index = equal_weighted_average(coffee_columns)

# Coffee-specific weather variables
weather_vars = ["TempIndo", "PrecIndo", "TempViet", "PrecViet", "TempBras", "PrecBras"]
weather_diffs = [calculate_first_differences(var) for var in weather_vars]

# Currency pairs relevant to coffee trade
fx_pairs = ["IDRUSD", "VNDUSD", "BRLUSD", "CHFUSD", "EURUSD"]
fx_volatilities = [calculate_squared_returns(pair) for pair in fx_pairs]
```

### Model Implementation Differences

#### File Structure and Naming
- **Simplified Naming**: Coffee models use shorter filenames (`OLS_4.py`, `EN_2.py`, `LASSO_2.py`, `BAG_2.py`)
- **Data Sources**: Read from `coffee_volatility.csv` instead of `clean_volatility.csv`
- **Variable Descriptions**: Use `Variable_Desc.csv` instead of `Var_Desc.csv`

#### Coffee-Specific Model Configurations

**Model Parameters:**
- Identical HAR framework (1, 5, 22-day lags)
- Same forecasting horizons (1, 5, 10, 22, 44, 66 days)
- Consistent cross-validation setup (5 folds, 80% training window)
- Same Model Confidence Set parameters (10% significance level)

**Output Files:**
```bash
# Coffee-specific outputs (without "_clean" suffix)
best_models_ols.csv
var_imp_lasso.csv  
mcs_results_en.csv
bagging_fold_losses_mcs.csv
```

### Coffee Market Insights

#### Regional Specialization
- **Robusta Focus**: Indonesian and Vietnamese variables capture robusta market dynamics
- **Arabica Integration**: Brazilian variables and direct arabica price inclusion
- **Processing Countries**: Currency exposure reflects both origin and processing locations

#### Supply Chain Considerations
- **Weather Impact**: Three distinct climate regions with different seasonal patterns
- **Currency Hedging**: Multi-currency exposure reflecting complex global supply chains
- **Industry Consolidation**: Mix of regional specialists and global players in equity component

### Comparative Analysis Framework

Both cocoa and coffee implementations share:
- **Identical Statistical Methods**: Same volatility estimators and model architectures
- **Consistent Evaluation**: Uniform metrics (MSE, MAE, QLike) and MCS procedures
- **Parallel Processing**: Can run both analyses simultaneously
- **Standardized Outputs**: Compatible result formats for cross-commodity comparison

This parallel structure enables direct comparison of:
- **Model Performance**: Which approaches work best for each commodity
- **Variable Importance**: How different factors affect cocoa vs. coffee volatility
- **Forecasting Horizons**: Whether optimal prediction windows differ between markets
- **Feature Relevance**: Comparative analysis of weather, currency, and equity impacts

The models predict volatility at multiple horizons:
- **H=1**: 1-day ahead (daily trading decisions)
- **H=5**: 1-week ahead (weekly planning)
- **H=10**: 2-week ahead (short-term risk management)
- **H=22**: 1-month ahead (monthly hedging)
- **H=44**: 2-month ahead (medium-term planning)
- **H=66**: 3-month ahead (seasonal planning)

## Model Evaluation

### Metrics
- **MSE**: Mean Squared Error (primary ranking metric)
- **MAE**: Mean Absolute Error (robustness check)
- **QLike**: Quasi-likelihood (distribution-based evaluation)

### Cross-Validation
- **Rolling Window**: Expanding 80% training window
- **5-Fold Design**: Maintains temporal structure
- **66-Day Forecast Window**: Realistic out-of-sample evaluation

### Statistical Testing
- **Model Confidence Set**: Identifies statistically equivalent best models
- **Bootstrap Procedures**: 1000 bootstrap samples for robust inference
- **10% Significance Level**: Conservative model selection

## Expected Outputs

### Cocoa Analysis Results
- `mcs_complete_results.csv` - Complete MCS analysis across all metrics
- `mcs_models_only.csv` - Models included in confidence sets
- `mcs_summary_by_case.csv` - Best cocoa models by volatility type and horizon
- `volatility_importance_results.csv` - Variable importance by category

### Coffee Analysis Results  
- `mcs_complete_results.csv` - Complete MCS analysis for coffee models
- `mcs_models_only.csv` - Coffee models in confidence sets
- `mcs_summary_by_case.csv` - Best coffee models by volatility type and horizon
- `coffee_volatility_importance_results.csv` - Coffee variable importance by category

### Cross-Commodity Comparison
- **Model Performance**: Compare forecasting accuracy between cocoa and coffee
- **Variable Importance**: Analyze which factors matter most for each commodity
- **Regional Effects**: Weather vs. currency vs. equity impacts across markets
- **Optimal Horizons**: Identify best prediction windows for each commodity

### Interpretation
Results identify which modeling approaches and feature sets provide the most reliable volatility forecasts for each commodity, enabling practitioners to:
- Select appropriate models for their specific commodity and forecasting horizon
- Understand key risk factors driving volatility in cocoa vs. coffee markets  
- Implement targeted risk management strategies based on commodity-specific insights
- Develop cross-commodity hedging strategies using comparative volatility patterns

## Future Extensions

This framework is designed for extensibility:
- **Additional Commodities**: Framework can be adapted for other agricultural commodities (sugar, cotton, etc.)
- **Real-time Forecasting**: Live data integration capabilities
- **Deep Learning**: Neural network model extensions
- **Multi-step Forecasting**: Joint horizon optimization
- **Risk Management**: VaR and ES prediction integration
- **Cross-Commodity Models**: Joint volatility modeling across cocoa and coffee
- **Supply Chain Analysis**: Integration of shipping and logistics variables

## Technical Notes

### Performance Optimization
- Numba-accelerated volatility calculations
- Polars for fast data processing
- Parallel model execution capability
- Efficient cross-validation design

### Reproducibility
- Fixed random seeds throughout
- Deterministic model training
- Comprehensive logging and diagnostics
- Version-controlled feature engineering

## References

- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453-497.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67-78.

## Contact

This project has been authored by Riccardo Djordjevic under the supervision of [Dr. Thomas Walther](https://sites.google.com/view/thomas-walther) as bachelor thesis for the BSc in Economics and Business Economics, with minor in Applied Data Science at [Utrecht University](https://www.uu.nl/en/bachelors/economics-and-business-economics).

[LinkedIn](https://www.linkedin.com/in/riccardo-djordjevic/)
Email: [riccardo.djordjevic@outlook.com](mailto:riccardo.djordjevic@outlook.com)
[Fairventures Website](https://www.fairventures.earth/)


---

**Note**: This repository contains complete implementations for both cocoa and coffee volatility analysis, enabling direct cross-commodity comparison and comprehensive agricultural market risk assessment.
