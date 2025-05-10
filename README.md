# T2 MEGA FACTOR TIMING V2

A comprehensive framework for implementing and testing factor timing strategies with a focus on selecting the top 5 performing factors and equal-weighting them. This pipeline handles Cross-Sectional (CS) and Time-Series (TS) factors separately.

## Overview

This framework implements a factor timing methodology designed to identify and weight the most promising factor timing portfolios. The pipeline selects the top 5 best-performing portfolios from historical data and assigns equal weights (20% each), then evaluates performance directly on the next month without using a validation window. The complete pipeline includes data preparation, rolling window construction, portfolio selection, factor rotation, and comprehensive performance analysis.

## Key Features

- Selects top 5 best-performing factor timing portfolios based on training data
- Assigns equal weights (20% each) to selected portfolios
- Uses 60-month training windows and 1-month direct testing (no validation window)
- Supports analysis of conditioning variables' importance
- Analyzes both Cross-Sectional (CS) and Time-Series (TS) factors
- Provides comprehensive performance metrics and visualizations

## Pipeline Steps

The complete pipeline consists of the following steps:

### Step 0: Delete Output Files

```bash
python Step_0_Delete_Output.py
```
Cleans up any existing output files to ensure a fresh run of the pipeline.

### Step 1: Data Preparation

```bash
python Step_1_data_prep.py
```
Loads and preprocesses raw factor data, conditioning variables, and creates factor timing portfolios. The output is saved to `prepared_data.pkl`.

Alternative versions are available for specialized data preparation:
- `Step_1_data_prep_T60.py`: 60-month training window version
- `Step_1_data_prep_Bloom.py`: Bloomberg data version

### Step 2: Create Rolling Windows

```bash
python Step_2_rolling_windows.py
```
Constructs rolling windows for training (60 months) and testing (1 month) periods. Unlike previous versions, this implementation doesn't use a validation window and instead directly tests on the next month after training. Results are saved to `rolling_windows.pkl`.

### Step 3: Portfolio Selection and Weight Optimization

```bash
python Step_3_shrinkage_TOP5.py [options]
```
Selects the top 5 best-performing portfolios based on training data and assigns equal weights (20% each) to these portfolios. This script uses a fixed lambda parameter instead of validation-based optimization. Results are saved to `shrinkage_results.pkl`.

Options include:
- `--max_windows N`: Maximum number of windows to process (0 for all)
- `--window_indices INDICES`: Specific window indices to process

### Step 4: Extract Portfolio Weights

```bash
python Step_4_extract_weights.py
```
Extracts the portfolio weights from the shrinkage results file and saves them to `unrotated_optimal_weights.xlsx` for further analysis.

### Step 5: Factor Rotation

```bash
python Step_5_factor_rotation.py
```
Rotates the portfolio weights back to the original factor space, properly accounting for CS and TS factor designations. The rotated weights are saved to `rotated_optimal_weights.xlsx`.

### Step 6: Prepare Output and Performance Metrics

```bash
python Step_6_Prep_Output.py [options]
```
Generates comprehensive performance metrics, summary tables, and exports results to `factor_timing_results.xlsx` and `test_results.pkl`. This script integrates the outputs from previous steps and presents them in a structured format.

### Step 7: Plot Results

```bash
python Step_7_plot_results.py
```
Creates visualizations from the test results, including performance charts, metrics evolution, and factor weight distributions.

### Step 8: Analyze Factor Weights

```bash
python Step_8_analyze_weights.py
```
Provides detailed analysis of factor weights from the top 5 portfolios, including:
- Distribution between CS and TS factors
- Top factors by weight
- CS vs TS comparative performance

### Step 10: Analyze Conditioning Variables

```bash
python Step_10_analyze_conditioning_variables.py
```
Analyzes which conditioning variables (like GDP, inflation, etc.) were most important at different points in time based on the unrotated weights. Generates visualizations of conditioning variable importance and their time series evolution.

*Note: There is no Step 9 in the current pipeline implementation.*

## Command-Line Interface

All scripts use a consistent command-line interface, allowing you to specify window parameters in the same way across all components.

### Common Parameters

All scripts support the following parameters:

```
--window_indices INDICES   Specific window indices to process, comma-separated (e.g., "120,121,122")
--start_date DATE          Start date for test period (YYYY-MM-DD)
--end_date DATE            End date for test period (YYYY-MM-DD)
--max_windows N            Maximum number of windows to process (default: 5, 0 for all windows)
--max_portfolios N         Maximum number of portfolios to analyze (default: 2000)
--find_2010_window         Find and process window around 2010
```

Additionally, `run_factor_timing_test.py` supports these parameters:

```
--run_shrinkage            Force rerunning shrinkage optimization even if results exist
--export_excel             Export results to Excel (default: True if not already exists)
--export_plots             Export plots to PDF files (default: True)
```

**Note**: The parameters `--window_indices`, `--start_date/--end_date`, and `--find_2010_window` are mutually exclusive. If none are provided, the script will process up to `max_windows` most recent windows.

### Examples

1. Process specific window indices:
   ```
   python run_factor_timing_test.py --window_indices 120,121,122
   ```

2. Process windows in a date range:
   ```
   python run_factor_timing_test.py --start_date 2015-01-01 --end_date 2015-12-31
   ```

3. Process all windows:
   ```
   python run_factor_timing_test.py --max_windows 0
   ```

4. Generate only plots from existing results:
   ```
   python plot_factor_results.py
   ```

## Input Files

- **T2 Mega Factor Conditioning Variables.xlsx**: Primary input file containing:
  - Factor returns data (CS and TS variants)
  - Conditioning variables (macro predictors)
  - Historical performance data

## Output Files

- **prepared_data.pkl**: Preprocessed factor timing data with standardized conditioning variables
- **rolling_windows.pkl**: Rolling window splits for time series analysis (train/test)
- **shrinkage_results.pkl**: Results from shrinkage optimization, including optimal lambda values
- **factor_weights.pkl**: Factor weights data with proper CS/TS designations
- **test_results.pkl**: Complete test results and performance metrics
- **factor_timing_results.xlsx**: Excel file with detailed results and statistics including:
  - Performance Summary sheet
  - Window Details sheet
  - Long-Only Weights sheet
  - Long-Short Weights sheet

### Generated Visualizations

- **lambda_plot.pdf**: Evolution of lambda values over time
- **sharpe_plot.pdf**: Evolution of Sharpe ratios over time
- **cumulative_plot.pdf**: Cumulative returns comparison
- **drawdown_plot.pdf**: Drawdown comparison
- **factor_weights_long_short.pdf**: Long-short factor weights over time
- **factor_weights_with_long_only_constraint.pdf**: Long-only factor weights over time
- **cs_vs_ts_allocation.pdf**: Distribution of allocation between CS and TS factors

## Comprehensive Output Files Reference

The factor timing framework generates multiple output files at each stage of processing. Here's a detailed breakdown of all output files and their contents:

### Data Processing Files

1. **prepared_data.pkl**
   - Contains all preprocessed data needed for the factor timing model
   - **factor_returns**: Original factor returns data (76 factors, cleaned and scaled)
   - **macro_vars_std**: Standardized macroeconomic and technical conditioning variables
   - **trailing_*MTR**: Trailing returns at different time horizons for each factor
   - **factor_timing**: ~30,000 factor timing portfolios (cross-products of factors and predictors)
   - Format: Python pickle (dictionary of pandas DataFrames)

2. **rolling_windows.pkl**
   - Contains time-series partitioned data for model training and evaluation
   - **training_windows**: List of 60-month windows for model training
   - **validation_windows**: List of 12-month windows for hyperparameter tuning
   - **testing_windows**: List of 1-month windows for out-of-sample evaluation
   - **window_dates**: Dictionary mapping window indices to their date ranges
   - **filtered_portfolios**: List of ~372 statistically significant portfolios
   - Format: Python pickle (dictionary of lists and DataFrames)

### Optimization and Weight Files

3. **shrinkage_results.pkl**
   - Contains complete results from the shrinkage optimization process
   - **window_indices**: List of processed window indices
   - **optimal_lambdas**: Optimal lambda values for each window
   - **portfolio_weights**: Optimized weights for filtered portfolios
   - **training_sharpe**: In-sample Sharpe ratios
   - **validation_sharpe**: Validation period Sharpe ratios
   - **test_sharpe**: Out-of-sample Sharpe ratios
   - **shrinkage_intensities**: Ledoit-Wolf shrinkage intensity (δ*) values
   - **test_returns**: Portfolio returns in testing periods
   - Format: Python pickle (dictionary of arrays and scalars)

4. **unrotated_optimal_weights.xlsx**
   - Contains optimal weights for the filtered factor timing portfolios
   - Rows: ~372 filtered timing portfolios
   - Columns: Test dates (window end dates)
   - Values: Optimal portfolio weights from shrinkage optimization
   - Format: Excel spreadsheet (single sheet)

5. **rotated_optimal_weights.xlsx**
   - Contains optimal weights rotated back to original factors
   - Rows: ~76 original factors
   - Columns: Test dates (window end dates)
   - Values: Aggregated weights per original factor
   - Format: Excel spreadsheet (single sheet)

6. **factor_weights.pkl**
   - Complete factor rotation results with long-short and long-only constraints
   - **window_indices**: List of processed window indices
   - **factor_weights**: Long-short factor weights for each window
   - **long_only_weights**: Factor weights with long-only constraint applied
   - **factor_names**: List of unique factor names
   - **window_dates**: Dictionary mapping window indices to their date ranges
   - Format: Python pickle (dictionary of arrays and metadata)

### Results and Analysis Files

7. **test_results.pkl**
   - Comprehensive test results from the complete pipeline
   - **factor_weights**: Long-short factor weights
   - **long_only_weights**: Long-only factor weights
   - **window_dates**: Date information for each window
   - **strategy_returns**: Time series of strategy returns
   - **benchmark_returns**: Time series of benchmark returns
   - **performance_metrics**: Sharpe, volatility, returns, drawdowns, etc.
   - **lambda_values**: Regularization parameters used
   - **cs_vs_ts_allocation**: Allocation breakdown by factor type
   - Format: Python pickle (nested dictionaries of results)

8. **factor_timing_results.xlsx**
   - Consolidated results formatted for analysis and presentation
   - **Performance Summary**: Overall metrics (returns, Sharpe, volatility, drawdowns)
   - **Window Details**: Lambda, shrinkage intensity, and metrics per window
   - **Long-Only Weights**: Factor weights with long-only constraint
   - **Long-Short Weights**: Unconstrained factor weights
   - **Plots Guide**: Information about generated visualizations
   - Format: Excel workbook (multiple sheets)

### Visualization Files

9. **lambda_plot.pdf**
   - Time series plot showing the evolution of optimal lambda values
   - X-axis: Window dates
   - Y-axis: Lambda values (regularization strength)
   - Format: PDF visualization

10. **sharpe_plot.pdf**
    - Comparison of Sharpe ratios across different optimization stages
    - Shows training, validation, and test Sharpe ratios
    - Format: PDF visualization

11. **cumulative_plot.pdf**
    - Cumulative returns of the factor timing strategy vs benchmark
    - X-axis: Time
    - Y-axis: Cumulative return (1 + r)
    - Format: PDF visualization

12. **drawdown_plot.pdf**
    - Maximum drawdowns of the strategy and benchmark
    - Shows portfolio value decline from previous peaks
    - Format: PDF visualization

13. **factor_weights_long_short.pdf**
    - Time series plot of long-short factor weights
    - Shows how weights for top factors evolve over time
    - Format: PDF visualization

14. **factor_weights_with_long_only_constraint.pdf**
    - Time series plot of long-only factor weights
    - Shows evolution of implementable weights over time
    - Format: PDF visualization

15. **cs_vs_ts_allocation.pdf**
    - Pie chart showing allocation between Cross-Sectional and Time-Series factors
    - Comparative analysis of CS vs TS factor performance
    - Format: PDF visualization

## Cross-Sectional vs Time-Series Factors

This framework carefully maintains the distinction between Cross-Sectional (CS) and Time-Series (TS) factors:

- **Cross-Sectional (CS)**: Factors based on the relative performance of securities within the same time period
- **Time-Series (TS)**: Factors based on the performance of securities relative to their own history

Analysis of CS vs TS factor performance reveals:
- Distribution between CS (42%) and TS (58%) factors in optimal portfolios
- Some factors perform better in CS form (e.g., Best ROE, 12MTR, Debt to EV)
- Others perform better in TS form (e.g., 12-1MTR, Inflation, 10Yr Bond)

## Performance Metrics

The framework calculates and reports comprehensive performance metrics including:
- Annualized returns
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Win rate
- Portfolio turnover

## Technical Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Joblib (for parallelization)

## Project Structure

The implementation follows a multi-phase approach:

### Phase 1: Data Preparation
- Script: `factor_timing_data_prep.py`
- Loads raw data and creates factor timing portfolios

### Phase 2: Rolling Windows
- Script: `factor_timing_rolling_windows.py`
- Creates rolling window splits for proper time series validation

### Phase 3: Shrinkage Optimization
- Script: `factor_timing_shrinkage.py`
- Optimizes portfolio weights with regularization

### Phase 4: Factor Rotation
- Script: `factor_rotation.py`
- Rotates portfolio weights to interpretable factor weights

### Phase 5: Performance Evaluation
- Scripts: `run_factor_timing_test.py` and `plot_factor_results.py`
- Calculates performance metrics and generates visualizations

## Running the Complete Pipeline Step by Step

To run the entire factor timing pipeline for all available windows, follow these steps:

### 1. Data Preparation
```
python factor_timing_data_prep.py
```
This loads the raw factor and conditioning variable data and prepares factor timing portfolios.

### 2. Generate Rolling Windows
```
python factor_timing_rolling_windows.py
```
This creates rolling windows for training and testing periods (without validation).

### 3. Run Shrinkage Optimization for All Windows
```
python Step_3_shrinkage_TOP5.py --max_windows 0
```
The `--max_windows 0` parameter processes all available windows. This is the most time-consuming step.

### 4. Extract Portfolio Weights
```
python Step_4_extract_weights.py
```
This extracts the optimal weights from the shrinkage results.

### 5. Perform Factor Rotation
```
python Step_5_factor_rotation.py
```
This rotates portfolio weights into factor weights.

### 6. Generate Performance Metrics and Visualizations
```
python Step_6_Prep_Output.py --max_windows 0
```
This processes all windows and generates performance metrics. The script will automatically use the existing shrinkage results without rerunning the optimization.

If you want to force rerunning the shrinkage optimization:
```
python run_factor_timing_test.py --max_windows 0 --run_shrinkage
```

### 7. Generate Additional Visualizations
```
python Step_7_plot_results.py
```
This creates additional visualizations from the test results.

### 8. Analyze Weights
```
python Step_8_analyze_weights.py
```
This analyzes the weights from the top 5 portfolios.

### 10. Analyze Conditioning Variables
```
python Step_10_analyze_conditioning_variables.py
```
This analyzes the importance of conditioning variables in the top 5 portfolios.

## Recent Modifications

### Enhanced Mean Returns Impact

The `factor_timing_shrinkage.py` script has been modified to scale the mean returns vector by a factor of 10.0 in the `optimal_portfolio_weights` function. This emphasizes the impact of expected returns during portfolio optimization:

```python
# Adjust the impact of the mean returns vector
return_multiplier = 10.0  # Adjust this value to control emphasis
mu_hat = mu_hat * return_multiplier
```

### Optimized Performance Analysis

The `run_factor_timing_test.py` script has been modified to skip rerunning the time-consuming shrinkage optimization step when not needed. It now checks if the shrinkage results file exists and uses it directly unless the `--run_shrinkage` flag is specified.