# Factor Timing Framework

A comprehensive framework for implementing and testing factor timing strategies, with proper handling of Cross-Sectional (CS) and Time-Series (TS) factors.

## Overview

This framework implements a robust factor timing methodology for financial markets, allowing for the optimization of portfolio weights using shrinkage techniques, factor rotation, and comprehensive performance evaluation. It analyzes both Cross-Sectional (CS) and Time-Series (TS) factors to identify the most effective combinations for portfolio construction.

## Key Features

- Analyzes 102 factors (49 cross-sectional and 53 time-series variants)
- Applies optimal regularization via shrinkage techniques
- Implements factor rotation to identify the most important factors
- Supports long-short and long-only constraints
- Provides comprehensive performance metrics and visualizations
- Maintains consistent separation between CS and TS factors
- Handles rolling window analysis for proper out-of-sample testing

## Workflow

The complete workflow consists of the following steps:

1. **Data Preparation**:
   ```
   python factor_timing_data_prep.py
   ```
   Loads the raw factor and conditioning variable data, prepares factor timing portfolios by combining factors with lagged predictors, and saves to `prepared_data.pkl`.

2. **Create Rolling Windows**:
   ```
   python factor_timing_rolling_windows.py
   ```
   Constructs rolling windows for training (60 months), validation (12 months), and testing (1 month) periods, saving the result to `rolling_windows.pkl`.

3. **Run Shrinkage Optimization**:
   ```
   python factor_timing_shrinkage.py [options]
   ```
   Optimizes portfolio weights using shrinkage techniques with Ledoit-Wolf covariance estimation, tuning the optimal lambda parameter for each window, and saving results to `shrinkage_results.pkl`.

4. **Extract Shrinkage Intensities**:
   ```
   python print_intensities.py
   ```
   Analyzes and displays the Ledoit-Wolf shrinkage intensities from the optimization results, showing how the covariance structure stability evolves across windows.

5. **Extract Optimal Weights**:
   ```
   python extract_weights.py
   ```
   Extracts the optimal weights for filtered timing portfolios from shrinkage results and saves them to `unrotated_optimal_weights.xlsx` for further analysis.

6. **Process Factor Rotation**:
   ```
   python factor_rotation.py [options]
   ```
   Rotates portfolio weights into factor weights, properly preserving CS and TS factor designations, saving to `rotated_optimal_weights.xlsx`.

7. **Run Complete Test Pipeline**:
   ```
   python run_factor_timing_test.py [options]
   ```
   Executes the full pipeline end-to-end, calculating comprehensive performance metrics, saving detailed results to `test_results.pkl` and `factor_timing_results.xlsx`.

8. **Generate Plots**:
   ```
   python plot_factor_results.py [options]
   ```
   Creates visualizations from test results without rerunning the entire pipeline, including lambda evolution, Sharpe ratios, cumulative returns, drawdowns, and factor weights.

9. **Analyze Factor Weights**:
   ```
   python analyze_weights.py
   ```
   Provides in-depth analysis of factor weights, including CS vs TS distribution and comparative performance.

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
- **rolling_windows.pkl**: Rolling window splits for time series analysis (train/validation/test)
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