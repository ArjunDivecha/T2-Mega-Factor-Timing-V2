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

Many scripts in the pipeline support a consistent command-line interface for specifying which windows to process.

### Common Parameters

Supported parameters in applicable scripts include:

```
--window_indices INDICES   Specific window indices to process, comma-separated (e.g., "120,121,122")
--start_date DATE          Start date for test period (YYYY-MM-DD)
--end_date DATE            End date for test period (YYYY-MM-DD)
--max_windows N            Maximum number of windows to process (default: 5, 0 for all windows)
--lambda_value LAMBDA      Fixed shrinkage intensity to use (where applicable)
```

Specific scripts may support additional parameters as needed for their functionality.

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

## Input Data Requirements

The pipeline requires the following input data:

- **Factor returns data**: Monthly returns for various investment factors, including both CS and TS variants
- **Conditioning variables**: Macroeconomic indicators and market signals (GDP, inflation, volatility indices, etc.)
- **Historical performance data**: Benchmark performance metrics for comparison

The data files should be placed in the appropriate directory structure as expected by the data preparation scripts.

## Output Files

### Data Processing Files

1. **prepared_data.pkl**
   - Contains all preprocessed data for the factor timing model
   - Includes factor returns, portfolio mappings, conditioning variables
   - Format: Python pickle (dictionary of DataFrames)

2. **rolling_windows.pkl**
   - Contains the rolling window definitions
   - **train_indices**: Training window indices (60 months per window)
   - **test_indices**: Test window indices (1 month per window)
   - **train_dates**: Corresponding dates for training periods
   - **test_dates**: Corresponding dates for test periods
   - Format: Python pickle (dictionary)

### Optimization and Weight Files

3. **shrinkage_results.pkl**
   - Results from the top 5 portfolio selection process
   - **window_indices**: List of processed window indices
   - **selected_portfolios**: Top 5 portfolios selected for each window
   - **train_sharpes**: Training Sharpe ratios for selected portfolios
   - **test_sharpes**: Test Sharpe ratios for evaluation
   - **optimal_weights**: Equal weights (20%) assigned to selected portfolios
   - Format: Python pickle (nested dictionaries)

4. **unrotated_optimal_weights.xlsx**
   - Contains the equal weights (20%) for the top 5 selected portfolios
   - Rows: Selected portfolios
   - Columns: Test dates (window end dates)
   - Values: Fixed 20% weights for selected portfolios, 0% for others
   - Format: Excel spreadsheet

5. **rotated_optimal_weights.xlsx**
   - Contains weights rotated back to original factor space
   - Rows: Original factors (CS and TS variants)
   - Columns: Test dates
   - Values: Derived weights for each original factor
   - Format: Excel spreadsheet

### Results and Analysis Files

6. **test_results.pkl**
   - Comprehensive performance results and metrics
   - **strategy_returns**: Historical returns series
   - **performance_metrics**: Performance statistics (Sharpe, returns, volatility)
   - **cumulative_returns**: Cumulative return series
   - **selected_portfolios**: Record of which portfolios were selected
   - Format: Python pickle (nested dictionaries)

7. **factor_timing_results.xlsx**
   - Consolidated results for analysis
   - **Performance Summary**: Overall metrics 
   - **Monthly Returns**: Month-by-month returns
   - **Window Analysis**: Per-window performance
   - **Factor Weights**: Factor weight evolution
   - **CS vs TS Analysis**: Factor type breakdown
   - Format: Excel spreadsheet (multiple sheets)

### Analysis Outputs

8. **Weight Analysis Directory**
   - Various visualizations and analysis of factor weights
   - CS vs TS distribution charts
   - Top factor analysis
   - Comparative performance metrics
   - Format: PDF files and Excel reports

9. **Conditioning Variable Analysis Directory**
   - Analysis of conditioning variable importance
   - Heatmaps of variable importance over time
   - Top conditioning variables by factor
   - Time series plots of variable importance
   - Format: PDF files and Excel reports

## Cross-Sectional vs Time-Series Factors

The pipeline maintains a clear distinction between Cross-Sectional (CS) and Time-Series (TS) factors throughout processing:

- **Cross-Sectional (CS)**: Factors that compare securities against each other at the same point in time
- **Time-Series (TS)**: Factors that compare a security's current values against its own historical values

The Step_8_analyze_weights.py script provides detailed analysis of CS vs TS performance, showing:
- The distribution between CS and TS factors in the selected portfolios
- Which factors perform better in their CS or TS implementations
- How this distribution evolves over time

## Performance Metrics

The pipeline calculates and reports the following performance metrics:
- Annualized returns
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Win rate (% of months with positive returns)
- Monthly return statistics
- Comparison against benchmark performance

## Technical Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- scikit-learn (for machine learning components)
- Joblib (for parallelization where applicable)
- openpyxl (for Excel file handling)

## Project Structure

The implementation follows a sequential pipeline approach with numbered steps:

### Step 0-1: Preparation Phase
- Scripts: `Step_0_Delete_Output.py`, `Step_1_data_prep.py`
- Prepare the environment and preprocess raw data

### Step 2-3: Window Creation and Portfolio Selection
- Scripts: `Step_2_rolling_windows.py`, `Step_3_shrinkage_TOP5.py`
- Create time windows and select top 5 portfolios

### Step 4-5: Weight Extraction and Factor Rotation
- Scripts: `Step_4_extract_weights.py`, `Step_5_factor_rotation.py`
- Extract and transform portfolio weights

### Step 6-8: Analysis and Visualization
- Scripts: `Step_6_Prep_Output.py`, `Step_7_plot_results.py`, `Step_8_analyze_weights.py`
- Generate performance metrics and visualizations

### Step 9-9.5: Country Weight Allocation and Performance Analysis
- Scripts: `Step_9 Write Country Weights.py`, `Step_9.5 Calculate Portfolio Returns.py`
- Convert feature weights to country weights and analyze resulting portfolio performance

### Step 10: Conditioning Variable Analysis
- Script: `Step_10_analyze_conditioning_variables.py`
- Analyze the importance of conditioning variables

## Running the Complete Pipeline Step by Step

To run the entire factor timing pipeline, follow these steps in order:

### Step 0: Clean Up Previous Outputs
```bash
python Step_0_Delete_Output.py
```
This removes any existing output files to ensure a clean run.

### Step 1: Data Preparation
```bash
python Step_1_data_prep.py
```
This preprocesses the raw factor data and conditioning variables.

### Step 2: Generate Rolling Windows
```bash
python Step_2_rolling_windows.py
```
This creates the training (60 months) and testing (1 month) windows.

### Step 3: Select Top 5 Portfolios
```bash
python Step_3_shrinkage_TOP5.py --max_windows 0
```
This selects the top 5 portfolios based on training data and assigns 20% weight to each. The `--max_windows 0` parameter processes all available windows.

### Step 4: Extract Portfolio Weights
```bash
python Step_4_extract_weights.py
```
This extracts the weights (20% each) for the selected top 5 portfolios.

### Step 5: Perform Factor Rotation
```bash
python Step_5_factor_rotation.py
```
This rotates the portfolio weights back to original factor space.

### Step 6: Generate Performance Metrics
```bash
python Step_6_Prep_Output.py --max_windows 0
```
This calculates performance metrics and produces the main results files.

### Step 7: Create Visualizations
```bash
python Step_7_plot_results.py
```
This generates performance charts and visualizations.

### Step 8: Analyze Factor Weights
```bash
python Step_8_analyze_weights.py
```
This analyzes the distribution and importance of factors in the selected portfolios.

### Step 9: Convert Feature Weights to Country Weights

**⚠️ WARNING: Before running Step 9 and Step 9.5, ensure that the required files in `/Users/macbook2024/Dropbox/AAA Backup/Transformer/T2 Factor Timing/` are up-to-date. These scripts rely on external files from this directory.**

```bash
python "Step_9 Write Country Weights.py"
```
Converts feature importance weights from the factor timing model into country-specific investment weights for stock market forecasting. The script reads feature weights from the "Long-Only Weights" sheet in "factor_timing_results.xlsx" and distributes these weights to countries based on their factor values. Output is saved to "Final_Country_Weights.xlsx".

### Step 9.5: Calculate Portfolio Returns

```bash
python "Step_9.5 Calculate Portfolio Returns.py"
```
Calculates and analyzes the performance of the country-weighted investment portfolio by applying country weights to historical returns data. Compares performance against an equal-weight benchmark and generates comprehensive performance metrics and visualizations. Output files include "Final_Portfolio_Returns.xlsx" and "Final_Portfolio_Returns.pdf".

### Step 10: Analyze Conditioning Variables

```bash
python Step_10_analyze_conditioning_variables.py
```
This analyzes which conditioning variables had the most influence on portfolio selection.


## Recent Modifications

### Top 5 Portfolio Selection

The pipeline has been modified to select the top 5 performing portfolios instead of the previous top 10 approach. Each selected portfolio now receives an equal weight of 20%.

### Removal of Validation Window

The validation window has been removed from the rolling window creation. The pipeline now directly tests on the next month after training, simplifying the process and reducing complexity.

### Updated Testing Methodology

The testing methodology has been streamlined to focus on direct evaluation of the top 5 portfolios without additional optimization steps.

### Enhanced Analysis Components

Additional analysis components have been added to provide deeper insights into factor weights (Step 8) and conditioning variable importance (Step 10).