#!/usr/bin/env python
"""
# Analyze Conditioning Variables Importance
#
# INPUT FILES:
# - unrotated_optimal_weights.xlsx: Excel file with unrotated weights for factor interactions
#
# This script analyzes which conditioning variables were most important at different points in time
# by examining the unrotated optimal weights from the factor timing model.
# 
# The script provides:
# - Top conditioning variables by average absolute weight
# - Time series analysis of conditioning variable importance
# - Heatmap visualization of the most important variables over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
import os
import requests
import json
import time
import dotenv

# We'll use requests to communicate with LM Studio instead of anthropic
import requests

# Check if LM Studio is available by trying to connect to its API
LM_STUDIO_AVAILABLE = False
try:
    response = requests.get("http://localhost:1234/v1/models")
    if response.status_code == 200:
        LM_STUDIO_AVAILABLE = True
        print("LM Studio API is available")
    else:
        print(f"LM Studio API returned status code {response.status_code}")
except Exception as e:
    print(f"Error connecting to LM Studio API: {e}")
    print("Warning: LM Studio API not available. Analysis will be skipped.")

# Set plot style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_unrotated_weights():
    """Load the unrotated weights file"""
    print("Loading unrotated optimal weights...")
    df = pd.read_excel('unrotated_optimal_weights.xlsx')
    print(f"Loaded data with {df.shape[0]} factor interactions across {df.shape[1]-1} time periods")
    return df

def extract_components(interaction_name):
    """Extract factor and conditioning variable from interaction name"""
    # This is a heuristic based on the naming pattern in the file
    # You may need to adjust this based on your specific naming convention
    parts = interaction_name.split('_')
    
    if len(parts) >= 2:
        if parts[-2] in ['CS', 'TS']:
            # Format: Factor_CS/TS_ConditioningVariable
            factor = '_'.join(parts[:-2]) + '_' + parts[-2]
            conditioning_var = parts[-1]
        else:
            # Format: Factor_ConditioningVariable
            factor = parts[0]
            conditioning_var = '_'.join(parts[1:])
    else:
        # Single part, likely a base factor
        factor = interaction_name
        conditioning_var = "Base"
        
    return factor, conditioning_var

def analyze_weights(df):
    """Analyze the weights to identify important conditioning variables"""
    # Create a copy of the dataframe with only the weights
    weights_df = df.copy()
    interaction_names = weights_df.iloc[:, 0]
    weights_df = weights_df.iloc[:, 1:]
    
    # Extract factor and conditioning variable from interaction names
    components = [extract_components(name) for name in interaction_names]
    factors = [comp[0] for comp in components]
    conditioning_vars = [comp[1] for comp in components]
    
    # Add components back to the dataframe
    weights_df.insert(0, 'Interaction', interaction_names)
    weights_df.insert(1, 'Factor', factors)
    weights_df.insert(2, 'ConditioningVar', conditioning_vars)
    
    # Calculate absolute weights for importance
    abs_weights = weights_df.iloc[:, 3:].abs()
    
    # Calculate average absolute weight for each interaction
    weights_df['AvgAbsWeight'] = abs_weights.mean(axis=1)
    
    # Sort by average absolute weight
    sorted_weights = weights_df.sort_values('AvgAbsWeight', ascending=False)
    
    return sorted_weights

def analyze_conditioning_vars(sorted_weights):
    """Analyze conditioning variables across all factors"""
    # Group by conditioning variable and calculate average importance
    cond_var_importance = sorted_weights.groupby('ConditioningVar')['AvgAbsWeight'].mean().sort_values(ascending=False)
    
    # Count non-zero weights for each conditioning variable
    non_zero_counts = {}
    for cond_var in sorted_weights['ConditioningVar'].unique():
        subset = sorted_weights[sorted_weights['ConditioningVar'] == cond_var]
        non_zero = (subset.iloc[:, 3:-1] != 0).sum().sum()
        non_zero_counts[cond_var] = non_zero
    
    non_zero_df = pd.DataFrame.from_dict(non_zero_counts, orient='index', columns=['NonZeroCount'])
    non_zero_df = non_zero_df.sort_values('NonZeroCount', ascending=False)
    
    return cond_var_importance, non_zero_df

def analyze_time_variation(sorted_weights):
    """Analyze time variation of conditioning variables importance"""
    # Get date columns (all columns except Interaction, Factor, ConditioningVar, and AvgAbsWeight)
    date_columns = sorted_weights.columns[3:-1]
    
    # Create a DataFrame to store time variation
    time_df = pd.DataFrame(index=date_columns)
    
    # Get top conditioning variables
    top_cond_vars = sorted_weights.groupby('ConditioningVar')['AvgAbsWeight'].mean().nlargest(10).index.tolist()
    
    # For each top conditioning variable, calculate average absolute weight for each date
    for cond_var in top_cond_vars:
        subset = sorted_weights[sorted_weights['ConditioningVar'] == cond_var]
        for date in date_columns:
            time_df.loc[date, cond_var] = subset[date].abs().mean()
    
    # Fill NaN values with 0
    time_df = time_df.fillna(0)
    
    # Convert index to datetime
    time_df.index = pd.to_datetime(time_df.index)
    
    # Sort by date
    time_df = time_df.sort_index()
    
    # Resample to quarterly frequency for better visualization
    quarterly_df = time_df.resample('Q').mean()
    
    return quarterly_df, top_cond_vars

def analyze_latest_period(sorted_weights, time_df):
    """Analyze the latest time period data"""
    # Get date columns (all columns except Interaction, Factor, ConditioningVar, and AvgAbsWeight)
    date_columns = sorted_weights.columns[3:-1]
    
    # Get the latest date (last column before AvgAbsWeight)
    latest_date = date_columns[-1]
    print(f"\nAnalyzing latest period: {latest_date}")
    
    # Get top factors by weight in the latest period
    factor_weights = {}
    for factor in sorted_weights['Factor'].unique():
        subset = sorted_weights[sorted_weights['Factor'] == factor]
        factor_weights[factor] = subset[latest_date].abs().sum()
    
    top_factors_latest = pd.Series(factor_weights).sort_values(ascending=False).head(10)
    print("\nTop 10 factors by total weight in latest period:")
    print(top_factors_latest)
    
    # Get top conditioning variables in the latest period
    cond_var_weights = {}
    for cond_var in sorted_weights['ConditioningVar'].unique():
        subset = sorted_weights[sorted_weights['ConditioningVar'] == cond_var]
        cond_var_weights[cond_var] = subset[latest_date].abs().sum()
    
    top_cond_vars_latest = pd.Series(cond_var_weights).sort_values(ascending=False).head(10)
    print("\nTop 10 conditioning variables by total weight in latest period:")
    print(top_cond_vars_latest)
    
    # Get top factor-conditioning variable interactions in the latest period
    interaction_weights = sorted_weights[['Factor', 'ConditioningVar', latest_date]].copy()
    interaction_weights['AbsWeight'] = interaction_weights[latest_date].abs()
    top_interactions_latest = interaction_weights.nlargest(10, 'AbsWeight')
    print("\nTop 10 factor-conditioning variable interactions in latest period:")
    print(top_interactions_latest[['Factor', 'ConditioningVar', 'AbsWeight']])
    
    # Get time series of top conditioning variables for the latest year
    if isinstance(time_df.index, pd.DatetimeIndex):
        latest_year = time_df.iloc[-4:] if len(time_df) >= 4 else time_df.iloc[-len(time_df):]
        print("\nTrend of top conditioning variables in the latest year:")
        print(latest_year)
    
    return {
        'latest_date': latest_date,
        'top_factors': top_factors_latest,
        'top_cond_vars': top_cond_vars_latest,
        'top_interactions': top_interactions_latest[['Factor', 'ConditioningVar', 'AbsWeight']],
        'latest_year_trend': latest_year if 'latest_year' in locals() else None
    }

def plot_top_conditioning_vars(cond_var_importance, output_dir='.'):
    """Plot top conditioning variables by average absolute weight"""
    plt.figure(figsize=(12, 8))
    top_n = 15
    top_vars = cond_var_importance.head(top_n)
    
    ax = top_vars.plot(kind='barh')
    plt.title(f'Top {top_n} Conditioning Variables by Average Absolute Weight')
    plt.xlabel('Average Absolute Weight')
    plt.ylabel('Conditioning Variable')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'top_conditioning_variables.pdf')
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    
    return top_vars

def plot_time_variation(time_df, top_cond_vars, output_dir='.'):
    """Plot how conditioning variable importance changes over time"""
    # Convert index to datetime for better plotting
    time_df.index = pd.to_datetime(time_df.index)
    
    # Plot time series
    plt.figure(figsize=(15, 10))
    for cond_var in top_cond_vars[:5]:  # Plot top 5 for clarity
        plt.plot(time_df.index, time_df[cond_var], label=cond_var)
    
    plt.title('Importance of Top Conditioning Variables Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sum of Absolute Weights')
    plt.legend()
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'conditioning_variables_time_series.pdf')
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")
    
    # Create heatmap for all top variables
    plt.figure(figsize=(15, 10))
    
    # Resample to quarterly data for better visualization
    quarterly_df = time_df.resample('Q').mean()
    
    # Normalize the data for better visualization
    normalized_df = quarterly_df.div(quarterly_df.max(axis=0), axis=1)
    
    sns.heatmap(normalized_df[top_cond_vars[:10]].T, cmap='viridis', 
                xticklabels=[d.strftime('%Y-%m') for d in quarterly_df.index], 
                yticklabels=top_cond_vars[:10])
    plt.title('Relative Importance of Top Conditioning Variables Over Time')
    plt.xlabel('Date')
    plt.ylabel('Conditioning Variable')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'conditioning_variables_heatmap.pdf')
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

def analyze_factor_specific_conditioning(sorted_weights):
    """Analyze which conditioning variables are most important for each factor"""
    # Get top factors by average absolute weight
    top_factors = sorted_weights.groupby('Factor')['AvgAbsWeight'].mean().nlargest(5).index.tolist()
    
    factor_specific_importance = {}
    for factor in top_factors:
        # Get subset of data for this factor
        factor_subset = sorted_weights[sorted_weights['Factor'] == factor]
        
        # Group by conditioning variable and calculate average importance
        cond_var_importance = factor_subset.groupby('ConditioningVar')['AvgAbsWeight'].mean().sort_values(ascending=False)
        factor_specific_importance[factor] = cond_var_importance
    
    return factor_specific_importance, top_factors

def plot_factor_specific_conditioning(factor_specific_importance, top_factors, output_dir='.'):
    """Plot conditioning variable importance for specific factors"""
    for factor in top_factors:
        plt.figure(figsize=(12, 8))
        top_n = 10
        top_vars = factor_specific_importance[factor].head(top_n)
        
        ax = top_vars.plot(kind='barh')
        plt.title(f'Top {top_n} Conditioning Variables for {factor}')
        plt.xlabel('Average Absolute Weight')
        plt.ylabel('Conditioning Variable')
        plt.tight_layout()
        
        # Clean factor name for filename
        factor_name = factor.replace('/', '_').replace(' ', '_')
        output_file = os.path.join(output_dir, f'conditioning_vars_{factor_name}.pdf')
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")

def get_claude_analysis(analysis_data):
    """Get detailed analysis from LM Studio API"""
    
    # Load environment variables from .env file (keeping this for other potential env vars)
    dotenv.load_dotenv()

    # No need for API key with LM Studio as it's running locally
    print("Using local LM Studio API - no API key required")

    # Prepare the prompt with analysis data
    prompt = f"""
    You are writing a formal financial research paper analyzing conditioning variables in a factor timing model.
    
    DO NOT include any thinking or planning in your response. Just write the final analysis directly.
    Format your response with proper markdown headings, bullet points, and emphasis where appropriate.
    
    You are an expert in financial analysis and factor investing. I need you to analyze the following data about conditioning variables in a factor timing model.
    
    Top 10 conditioning variables by average absolute weight:
    {analysis_data['top_by_weight']}
    
    Top 10 conditioning variables by non-zero weight count:
    {analysis_data['top_by_count']}
    
    Factor-specific conditioning variables:
    {analysis_data['factor_specific']}
    
    Latest period analysis (as of {analysis_data['latest_data']['latest_date']}):
    
    Top factors by weight in latest period:
    {analysis_data['latest_data']['top_factors']}
    
    Top conditioning variables in latest period:
    {analysis_data['latest_data']['top_cond_vars']}
    
    Top factor-conditioning variable interactions in latest period:
    {analysis_data['latest_data']['top_interactions']}
    
    Based on this data, please provide:
    1. A comprehensive analysis of which conditioning variables appear most important overall and why they might matter for factor timing
    2. An explanation of how different factors are influenced by different conditioning variables and what economic intuition might explain these relationships
    3. Insights about the time-varying nature of conditioning variable importance and what this means for investors
    4. Practical implications for factor timing strategies based on these findings
    5. A specific analysis of the latest period data, highlighting which factors currently have high predictions and which conditioning variables are currently most important, including any notable shifts from historical patterns
    
    Please write in a professional, academic style suitable for a financial research paper.
    """
    
    # Check if LM Studio is available
    if not LM_STUDIO_AVAILABLE:
        print("Error: LM Studio API is not available. Please ensure LM Studio is running on http://localhost:1234")
        return None
    
    # Make the API call with retry logic
    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            # Create a request to LM Studio API
            data = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4000,
                "temperature": 0.7,
                "model": "qwen3-30b-a3b-mlx@8bit"  # Specify the model you want to use
            }
            
            response = requests.post("http://localhost:1234/v1/chat/completions", json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Remove the thinking part if it exists
                if content.startswith('<think>'):
                    thinking_end = content.find('</think>')
                    if thinking_end != -1:
                        content = content[thinking_end + 8:].strip()  # 8 is the length of '</think>'
                
                # Clean up the content and improve formatting
                # Add title
                formatted_content = "# Conditioning Variables Analysis for Factor Timing Model\n\n"
                
                # Add sections with proper markdown formatting
                sections = content.split('\n\n')
                for section in sections:
                    if section.strip():
                        # Check if this is a section header
                        if section.strip().startswith('#'):
                            formatted_content += section + "\n\n"
                        # If it contains a numbered list item, preserve it
                        elif any(line.strip().startswith(str(i) + '.') for i in range(1, 6) for line in section.split('\n')):
                            formatted_content += section + "\n\n"
                        # Otherwise, add a section header if it seems like a new topic
                        else:
                            # Look for potential section titles (first sentence of paragraph)
                            first_line = section.split('.')[0].strip()
                            if len(first_line) < 100 and first_line and not first_line.startswith('-'):
                                formatted_content += "## " + first_line + "\n\n"
                                rest_of_section = '.'.join(section.split('.')[1:]).strip()
                                if rest_of_section:
                                    formatted_content += rest_of_section + "\n\n"
                            else:
                                formatted_content += section + "\n\n"
                
                return formatted_content
            else:
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
            
        except Exception as e:
            last_error = e
            print(f"API call failed: {e}. Retrying in {attempt + 1} seconds...")
            time.sleep(attempt + 1)  # Exponential backoff
    
    print(f"Failed to get analysis from LM Studio API after {max_retries} attempts: {last_error}")
    return None
    
    return None

def main():
    """Main function to analyze conditioning variables"""
    print("=== ANALYZING CONDITIONING VARIABLES IMPORTANCE ===")
    
    # Create output directory for plots
    output_dir = 'conditioning_var_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_unrotated_weights()
    
    # Analyze weights
    print("\nAnalyzing weights...")
    sorted_weights = analyze_weights(df)
    
    # Analyze conditioning variables across all factors
    print("\nAnalyzing conditioning variables importance...")
    cond_var_importance, non_zero_df = analyze_conditioning_vars(sorted_weights)
    
    # Print top conditioning variables
    print("\nTop 10 conditioning variables by average absolute weight:")
    print(cond_var_importance.head(10))
    
    print("\nTop 10 conditioning variables by non-zero weight count:")
    print(non_zero_df.head(10))
    
    # Analyze time variation
    print("\nAnalyzing time variation of conditioning variables...")
    time_df, top_cond_vars = analyze_time_variation(sorted_weights)
    
    # Analyze factor-specific conditioning
    print("\nAnalyzing factor-specific conditioning variables...")
    factor_specific_importance, top_factors = analyze_factor_specific_conditioning(sorted_weights)
    
    # Print factor-specific conditioning variables
    factor_specific_text = ""
    for factor in top_factors:
        print(f"\nTop 5 conditioning variables for {factor}:")
        print(factor_specific_importance[factor].head(5))
        factor_specific_text += f"\nTop 5 conditioning variables for {factor}:\n"
        factor_specific_text += str(factor_specific_importance[factor].head(5)) + "\n"
    
    # Analyze latest period data
    latest_data = analyze_latest_period(sorted_weights, time_df)
    
    # Create plots
    print("\nCreating plots...")
    plot_top_conditioning_vars(cond_var_importance, output_dir)
    plot_time_variation(time_df, top_cond_vars, output_dir)
    plot_factor_specific_conditioning(factor_specific_importance, top_factors, output_dir)
    
    # Create summary Excel file
    print("\nCreating summary Excel file...")
    with pd.ExcelWriter(os.path.join(output_dir, 'conditioning_variables_summary.xlsx')) as writer:
        # Overall importance
        cond_var_importance.reset_index().to_excel(writer, sheet_name='Overall_Importance', index=False)
        
        # Non-zero counts
        non_zero_df.reset_index().to_excel(writer, sheet_name='NonZero_Counts', index=False)
        
        # Time variation
        time_df.to_excel(writer, sheet_name='Time_Variation')
        
        # Factor-specific importance
        for i, factor in enumerate(top_factors):
            sheet_name = f'Factor_{i+1}'
            factor_specific_importance[factor].reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
            
        # Top interactions
        sorted_weights.head(100)[['Interaction', 'Factor', 'ConditioningVar', 'AvgAbsWeight']].to_excel(
            writer, sheet_name='Top_Interactions', index=False)
        
        # Latest period data
        latest_period_df = pd.DataFrame({
            'Factor': latest_data['top_factors'].index,
            'Weight': latest_data['top_factors'].values
        })
        latest_period_df.to_excel(writer, sheet_name='Latest_Period_Factors', index=False)
        
        latest_cond_vars_df = pd.DataFrame({
            'ConditioningVar': latest_data['top_cond_vars'].index,
            'Weight': latest_data['top_cond_vars'].values
        })
        latest_cond_vars_df.to_excel(writer, sheet_name='Latest_Period_CondVars', index=False)
    
    # Prepare data for Claude analysis
    analysis_data = {
        'top_by_weight': str(cond_var_importance.head(10)),
        'top_by_count': str(non_zero_df.head(10)),
        'factor_specific': factor_specific_text,
        'latest_data': latest_data
    }
    
    # Get LM Studio analysis
    lm_studio_analysis = get_claude_analysis(analysis_data)
    
    if lm_studio_analysis:
        # Save LM Studio analysis to file
        with open(os.path.join(output_dir, 'lm_studio_analysis.md'), 'w') as f:
            f.write(lm_studio_analysis)
        
        # Print a preview of the analysis
        print("\nAnalysis from LM Studio (preview):")
        
        # Get the first few sections for preview
        sections = lm_studio_analysis.split('##')
        if len(sections) > 1:
            preview = sections[0] + '## ' + sections[1]
            if len(preview) > 800:
                preview = preview[:800] + '...'
        else:
            preview_length = 500  # Show first 500 characters
            preview = lm_studio_analysis[:preview_length] + "..." if len(lm_studio_analysis) > preview_length else lm_studio_analysis
            
        print(preview)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")

if __name__ == "__main__":
    main()
