import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from datetime import datetime
import os
import sys
import subprocess
import socket
import time

"""
INPUT FILE:
- Visualizer.xlsx: Contains two sheets ('Top60' and 'Mega60') with country weights data
  Each sheet has dates as index and countries as columns

OUTPUT FILES:
- country_comparison_*.html: HTML files with interactive country comparisons
- Various interactive visualizations displayed in Dash application
"""

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill the process using the specified port"""
    print(f"Attempting to kill process using port {port}...")
    
    # Different commands for different platforms
    if sys.platform.startswith('win'):
        cmd = f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{port}\') do taskkill /F /PID %a'
        os.system(cmd)
    else:  # macOS or Linux
        try:
            # Get the PID of the process using the port
            find_pid_cmd = f"lsof -i :{port} -t"
            result = subprocess.run(find_pid_cmd, shell=True, capture_output=True, text=True)
            pids = result.stdout.strip().split('\n')
            
            if pids and pids[0]:
                # Get the current process ID to avoid killing ourselves
                current_pid = os.getpid()
                
                for pid in pids:
                    if pid and int(pid) != current_pid:
                        print(f"Killing process with PID {pid} on port {port}")
                        kill_cmd = f"kill -9 {pid}"
                        subprocess.run(kill_cmd, shell=True)
                
                # Wait a moment for the port to be freed
                time.sleep(1)
            else:
                print(f"No process found using port {port}")
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")

def find_available_port(start_port=8050, max_attempts=10):
    """Find an available port starting from start_port"""
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        if not is_port_in_use(port):
            return port
        
        print(f"Port {port} is in use, trying port {port+1}...")
        port += 1
        attempts += 1
    
    print(f"Could not find an available port after {max_attempts} attempts. Please close some applications and try again.")
    sys.exit(1)

# Read the Excel files
def read_portfolio_data(file_path, sheet_name):
    """Read portfolio data from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
        # Convert index to datetime if it's not already
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error reading {file_path} (sheet: {sheet_name}): {e}")
        return None

# Main comparison class
class CountryWeightsComparison:
    def __init__(self, file_path='Visualizer.xlsx', sheet1='Top60', sheet2='Mega60'):
        """
        Initialize the comparison using a single Excel file with two sheets
        
        Parameters:
        -----------
        file_path : str
            Path to the Excel file containing country weights data
        sheet1 : str
            Name of the sheet containing Top60 portfolio data
        sheet2 : str
            Name of the sheet containing Mega60 portfolio data
        """
        self.portfolio1 = read_portfolio_data(file_path, sheet_name=sheet1)
        self.portfolio2 = read_portfolio_data(file_path, sheet_name=sheet2)
        
        if self.portfolio1 is None or self.portfolio2 is None:
            raise ValueError(f"Failed to load one or both portfolio sheets from {file_path}")
        
        # Store names for reference
        self.sheet1_name = sheet1
        self.sheet2_name = sheet2
        
        # Ensure both portfolios have the same countries
        self.common_countries = sorted(set(self.portfolio1.columns) & set(self.portfolio2.columns))
        self.all_countries = sorted(set(self.portfolio1.columns) | set(self.portfolio2.columns))
        
        print(f"Portfolio 1 ({sheet1}) shape: {self.portfolio1.shape}")
        print(f"Portfolio 2 ({sheet2}) shape: {self.portfolio2.shape}")
        print(f"Common countries: {len(self.common_countries)}")
        print(f"Total unique countries: {len(self.all_countries)}")
        
    def calculate_statistics(self, data, country):
        """Calculate statistics for a specific country"""
        if country not in data.columns:
            return {
                'avg_weight': 0,
                'max_weight': 0,
                'min_weight': 0,
                'std_dev': 0,
                'non_zero_months': 0,
                'total_months': len(data),
                'non_zero_pct': 0
            }
        
        country_data = data[country]
        non_zero_data = country_data[country_data > 0]
        
        return {
            'avg_weight': country_data.mean() * 100,  # Convert to percentage
            'max_weight': country_data.max() * 100,
            'min_weight': country_data.min() * 100,
            'std_dev': country_data.std() * 100,
            'non_zero_months': len(non_zero_data),
            'total_months': len(country_data),
            'non_zero_pct': len(non_zero_data) / len(country_data) * 100
        }
    
    def create_country_comparison(self, country, date_range='all'):
        """Create comparison chart for a specific country"""
        # Filter date range
        portfolio1_data = self._filter_date_range(self.portfolio1, date_range)
        portfolio2_data = self._filter_date_range(self.portfolio2, date_range)
        
        # Get data for the country
        p1_data = portfolio1_data[country] * 100 if country in portfolio1_data.columns else pd.Series(0, index=portfolio1_data.index)
        p2_data = portfolio2_data[country] * 100 if country in portfolio2_data.columns else pd.Series(0, index=portfolio2_data.index)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Scatter(
            x=p1_data.index,
            y=p1_data.values,
            mode='lines',
            name=self.sheet1_name,
            line=dict(color='blue', width=2),
            hovertemplate=f'Date: %{{x}}<br>Weight: %{{y:.3f}}%<br>Portfolio: {self.sheet1_name}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=p2_data.index,
            y=p2_data.values,
            mode='lines',
            name=self.sheet2_name,
            line=dict(color='red', width=2),
            hovertemplate=f'Date: %{{x}}<br>Weight: %{{y:.3f}}%<br>Portfolio: {self.sheet2_name}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Country Weights Comparison: {country}',
            xaxis_title='Date',
            yaxis_title='Weight (%)',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(x=0.7, y=0.95),
            height=500
        )
        
        # Add statistics
        stats1 = self.calculate_statistics(portfolio1_data, country)
        stats2 = self.calculate_statistics(portfolio2_data, country)
        
        # Add annotation with statistics
        annotation_text = f"""
        <b>{self.sheet1_name} Statistics:</b><br>
        Avg: {stats1['avg_weight']:.3f}%<br>
        Max: {stats1['max_weight']:.3f}%<br>
        Non-zero: {stats1['non_zero_pct']:.1f}%<br>
        <br>
        <b>{self.sheet2_name} Statistics:</b><br>
        Avg: {stats2['avg_weight']:.3f}%<br>
        Max: {stats2['max_weight']:.3f}%<br>
        Non-zero: {stats2['non_zero_pct']:.1f}%
        """
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=annotation_text,
            align='left',
            showarrow=False,
            bordercolor='black',
            borderwidth=1,
            bgcolor='rgba(255,255,255,0.9)'
        )
        
        return fig
    
    def create_all_countries_summary(self, date_range='all'):
        """Create summary visualization for all countries"""
        # Filter date range
        portfolio1_data = self._filter_date_range(self.portfolio1, date_range)
        portfolio2_data = self._filter_date_range(self.portfolio2, date_range)
        
        # Calculate average weights for all countries
        avg_weights = []
        
        for country in self.all_countries:
            p1_avg = portfolio1_data[country].mean() * 100 if country in portfolio1_data.columns else 0
            p2_avg = portfolio2_data[country].mean() * 100 if country in portfolio2_data.columns else 0
            avg_weights.append({
                'Country': country,
                f'{self.sheet1_name}': p1_avg,
                f'{self.sheet2_name}': p2_avg,
                'Difference': p2_avg - p1_avg
            })
        
        avg_weights_df = pd.DataFrame(avg_weights)
        avg_weights_df = avg_weights_df.sort_values(self.sheet1_name, ascending=False)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Average Weights by Country', f'Weight Differences ({self.sheet2_name} - {self.sheet1_name})',
                          'Non-Zero Month Percentage', 'Maximum Weights'],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Average weights comparison
        fig.add_trace(
            go.Bar(name=self.sheet1_name, x=avg_weights_df['Country'], y=avg_weights_df[self.sheet1_name],
                  marker_color='blue', opacity=0.7),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name=self.sheet2_name, x=avg_weights_df['Country'], y=avg_weights_df[self.sheet2_name],
                  marker_color='red', opacity=0.7),
            row=1, col=1
        )
        
        # Weight differences
        fig.add_trace(
            go.Bar(x=avg_weights_df['Country'], y=avg_weights_df['Difference'],
                  marker_color=avg_weights_df['Difference'].apply(lambda x: 'green' if x > 0 else 'red'),
                  showlegend=False),
            row=1, col=2
        )
        
        # Calculate non-zero percentages and max weights
        non_zero_data = []
        max_weights_data = []
        
        for country in self.all_countries:
            stats1 = self.calculate_statistics(portfolio1_data, country)
            stats2 = self.calculate_statistics(portfolio2_data, country)
            
            non_zero_data.append({
                'Country': country,
                f'{self.sheet1_name}': stats1['non_zero_pct'],
                f'{self.sheet2_name}': stats2['non_zero_pct']
            })
            
            max_weights_data.append({
                'Country': country,
                f'{self.sheet1_name}': stats1['max_weight'],
                f'{self.sheet2_name}': stats2['max_weight']
            })
        
        non_zero_df = pd.DataFrame(non_zero_data).sort_values(self.sheet1_name, ascending=False)
        max_weights_df = pd.DataFrame(max_weights_data).sort_values(self.sheet1_name, ascending=False)
        
        # Non-zero percentages
        fig.add_trace(
            go.Bar(name=self.sheet1_name, x=non_zero_df['Country'], y=non_zero_df[self.sheet1_name],
                  marker_color='blue', opacity=0.7, showlegend=False),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name=self.sheet2_name, x=non_zero_df['Country'], y=non_zero_df[self.sheet2_name],
                  marker_color='red', opacity=0.7, showlegend=False),
            row=2, col=1
        )
        
        # Maximum weights
        fig.add_trace(
            go.Bar(name=self.sheet1_name, x=max_weights_df['Country'], y=max_weights_df[self.sheet1_name],
                  marker_color='blue', opacity=0.7, showlegend=False),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name=self.sheet2_name, x=max_weights_df['Country'], y=max_weights_df[self.sheet2_name],
                  marker_color='red', opacity=0.7, showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            title=f'All Countries Portfolio Comparison Summary ({date_range})',
            height=1000,
            showlegend=True,
            barmode='group'
        )
        
        return fig
    
    def create_correlation_heatmap(self, date_range='all'):
        """Create correlation heatmap between portfolios"""
        # Filter date range
        portfolio1_data = self._filter_date_range(self.portfolio1, date_range)
        portfolio2_data = self._filter_date_range(self.portfolio2, date_range)
        
        correlations = []
        
        for country in self.common_countries:
            p1_data = portfolio1_data[country]
            p2_data = portfolio2_data[country]
            
            # Align the data
            aligned_data = pd.DataFrame({
                'p1': p1_data,
                'p2': p2_data
            }).dropna()
            
            if len(aligned_data) > 10:  # Need sufficient data for correlation
                corr = aligned_data['p1'].corr(aligned_data['p2'])
                correlations.append({
                    'Country': country,
                    'Correlation': corr
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        
        # Create heatmap
        fig = go.Figure(data=go.Bar(
            x=corr_df['Correlation'],
            y=corr_df['Country'],
            orientation='h',
            marker=dict(
                color=corr_df['Correlation'],
                colorscale='RdBu',
                cmin=-1,
                cmax=1,
                colorbar=dict(title='Correlation')
            )
        ))
        
        fig.update_layout(
            title=f'Country Weight Correlations: {self.sheet1_name} vs {self.sheet2_name} ({date_range})',
            xaxis_title='Correlation',
            yaxis_title='Country',
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def _filter_date_range(self, data, date_range):
        """Filter data based on date range"""
        if date_range == 'all':
            return data
        
        end_date = data.index.max()
        
        if date_range == 'last5y':
            start_date = end_date - pd.DateOffset(years=5)
        elif date_range == 'last10y':
            start_date = end_date - pd.DateOffset(years=10)
        elif date_range == 'last20y':
            start_date = end_date - pd.DateOffset(years=20)
        else:
            return data
        
        return data[data.index >= start_date]
    
    def create_interactive_dashboard(self):
        """Create an interactive dashboard with country selection"""
        import dash
        from dash import dcc, html, Input, Output, State
        import dash_bootstrap_components as dbc
        
        # Initialize Dash app
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Country Portfolio Weights Comparison", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Label("Select Country:"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in self.all_countries],
                        value=self.all_countries[0],
                        clearable=False
                    )
                ], width=6),
                
                dbc.Col([
                    html.Label("Date Range:"),
                    dcc.Dropdown(
                        id='date-range-dropdown',
                        options=[
                            {'label': 'All Dates', 'value': 'all'},
                            {'label': 'Last 5 Years', 'value': 'last5y'},
                            {'label': 'Last 10 Years', 'value': 'last10y'},
                            {'label': 'Last 20 Years', 'value': 'last20y'}
                        ],
                        value='all',
                        clearable=False
                    )
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id='tabs', value='country-comparison', children=[
                        dcc.Tab(label='Country Comparison', value='country-comparison'),
                        dcc.Tab(label='All Countries Summary', value='all-countries'),
                        dcc.Tab(label='Correlation Analysis', value='correlation')
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dcc.Loading(
                        dcc.Graph(id='main-graph'),
                        type="default"
                    )
                ])
            ], className="mt-4"),
            
            dbc.Row([
                dbc.Col([
                    html.Div(id='statistics-panel')
                ])
            ], className="mt-4")
        ], fluid=True)
        
        # Define callbacks
        @app.callback(
            [Output('main-graph', 'figure'),
             Output('statistics-panel', 'children')],
            [Input('tabs', 'value'),
             Input('country-dropdown', 'value'),
             Input('date-range-dropdown', 'value')]
        )
        def update_graph(tab, country, date_range):
            if tab == 'country-comparison':
                fig = self.create_country_comparison(country, date_range)
                
                # Create statistics panel
                portfolio1_data = self._filter_date_range(self.portfolio1, date_range)
                portfolio2_data = self._filter_date_range(self.portfolio2, date_range)
                
                stats1 = self.calculate_statistics(portfolio1_data, country)
                stats2 = self.calculate_statistics(portfolio2_data, country)
                
                stats_panel = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4(f"{self.sheet1_name} Statistics", className="text-white"), className="bg-primary"),
                            dbc.CardBody([
                                html.P(f"Average Weight: {stats1['avg_weight']:.3f}%"),
                                html.P(f"Maximum Weight: {stats1['max_weight']:.3f}%"),
                                html.P(f"Minimum Weight: {stats1['min_weight']:.3f}%"),
                                html.P(f"Standard Deviation: {stats1['std_dev']:.3f}%"),
                                html.P(f"Non-zero Months: {stats1['non_zero_months']} / {stats1['total_months']} ({stats1['non_zero_pct']:.1f}%)")
                            ])
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H4(f"{self.sheet2_name} Statistics", className="text-white"), className="bg-danger"),
                            dbc.CardBody([
                                html.P(f"Average Weight: {stats2['avg_weight']:.3f}%"),
                                html.P(f"Maximum Weight: {stats2['max_weight']:.3f}%"),
                                html.P(f"Minimum Weight: {stats2['min_weight']:.3f}%"),
                                html.P(f"Standard Deviation: {stats2['std_dev']:.3f}%"),
                                html.P(f"Non-zero Months: {stats2['non_zero_months']} / {stats2['total_months']} ({stats2['non_zero_pct']:.1f}%)")
                            ])
                        ])
                    ], width=6)
                ])
                
                return fig, stats_panel
                
            elif tab == 'all-countries':
                fig = self.create_all_countries_summary(date_range)
                return fig, html.Div()
                
            elif tab == 'correlation':
                fig = self.create_correlation_heatmap(date_range)
                return fig, html.Div()
        
        return app
    
    def save_static_comparison(self, country, filename=None, date_range='all'):
        """Save a static comparison chart for a specific country"""
        fig = self.create_country_comparison(country, date_range)
        
        if filename is None:
            filename = f"country_comparison_{country}_{date_range}.html"
        
        fig.write_html(filename)
        print(f"Saved comparison chart to {filename}")
    
    def export_all_comparisons(self, output_dir='country_comparisons'):
        """Export comparison charts for all countries"""
        os.makedirs(output_dir, exist_ok=True)
        
        for country in self.all_countries:
            for date_range in ['all', 'last5y', 'last10y', 'last20y']:
                filename = os.path.join(output_dir, f"{country}_{date_range}.html")
                self.save_static_comparison(country, filename, date_range)
        
        print(f"Exported all comparisons to {output_dir}")


# Main execution
if __name__ == "__main__":
    # Find an available port
    PORT = find_available_port(start_port=8050)
    print(f"Using port {PORT} for Dash application")
    
    # Initialize the comparison tool with the single Excel file and two sheet names
    print("Initializing CountryWeightsComparison...")
    comparison = CountryWeightsComparison('Visualizer.xlsx', 'Top60', 'Mega60')
    
    # Option 1: Run the interactive dashboard
    print("Creating interactive dashboard...")
    app = comparison.create_interactive_dashboard()
    print(f"Starting Dash server on port {PORT}...")
    app.run(debug=True, port=PORT)
    
    # Option 2: Create and save static visualizations
    # # Save a specific country comparison
    # comparison.save_static_comparison('Singapore', 'singapore_comparison.html')
    # 
    # # Create all countries summary
    # fig = comparison.create_all_countries_summary()
    # fig.show()
    # 
    # # Create correlation heatmap
    # corr_fig = comparison.create_correlation_heatmap()
    # corr_fig.show()
    # 
    # # Export all comparisons
    # comparison.export_all_comparisons()