import datetime

import os, sys, glob, time, threading, gc
import pandas as pd, numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import webview

from pathlib import Path

# Local modules
from utils.constants import DEBUG, MAX_DATA_AGE_SECONDS, PUNCH_STICKING_COLUMNS
from utils.data_loader import find_latest_csv, save_data_to_database
from utils.debug import debug_print
from layout.main_view import dashboard_layout
from layout.historical_view import historical_data_layout

# Configuration settings
# # Initialize global variables
df = pd.DataFrame()
last_csv_file = None
global_start_time = None
last_file_size = 0
last_row_count = 0
last_db_save_time = 0
current_folder = ""
pywebview_window = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id="folder-path", data=""),
    dcc.Store(id="selected-force", data=""),
    dcc.Interval(id='graph-update', interval=1000, n_intervals=0, disabled=False),
    dcc.Interval(id='clock-update', interval=1000, n_intervals=0),
    html.Div(id='page-content'),
    html.Div(id="dummy-output", style={"display": "none"})
])

# Switches between main dashboard and historical view layouts based on the current URL.
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    State('folder-path', 'data'),
)
def render_page(pathname, folder_path):
    global current_folder
    current_folder = folder_path

    if pathname == '/history':
        return historical_data_layout(folder_path)
    else:
        return dashboard_layout(folder_path)

# Redirects the app to the /history route when the “History” button is clicked.
@app.callback(
    Output("url", "pathname", allow_duplicate=True),
    Input("history-button", "n_clicks"),
    prevent_initial_call=True
)
def navigate_to_history(n_clicks):
    if n_clicks:
        return "/history"
    raise dash.exceptions.PreventUpdate

# Loads a CSV archive file and updates the historical data graph.
@app.callback(
    Output("historical-data-graph", "figure"),
    Input("archive-file-selector", "value"),
    Input("history-time-window", "value")
)
def update_historical_graph(selected_file, time_window):
    if not selected_file:
        return {
            'data': [],
            'layout': {
                'title': 'No archive file selected',
                'plot_bgcolor': '#1f2937',
                'paper_bgcolor': '#1f2937',
                'font': {'color': '#d1d5db'}
            }
        }
    
    try:
        # Load historical data
        hist_data = pd.read_csv(selected_file)
        debug_print(f"Loaded historical data: {len(hist_data)} rows")
        
        # Filter data by time window if needed
        if time_window > 0 and 'relative_time' in hist_data.columns:
            max_time = hist_data['relative_time'].max()
            min_time = max(0, max_time - time_window)
            hist_data = hist_data[hist_data['relative_time'] >= min_time]
            debug_print(f"Filtered to {len(hist_data)} rows in time window")
        
        # Create traces similar to your live-graph
        traces = []
        colors = {
            'compression_force_1': '#1E88E5',  # Blue
            'compression_force_2': '#FFC107',  # Amber
            'ejection_force': '#4CAF50',       # Green
            'pre_compression_force': '#E91E63'  # Pink
        }

        for col, color in colors.items():
            if col in hist_data.columns:
                hist_data[col] = pd.to_numeric(hist_data[col], errors='coerce').fillna(0)
                
                if hist_data[col].sum() == 0:
                    continue
                    
                if 'datetime' in hist_data.columns and not hist_data['datetime'].isna().all():
                    hist_data['datetime'] = pd.to_datetime(hist_data['datetime'], errors='coerce')
                    traces.append(
                        go.Scatter(
                            x=hist_data['datetime'],
                            y=hist_data[col].tolist(),
                            mode='lines',
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2, color=color)
                        )
                    )
                else:
                    traces.append(
                        go.Scatter(
                            x=hist_data['relative_time'].tolist(),
                            y=hist_data[col].tolist(),
                            mode='lines',
                            name=col.replace('_', ' ').title(),
                            line=dict(width=2, color=color)
                        )
                    )
        
        # Create layout
        if 'datetime' in hist_data.columns and not hist_data['datetime'].isna().all():
            layout = go.Layout(
                xaxis=dict(
                    title='Time',
                    type='date',
                    tickformat='%H:%M:%S.%L',
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title='Force (N)', 
                    autorange=True, 
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                margin=dict(l=60, r=10, t=20, b=30),
                hovermode='closest',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#d1d5db'),
                title=f"Historical Data: {os.path.basename(selected_file)}",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            layout = go.Layout(
                xaxis=dict(
                    title='Seconds from Start', 
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title='Force (N)', 
                    autorange=True, 
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                margin=dict(l=60, r=10, t=20, b=30),
                hovermode='closest',
                plot_bgcolor='#1f2937',
                paper_bgcolor='#1f2937',
                font=dict(color='#d1d5db'),
                title=f"Historical Data: {os.path.basename(selected_file)}",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
        return {'data': traces, 'layout': layout}
    except Exception as e:
        debug_print(f"Error loading historical data: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
        return {
            'data': [],
            'layout': {
                'title': f'Error loading data: {str(e)}',
                'plot_bgcolor': '#1f2937',
                'paper_bgcolor': '#1f2937',
                'font': {'color': '#d1d5db'}
            }
        }

# Updates the real-time clock in the historical view every second.
@app.callback(
    Output('history-current-time', 'children'),
    Input('clock-update', 'n_intervals')
)
def update_history_clock(n):
    return f"Historical View: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Updates the real-time clock in the main dashboard every second.
@app.callback(
    Output('current-time', 'children'),
    Input('clock-update', 'n_intervals')
)
def update_clock(n):
    return f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Function to generate punch stats figure
def generate_punch_stats_figure(force_type: str, data_df: pd.DataFrame) -> go.Figure:

    CONFIG = {
        "pre_compression": {
            "punch_cols": ["precomppunchno"],
            "force_cols": ["pre_compression_force"],
            "title": "Force by Punch Number"
        },
        "compression": {
            "punch_cols": ["compr1punchno", "compr2punchno"],
            "force_cols": ["compression_force_1", "compression_force_2"],
            "title": "Force by Punch Number"
        },
        "ejection": {
            "punch_cols": ["ejectpunchno"],
            "force_cols": ["ejection_force"],
            "title": "Force by Punch Number"
        }
    }
    
    # Default figure if no data or force type is invalid
    def default_figure(title="No data available"):
        return go.Figure(
            layout=go.Layout(
                title=title,
                plot_bgcolor="#1f2937",
                paper_bgcolor="#1f2937",
                font=dict(color="#d1d5db"),
                xaxis=dict(title="Punch Number", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="Force (N)", showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(l=40, r=10, t=30, b=30),
            )
        )

    if force_type not in CONFIG or data_df.empty:
        return default_figure()

    # Get the punch columns and force columns based on the force type
    config = CONFIG[force_type]
    punch_cols, force_cols, title = config["punch_cols"], config["force_cols"], config["title"]

    punch_forces = {p: [] for p in range(1,9)}
    
    try:
        for _, row in data_df.iterrows():

            row_forces = [float(row[col]) for col in force_cols if col in row and pd.notna(row[col])]
            # row_forces = [pd.numeric(row.get(col), errors='coerce') for col in force_cols]
            
            if not row_forces:
                continue
                
            row_force = sum(row_forces) / len(row_forces)
            
            combined_punches = []
            for pcol in punch_cols:
                if pcol not in row or pd.isna(row[pcol]) or row[pcol] == "":
                    continue    
                try:
                    punch_value = row[pcol]
                    if isinstance(punch_value, str):
                        for p in punch_value.split(","):
                            p = p.strip()
                            if p and p.isdigit():
                                punch_num = int(p)
                                # Only include punch numbers 1-8, skip zeros
                                if 1 <= punch_num <= 8:
                                    combined_punches.append(punch_num)
                    else:  # Handle numeric values directly
                        punch_num = int(punch_value)
                        if 1 <= punch_num <= 8:  # Skip zeros and validate range
                            combined_punches.append(punch_num)
                except Exception as e:
                    debug_print(f"Error parsing punch number from {row[pcol]}: {str(e)}")
            
            for punch in combined_punches:
                punch_forces.setdefault(punch, []).append(row_force)

    except Exception as e:
        debug_print(f"Error collecting punch force data: {str(e)}")
        return default_figure()

    punch_numbers = sorted([p for p in punch_forces.keys() if 1 <= p <= 8])

    if not punch_numbers:
        return default_figure()
    
    max_avg_forces = []
    max_forces = []
    
    window_size = 5

    for p in punch_numbers:
        values = punch_forces[p]
        
        if values:
            # Maximum instantaneous force
            max_val = max(values)
            max_forces.append(max_val)
            
            # Calculate maximum moving average
            if len(values) >= window_size:
                # Calculate moving averages with window_size
                moving_avgs = []
                for i in range(len(values) - window_size + 1):
                    window_avg = sum(values[i:i+window_size]) / window_size
                    moving_avgs.append(window_avg)
                
                max_avg = max(moving_avgs)
            else:
                # If not enough data points, use regular average
                max_avg = sum(values) / len(values)
                
            max_avg_forces.append(max_avg)
        else:
            max_forces.append(0)
            max_avg_forces.append(0)

    x_labels = [str(p) for p in punch_numbers]

    max_diff = []
    for max_val, max_avg in zip(max_forces, max_avg_forces):
        diff = max(0, max_val - max_avg)  # Ensure no negative differences
        max_diff.append(diff)

    # Set color based on force type
    colors = {
        "pre_compression": '#E91E63',  # Pink
        "compression": '#1E88E5',      # Blue
        "ejection": '#4CAF50',         # Green
    }
    
    base_color = colors.get(force_type, '#38B2AC')  # Default to teal if not found

    # Create text labels with reduced precision for cleaner display
    avg_text = [f"{avg:.0f}" for avg in max_avg_forces]
    max_text = [f"{max_val:.0f}" for max_val in max_forces]

    base_bar = go.Bar(
        x=x_labels, 
        y=max_avg_forces, 
        text=avg_text,
        textposition='auto',
        name="Avg",
        marker=dict(color=base_color),
        hovertemplate="Punch %{x}<br>Avg: %{y:.1f} N<extra></extra>"
    )

    top_bar = go.Bar(
        x=x_labels, 
        y=max_diff, 
        text=max_text,
        textposition='outside',
        name="Max",
        marker=dict(color='#F59E0B'),  # Amber color
        hovertemplate="Punch %{x}<br>Max: %{customdata:.1f} N<extra></extra>",
        customdata=max_forces,  
    )

    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(size=10)  
        ),
        xaxis=dict(
            title=dict(
                text="Punch Number",
                font=dict(size=10),  
                standoff=5, 
            ),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            categoryorder='array',
            categoryarray=x_labels
        ),
        yaxis=dict(
            title=dict(
                text="Force (N)",
                font=dict(size=10),  
                standoff=5,  
            ),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            range=[0, max(max(max_forces) * 1.25, 1)] 
        ),
        plot_bgcolor='#1f2937',
        paper_bgcolor='#1f2937',
        font=dict(color='#d1d5db'),
        barmode='stack',
        uniformtext=dict(
            mode='hide',
            minsize=8
        ),
        margin=dict(l=40, r=10, t=40, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)  
        )
    )

    return go.Figure(data=[base_bar, top_bar], layout=layout)

@app.callback(
    [Output('live-graph', 'figure'),
     Output('punch-status-indicator', 'children'),
     Output('pre-compression-stats', 'children'),
     Output('compression-stats', 'children'),
     Output('ejection-stats', 'children'),
     Output('pre-compression-graph', 'figure'),
     Output('compression-graph', 'figure'),
     Output('ejection-graph', 'figure'),
     *[Output(f'punch-status-{i}', 'children') for i in range(1, 9)],
     *[Output(f'punch-status-{i}', 'className') for i in range(1, 9)]],
    [Input('graph-update', 'n_intervals'),
     Input("time-window-input", "value"),
     Input('show-tf-stats', 'value')],
    [State('folder-path', 'data')]
)
def update_graphs(n_intervals, time_window, show_tf_stats, chosen_folder):
    global df, last_db_save_time, last_csv_file, global_start_time, last_file_size, last_row_count

    force_columns = {
        'compr1data': 'compression_force_1',
        'compr2data': 'compression_force_2', 
        'ejectdata': 'ejection_force',
        'precompdata': 'pre_compression_force',
        'COMPR1DATA': 'compression_force_1',
        'COMPR2DATA': 'compression_force_2',
        'EJECTDATA': 'ejection_force',
        'PRECOMPDATA': 'pre_compression_force',
        'compression1': 'compression_force_1',
        'compression2': 'compression_force_2',
        'ejection': 'ejection_force',
        'precompression': 'pre_compression_force',
    }
    
    ######## Default values for the graphs and stats ########
    default_figure = {
        'data': [],
        'layout': {
            'plot_bgcolor': '#1f2937',
            'paper_bgcolor': '#1f2937',
            'font': {'color': '#d1d5db'},
            'xaxis': {
                'title': 'Seconds from Start',
                'showgrid': True,
                'gridcolor': 'rgba(255,255,255,0.1)'
            },
            'yaxis': {
                'title': 'Force (N)',
                'showgrid': True,
                'gridcolor': 'rgba(255,255,255,0.1)'
            },
            'margin': {'l': 60, 'r': 10, 't': 5, 'b': 20},
            'hovermode': 'closest'
        }
    }

    default_stats = "No data available"
    
    # Default empty punch stats figures
    default_punch_stats_figure = {
        'data': [],
        'layout': {
            'title': 'No Data',
            'plot_bgcolor': '#1f2937',
            'paper_bgcolor': '#1f2937',
            'font': {'color': '#d1d5db'},
            'xaxis': {'showgrid': False},
            'yaxis': {'showgrid': False},
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 30},
        }
    }
    
    punch_texts = [f"Punch {i}" for i in range(1, 9)]
    punch_classes = ["status-badge badge-normal" for i in range(1, 9)]
    #########################################################


    # Check if the folder is selected
    if not chosen_folder:
        return (default_figure, "No folder selected. Please choose a folder first.",
                default_stats, default_stats, default_stats,
                default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                *punch_texts, *punch_classes)

    # Find the latest CSV file in the selected folder
    latest_csv = find_latest_csv(chosen_folder)
    if not latest_csv:
        return (default_figure, "No CSV files found in the selected folder.",
                default_stats, default_stats, default_stats,
                default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                *punch_texts, *punch_classes)

    current_file_size = os.path.getsize(latest_csv)
    file_changed = (latest_csv != last_csv_file) or (current_file_size != last_file_size)

    try:
        if file_changed:
            debug_print(f"Reading new CSV file: {latest_csv}")
            
            #### Preprocessing steps ####
            try:
                new_data = pd.read_csv(latest_csv, encoding='utf-8')
            except UnicodeDecodeError:
                new_data = pd.read_csv(latest_csv, encoding='latin1')

            
            debug_print(f"CSV columns (original): {new_data.columns.tolist()}")
            
            new_data.columns = new_data.columns.str.lower().str.strip()
            debug_print(f"CSV columns (lowercase): {new_data.columns.tolist()}")
            
            if global_start_time is None and 'timestamp' in new_data.columns:
                try:
                    new_data['timestamp'] = pd.to_numeric(new_data['timestamp'], errors='coerce')
                    global_start_time = float(new_data['timestamp'].min())
                    debug_print(f"Setting global start time to: {global_start_time}")
                except Exception as e:
                    debug_print(f"Error setting global start time: {str(e)}")
                    global_start_time = 0
            
            if 'timestamp' in new_data.columns:
                new_data['timestamp'] = pd.to_numeric(new_data['timestamp'], errors='coerce')
                new_data['relative_time'] = (new_data['timestamp'] - global_start_time) / 1000  # Convert ms to seconds
                new_data['absolute_time'] = new_data['timestamp']
            else:
                debug_print("ERROR: No timestamp column found in CSV! This will cause graphing issues.")
            
            # Handle actualtime(ms) column from the dummy generator
            if 'actualtime(ms)' in new_data.columns:
                try:
                    new_data['datetime'] = pd.to_datetime(new_data['actualtime(ms)'], errors='coerce')
                    new_data['time_str'] = new_data['datetime'].dt.strftime('%H:%M:%S.%f')
                    debug_print("Successfully processed actualtime(ms) column")
                except Exception as e:
                    debug_print(f"Error processing actualtime(ms) column: {str(e)}")
                    new_data['datetime'] = pd.NaT
                    new_data['time_str'] = ""
            elif 'actualtime' in new_data.columns:
                try:
                    new_data['datetime'] = pd.to_datetime(new_data['actualtime'], errors='coerce')
                    new_data['time_str'] = new_data['datetime'].dt.strftime('%H:%M:%S.%f')
                    debug_print("Successfully processed actualtime column")
                except Exception as e:
                    debug_print(f"Error processing actualtime column: {str(e)}")
                    new_data['datetime'] = pd.NaT
                    new_data['time_str'] = ""
            else:
                debug_print("No time column found, will use relative time")
                new_data['datetime'] = pd.NaT
                new_data['time_str'] = ""
            
            # Map CSV column names to our expected dataframe column names
            for source_col, target_col in force_columns.items():
                if source_col in new_data.columns:
                    new_data[target_col] = pd.to_numeric(new_data[source_col], errors='coerce').fillna(0)
                else:
                    if target_col not in new_data.columns:
                        new_data[target_col] = 0
                        debug_print(f"WARNING: Required column {source_col} missing from CSV!")
            
            # Map punch number columns
            punch_columns = ['compr1punchno', 'compr2punchno', 'ejectpunchno', 'precomppunchno']
            for col in punch_columns:
                if col not in new_data.columns:
                    new_data[col] = ""
            
            # Create a synthetic 'sticking' column combining all punch-specific sticking columns if they exist
            punch_sticking_columns = [col for col in new_data.columns if 'punch' in col and 'sticking' in col]
            if punch_sticking_columns:
                debug_print(f"Found punch-specific sticking columns: {punch_sticking_columns}")
                # Use the maximum sticking value across all punch columns for each row
                new_data['sticking'] = new_data[punch_sticking_columns].max(axis=1)
                debug_print("Created synthetic 'sticking' column from punch-specific values")
            elif 'sticking' not in new_data.columns:
                new_data['sticking'] = 0
                debug_print("No sticking columns found, creating default sticking=0 column")
                
            # Ensure we have a 'normal' column (1 when sticking is 0, 0 otherwise)
            if 'normal' not in new_data.columns:
                new_data['normal'] = (new_data['sticking'] == 0).astype(int)
                debug_print("Created synthetic 'normal' column based on sticking values")
                
            debug_print(f"Processed data columns: {new_data.columns.tolist()}")
            debug_print(f"Sample data row: {new_data.iloc[0].to_dict() if len(new_data) > 0 else 'No data'}")
            
            MAX_DF_SIZE = 10000  # Maximum rows to keep in memory

            if df.empty:
                df = new_data.copy()
                debug_print(f"Created new dataframe with {len(df)} rows")
            else:
                df = pd.concat([df, new_data], ignore_index=True)
                debug_print(f"Updated dataframe, now has {len(df)} rows")
    
            # Apply data windowing to limit memory usage
            if len(df) > MAX_DF_SIZE:
                df = df.tail(MAX_DF_SIZE).copy()
                debug_print(f"Applied data windowing, trimmed to {len(df)} rows")
                
            last_csv_file = latest_csv
            last_file_size = current_file_size
            last_row_count = len(df)

            # Make archive if new file is detected and time has passed
            if chosen_folder:
                last_db_save_time = save_data_to_database(df, chosen_folder, last_db_save_time)

            if 'relative_time' in df.columns and len(df) > 1000:
                latest_time = df['relative_time'].max()
                cutoff_time = latest_time - MAX_DATA_AGE_SECONDS
                df = df[df['relative_time'] >= cutoff_time].reset_index(drop=True)
                debug_print(f"Removed data older than {MAX_DATA_AGE_SECONDS} seconds, now {len(df)} rows")
            

    except Exception as e:
        debug_print(f"Error reading CSV: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
        return (default_figure, f"Error reading CSV: {str(e)}",
                default_stats, default_stats, default_stats,
                default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                *punch_texts, *punch_classes)

    try:
        if df.empty:
            debug_print("DataFrame is empty, returning default figure")
            return (default_figure, "No data loaded.",
                    default_stats, default_stats, default_stats,
                    default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                    *punch_texts, *punch_classes)
        
        required_cols = ['relative_time', 'compression_force_1', 'compression_force_2', 'ejection_force', 'pre_compression_force']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            debug_print(f"Missing required columns: {missing_cols}")
            return (default_figure, f"Missing columns: {', '.join(missing_cols)}",
                    default_stats, default_stats, default_stats,
                    default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                    *punch_texts, *punch_classes)
        
        latest_time = df['relative_time'].max()
        debug_print(f"Latest time: {latest_time}")
        
        try:
            time_window = float(time_window)  # Convert to float (no change needed here)
        except (ValueError, TypeError):
            time_window = 10.0

        time_window = max(0.5, time_window)  # Ensure minimum of 0.5 seconds
        debug_print(f"Final time_window value: {time_window}")

        latest_time = df['relative_time'].max() if 'relative_time' in df.columns else 0
        x_min = max(0, latest_time - time_window)
        debug_print(f"Latest time: {latest_time}, Window start: {x_min}")

        # Then filter the dataframe to get points within the time window
        window_df = df[df['relative_time'] >= x_min].copy() if 'relative_time' in df.columns else df.copy()
        debug_print(f"Filtered data window: {len(window_df)} rows (from {len(df)} total rows)")

        if 'datetime' in window_df.columns and not window_df['datetime'].isna().all():
            window_df['datetime'] = pd.to_datetime(window_df['datetime'], errors='coerce')
            latest_time = window_df['datetime'].max()
            computed_start = latest_time - pd.Timedelta(seconds=float(time_window))
            available_start = window_df['datetime'].min()
            start_time = max(computed_start, available_start)
    
            window_df = window_df[
                (window_df['datetime'] >= start_time) & (window_df['datetime'] <= latest_time)].copy()

            min_time = start_time
            max_time = latest_time

        else:
            # Handle regular time case
            if 'relative_time' in window_df.columns:
                latest_time = window_df['relative_time'].max()
            else:
                latest_time = 0    
    
            x_max = latest_time
            # Ensure we're using the full time_window in seconds
            x_min = max(0, x_max - time_window)
    
            debug_print(f"Time window in seconds: {time_window}")
            debug_print(f"Setting x-axis range: [{x_min} to {x_max}]")
    
            # Make sure we have a valid range
            if x_min >= x_max:
                x_min = max(0, x_max - 0.1)
    
            # Re-filter the data to match the exact time window
            if 'relative_time' in window_df.columns:
                window_df = window_df[window_df['relative_time'] >= x_min].copy()

        if window_df.empty:
            debug_print("Warning: No data in selected time window")
            return (default_figure, "No data in selected time window.",
                    default_stats, default_stats, default_stats,
                    default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                    *punch_texts, *punch_classes)
            
        if len(window_df) > 200:  # Reduce points for plotting
            debug_print(f"Downsampling from {len(window_df)} points")
            if 'datetime' in window_df.columns and not window_df['datetime'].isna().all():
                # For datetime index
                window_df = window_df.sort_values('datetime')
                indices = np.round(np.linspace(0, len(window_df)-1, 300)).astype(int)
                window_df = window_df.iloc[indices].copy()
            elif 'relative_time' in window_df.columns:
                # For numeric time index
                window_df = window_df.sort_values('relative_time')
                indices = np.round(np.linspace(0, len(window_df)-1, 300)).astype(int)
                window_df = window_df.iloc[indices].copy()
            debug_print(f"Downsampled to {len(window_df)} points for plotting")

        traces = []
        colors = {
            'compression_force_1': '#1E88E5',  # Blue
            'compression_force_2': '#FFC107',  # Amber
            'ejection_force': '#4CAF50',       # Green
            'pre_compression_force': '#E91E63'  # Pink
        }

        for col, color in colors.items():
            try:
                if col in window_df.columns:
                    window_df[col] = pd.to_numeric(window_df[col], errors='coerce').fillna(0)
                    
                    if window_df[col].sum() == 0:
                        continue
                        
                    if 'datetime' in window_df.columns and not window_df['datetime'].isna().all():
                        x_values = window_df['datetime']
                        traces.append(
                            go.Scatter(
                                x=x_values,
                                y=window_df[col].tolist(),
                                mode='lines',
                                name=col.replace('_', ' ').title(),
                                line=dict(width=2, color=color)
                            )
                        )
                    else:
                        traces.append(
                            go.Scatter(
                                x=window_df['relative_time'].tolist(),
                                y=window_df[col].tolist(),
                                mode='lines',
                                name=col.replace('_', ' ').title(),
                                line=dict(width=2, color=color)
                            )
                        )
                else:
                    debug_print(f"Warning: Column {col} not found in window_df")
            except Exception as e:
                debug_print(f"Error creating trace for {col}: {str(e)}")

        if 'datetime' in window_df.columns and not window_df['datetime'].isna().all():
            # Ensure that the datetime column is processed correctly.
            window_df['datetime'] = pd.to_datetime(window_df['datetime'], errors='coerce')
            latest_time = window_df['datetime'].max()
            computed_start = latest_time - pd.Timedelta(seconds=float(time_window))
            available_start = window_df['datetime'].min()
            start_time = max(computed_start, available_start)
            debug_print(f"Latest time: {latest_time}, Computed start: {computed_start}, Available start: {available_start}, Using start_time: {start_time}")
            
            window_df = window_df[
                (window_df['datetime'] >= start_time) & (window_df['datetime'] <= latest_time)].copy()
    
            min_time = start_time
            max_time = latest_time
            
            main_figure = {
                'data': traces,
                'layout': go.Layout(
                    xaxis=dict(
                        title='Time',
                        type='date',
                        tickformat='%H:%M:%S.%L',
                        range=[min_time, max_time],  # Set explicit range based on time window
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        title='Force (N)', 
                        autorange=True, 
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    margin=dict(l=60, r=10, t=5, b=20),
                    hovermode='closest',
                    plot_bgcolor='#1f2937',
                    paper_bgcolor='#1f2937',
                    font=dict(color='#d1d5db'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    uirevision=f'datetime_{time_window}'  # Change this value when time_window changes
                )
            }
        else:
            # For numeric time axis (relative_time)
            # Ensure we properly set the x-axis range based on the time window
            if 'relative_time' in window_df.columns:
                latest_time = window_df['relative_time'].max()
            else:
                latest_time = 0
        
            x_max = latest_time
            x_min = max(0, x_max - float(time_window))  # Ensure we convert time_window to float
    
            debug_print(f"Setting relative_time x-axis range: [{x_min} to {x_max}] (window: {time_window})")
    
            # Make sure we have a valid range
            if x_min >= x_max:
                x_min = max(0, x_max - 0.1)
    
            # Re-filter the data to match the exact time window
            if 'relative_time' in window_df.columns:
                window_df = window_df[window_df['relative_time'] >= x_min].copy()
    
            main_figure = {
                'data': traces,
                'layout': go.Layout(
                    xaxis=dict(
                        title='Seconds from Start', 
                        range=[x_min, x_max],  # Use explicit calculated range
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        title='Force (N)', 
                        autorange=True, 
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    margin=dict(l=60, r=10, t=5, b=20),
                    hovermode='closest',
                    plot_bgcolor='#1f2937',
                    paper_bgcolor='#1f2937',
                    font=dict(color='#d1d5db'),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    uirevision=f'relative_{time_window}'  # Change this to force update when time_window changes
                )
            }

        # Update the hover template for all traces to show more precise time
        for trace in main_figure['data']:
            trace.update(hovertemplate='Time: %{x:.3f}s<br>Force: %{y:.2f}N<extra></extra>')
    
        
        def calculate_max_avg(series, window_size=5):
            if len(series) < window_size:
                return series.mean()
        
            max_avg = 0
            for i in range(len(series) - window_size + 1):
                window_avg = series.iloc[i:i+window_size].mean()
                max_avg = max(max_avg, window_avg)
            return max_avg

        # For pre-compression
        pre_comp_max = window_df['pre_compression_force'].max()
        pre_comp_max_avg = calculate_max_avg(window_df['pre_compression_force'])

        # For compression (combined)
        comp_combined = (window_df['compression_force_1'] + window_df['compression_force_2']) / 2
        comp_max = comp_combined.max()
        comp_max_avg = calculate_max_avg(comp_combined)

        # For ejection
        eject_max = window_df['ejection_force'].max()
        eject_max_avg = calculate_max_avg(window_df['ejection_force'])

        # Additional stats if requested
        if 'tfstats' in (show_tf_stats or []):
            pre_comp_stats = f"Max: {pre_comp_max:.2f} / Avg: {pre_comp_max_avg:.2f} / Min: {window_df['pre_compression_force'].min():.2f}"
            comp_stats = f"Max: {comp_max:.2f} / Avg: {comp_max_avg:.2f} / Min: {comp_combined.min():.2f}"
            eject_stats = f"Max: {eject_max:.2f} / Avg: {eject_max_avg:.2f} / Min: {window_df['ejection_force'].min():.2f}"
        else:
            pre_comp_stats = comp_stats = eject_stats = "Stats hidden"

        # Create the punch stats figures
        pre_compression_graph = generate_punch_stats_figure("pre_compression", df)
        compression_graph = generate_punch_stats_figure("compression", df)
        ejection_graph = generate_punch_stats_figure("ejection", df)

        try:
            sticking_color_classes = {
                0: "status-badge badge-normal",     # Green
                1: "status-badge badge-warning",    # Yellow
                2: "status-badge badge-orange",     # Orange
                3: "status-badge badge-danger"      # Red
            }
            
            recent_data = df.tail(50)
            
            # Handle individual punch sticking columns from dummy generator
            punch_sticking_columns = [col for col in recent_data.columns if 'punch' in col.lower() and 'sticking' in col.lower()]
            if punch_sticking_columns:
                for i in range(1, 9):
                    # Look for punch-specific sticking column (e.g., 'punch1-sticking')
                    punch_sticking_col = next((col for col in punch_sticking_columns 
                                             if col.lower() == f'punch{i}-sticking'), None)
                    
                    if punch_sticking_col:
                        sticking_level = recent_data[punch_sticking_col].max()
                        if sticking_level > 0:
                            punch_classes[i-1] = sticking_color_classes.get(int(sticking_level), 
                                                                         "status-badge badge-normal")
                            # Add level indicator to text
                            level_indicator = "●" * int(sticking_level)
                            punch_texts[i-1] = f"Punch {i} {level_indicator}"
            # Fallback to old method if no punch-specific columns
            elif 'sticking' in recent_data.columns:
                # For each row with sticking > 0
                sticking_rows = recent_data[recent_data['sticking'] > 0]
                
                # Track which punches are showing sticking
                punches_with_sticking = {}
                
                # Process sticking information
                for _, row in sticking_rows.iterrows():
                    # Determine which punch numbers were active
                    active_punches = []
                    punch_columns = ['compr1punchno', 'compr2punchno', 'ejectpunchno', 'precomppunchno']
                    
                    for col in punch_columns:
                        if col in row and pd.notna(row[col]) and row[col] != "":
                            try:
                                punch_values = [int(p.strip()) for p in str(row[col]).split(',') if p.strip().isdigit()]
                                active_punches.extend(punch_values)
                            except:
                                continue
                    
                    for punch in active_punches:
                        if 1 <= punch <= 8:  # Only track punches 1-8
                            # Determine sticking level based on the sticking value
                            sticking_value = float(row['sticking'])
                            
                            # Simple mapping: 0-0.33 → level 1, 0.34-0.66 → level 2, 0.67-1.0 → level 3
                            if sticking_value > 0:
                                if sticking_value < 0.33:
                                    level = 1  # Mild
                                elif sticking_value < 0.66:
                                    level = 2  # Moderate
                                else:
                                    level = 3  # Severe
                                    
                                # Update the punch's sticking level (keep highest level)
                                if punch not in punches_with_sticking or level > punches_with_sticking[punch]:
                                    punches_with_sticking[punch] = level
                
                # Now update the status indicators for each punch
                for i in range(1, 9):
                    if i in punches_with_sticking:
                        level = punches_with_sticking[i]
                        punch_classes[i-1] = sticking_color_classes[level]
                        # Add level indicator to text
                        level_indicator = "●" * level
                        punch_texts[i-1] = f"Punch {i} {level_indicator}"
                    
        except Exception as e:
            debug_print(f"Error determining punch status: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())

        return (main_figure,
                "Data loaded successfully.",
                pre_comp_stats, comp_stats, eject_stats,
                pre_compression_graph, compression_graph, ejection_graph,
                *punch_texts, *punch_classes)
                
    except Exception as e:
        debug_print(f"Error processing data: {str(e)}")
        import traceback
        debug_print(traceback.format_exc())
        return (default_figure,
                f"Error processing data: {str(e)}",
                default_stats, default_stats, default_stats,
                default_punch_stats_figure, default_punch_stats_figure, default_punch_stats_figure,
                *punch_texts, *punch_classes)

# This function was not used in the final code
# def find_latest_data_folder(base_dir):
#     date_folders = sorted(glob.glob(os.path.join(base_dir, "*")), reverse=True)
#     if not date_folders:
#         if glob.glob(os.path.join(base_dir, "*.csv")):
#             return base_dir
#         return None
        
#     for date_folder in date_folders:
#         if os.path.isdir(date_folder):
#             time_folders = sorted(glob.glob(os.path.join(date_folder, "*")), reverse=True)
#             if time_folders:
#                 for folder in time_folders:
#                     if os.path.isdir(folder):
#                         return folder
#     return None

@app.callback(
    Output("folder-path", "data"),
    [Input("folder-button", "n_clicks")],
    [State("folder-path", "data")],
    prevent_initial_call=True
)
def choose_folder(n_clicks, current_folder):
    global pywebview_window
    if n_clicks:
        # Create a mechanism to communicate with the main process
        try:
            # Check if we're running in the context with pywebview
            if pywebview_window is None:
                # Create a temporary file to signal the main process
                signal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "select_folder_signal.tmp")
                with open(signal_file, "w") as f:
                    f.write("select_folder")
                
                # Wait for the response file (timeout after 10 seconds)
                response_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_response.tmp")
                start_time = time.time()
                while not os.path.exists(response_file) and time.time() - start_time < 10:
                    time.sleep(0.1)
                
                if os.path.exists(response_file):
                    with open(response_file, "r") as f:
                        selected_folder = f.read().strip()
                    
                    # Clean up the temporary files
                    try:
                        os.remove(response_file)
                        os.remove(signal_file)
                    except:
                        pass
                    
                    if selected_folder and selected_folder != "CANCEL":
                        return selected_folder
            else:
                # Direct call if pywebview is available
                paths = pywebview_window.create_file_dialog(
                    dialog_type=webview.FOLDER_DIALOG
                )
                if paths:
                    return paths[0]
        except Exception as e:
            debug_print(f"Error in folder selection: {str(e)}")
    
    return current_folder
    
@app.callback(
    Output("selected-folder", "children"),
    [Input("folder-path", "data")]
)
def update_selected_folder(folder_path):
    if folder_path:
        return f"Data Source: {folder_path}"
    return "Data Source: Not selected"

@app.callback(
    Output("graph-update", "disabled"),
    [Input("toggle-button", "n_clicks")],
    [State("graph-update", "disabled")],
    prevent_initial_call=True
)
def start_monitoring(n_clicks, is_disabled):
    if n_clicks:
        if not is_disabled:
            raise dash.exceptions.PreventUpdate
        return False
    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("graph-update", "disabled", allow_duplicate=True),
    [Input("stop-button", "n_clicks")],
    prevent_initial_call=True
)
def stop_monitoring(n_clicks):
    if n_clicks:
        return True
    raise dash.exceptions.PreventUpdate

def run_dash():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=False, threaded=True, use_reloader=False, port=8050)

def check_for_folder_signal():
    global pywebview_window
    signal_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "select_folder_signal.tmp")
    
    while True:
        if os.path.exists(signal_file):
            try:
                os.remove(signal_file)
                
                # Show folder dialog from the main thread
                paths = pywebview_window.create_file_dialog(
                    dialog_type=webview.FOLDER_DIALOG
                )
                
                # Write response
                response_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder_response.tmp")
                with open(response_file, "w") as f:
                    if paths and len(paths) > 0:
                        f.write(paths[0])
                    else:
                        f.write("CANCEL")
            except Exception as e:
                print(f"Error processing folder signal: {str(e)}")
                
        time.sleep(0.5)

if __name__ == '__main__':
    dash_thread = threading.Thread(target=run_dash, daemon=True)
    dash_thread.start()
    
    time.sleep(1)
    
    pywebview_window = webview.create_window(
        "Production Monitoring Dashboard",
        "http://127.0.0.1:8050",
        width=1200, height=1000
    )

    signal_thread = threading.Thread(target=check_for_folder_signal, daemon=True)
    signal_thread.start()
    
    webview.start()
    sys.exit(0)