import os
import sys
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np

# Load data and model
df = pd.read_csv("data/raw/bank-full.csv", delimiter=';')  
# Load trained model
model_path = os.path.join("models", "random_forest.pkl")
model = joblib.load(model_path)  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_engineering import engineer_features, encode_features

# Dash app initialization
app = dash.Dash(__name__)
server = app.server

# Custom CSS styling
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Color scheme
colors = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#10b981',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'info': '#3b82f6',
    'light': '#f8fafc',
    'dark': '#1e293b',
    'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'card_bg': '#ffffff',
    'text_primary': '#1e293b',
    'text_secondary': '#64748b'
}

# Custom styling
custom_style = {
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    'backgroundColor': '#f1f5f9',
    'minHeight': '100vh',
    'margin': 0,
    'padding': 0
}

header_style = {
    'background': colors['gradient'],
    'color': 'white',
    'padding': '2rem 0',
    'marginBottom': '2rem',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    'textAlign': 'center'
}

card_style = {
    'backgroundColor': colors['card_bg'],
    'padding': '1.5rem',
    'borderRadius': '12px',
    'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    'margin': '1rem 0',
    'border': '1px solid #e2e8f0'
}

tab_style = {
    'borderTop': '1px solid #d6d3d1',
    'borderBottom': '1px solid #d6d3d1',
    'borderLeft': '1px solid #d6d3d1',
    'borderRight': '1px solid #d6d3d1',
    'backgroundColor': '#fafafa',
    'padding': '12px 24px',
    'fontWeight': 'bold',
    'color': colors['text_secondary']
}

tab_selected_style = {
    'borderTop': '3px solid ' + colors['primary'],
    'borderBottom': '1px solid #d6d3d1',
    'borderLeft': '1px solid #d6d3d1',
    'borderRight': '1px solid #d6d3d1',
    'backgroundColor': colors['primary'],
    'color': 'white',
    'padding': '12px 24px',
    'fontWeight': 'bold'
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-line", style={'marginRight': '15px'}),
            "Bank Marketing Analytics Dashboard"
        ], style={'fontSize': '2.5rem', 'fontWeight': '700', 'margin': 0}),
        html.P("Comprehensive analysis and prediction tool for bank marketing campaigns", 
               style={'fontSize': '1.2rem', 'opacity': '0.9', 'margin': '0.5rem 0 0 0'})
    ], style=header_style),

    # Main content
    html.Div([
        dcc.Tabs(id="main-tabs", value='exploration', children=[
            dcc.Tab(
                label='📊 Data Exploration', 
                value='exploration',
                style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    # KPI Cards
                    html.Div([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-users", style={'fontSize': '2rem', 'color': colors['info']}),
                                html.Div([
                                    html.H3(f"{len(df):,}", style={'margin': 0, 'color': colors['text_primary']}),
                                    html.P("Total Customers", style={'margin': 0, 'color': colors['text_secondary']})
                                ])
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'})
                        ], style={**card_style, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-percentage", style={'fontSize': '2rem', 'color': colors['success']}),
                                html.Div([
                                    html.H3(f"{(df['y'].value_counts()['yes'] / len(df) * 100):.1f}%", 
                                           style={'margin': 0, 'color': colors['text_primary']}),
                                    html.P("Success Rate", style={'margin': 0, 'color': colors['text_secondary']})
                                ])
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'})
                        ], style={**card_style, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-calendar", style={'fontSize': '2rem', 'color': colors['warning']}),
                                html.Div([
                                    html.H3(f"{df['duration'].mean():.0f}s", 
                                           style={'margin': 0, 'color': colors['text_primary']}),
                                    html.P("Avg Call Duration", style={'margin': 0, 'color': colors['text_secondary']})
                                ])
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'})
                        ], style={**card_style, 'textAlign': 'center'}),
                        
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-dollar-sign", style={'fontSize': '2rem', 'color': colors['danger']}),
                                html.Div([
                                    html.H3(f"€{df['balance'].mean():.0f}", 
                                           style={'margin': 0, 'color': colors['text_primary']}),
                                    html.P("Avg Balance", style={'margin': 0, 'color': colors['text_secondary']})
                                ])
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1rem'})
                        ], style={**card_style, 'textAlign': 'center'})
                    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '1rem', 'margin': '2rem 0'}),

                    # Filters Section
                    html.Div([
                        html.H3([
                            html.I(className="fas fa-filter", style={'marginRight': '10px'}),
                            "Interactive Filters"
                        ], style={'color': colors['text_primary'], 'marginBottom': '1rem'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Age Range:", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                dcc.RangeSlider(
                                    id='age-slider',
                                    min=df['age'].min(),
                                    max=df['age'].max(),
                                    step=1,
                                    marks={i: str(i) for i in range(int(df['age'].min()), int(df['age'].max())+1, 10)},
                                    value=[df['age'].min(), df['age'].max()],
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                            
                            html.Div([
                                html.Label("Duration Range (seconds):", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                dcc.RangeSlider(
                                    id='duration-slider',
                                    min=0,
                                    max=df['duration'].quantile(0.95),  # Use 95th percentile to avoid outliers
                                    step=10,
                                    marks={i: f"{i}s" for i in range(0, int(df['duration'].quantile(0.95))+1, 200)},
                                    value=[0, df['duration'].quantile(0.95)],
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ], style={'marginBottom': '2rem'}),
                        
                        html.Div([
                            html.Div([
                                html.Label("Job Categories:", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                dcc.Dropdown(
                                    id='job-filter',
                                    options=[{'label': 'All Jobs', 'value': 'all'}] + 
                                            [{'label': job.title(), 'value': job} for job in sorted(df['job'].unique())],
                                    value='all',
                                    multi=True,
                                    style={'marginTop': '5px'}
                                )
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                            
                            html.Div([
                                html.Label("Education Level:", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                dcc.Dropdown(
                                    id='education-filter',
                                    options=[{'label': 'All Levels', 'value': 'all'}] + 
                                            [{'label': edu.title(), 'value': edu} for edu in sorted(df['education'].unique())],
                                    value='all',
                                    multi=True,
                                    style={'marginTop': '5px'}
                                )
                            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block', 'verticalAlign': 'top'})
                        ])
                    ], style=card_style),

                    # Charts Section
                    html.Div(id='filtered-charts')
                ]
            ),

            dcc.Tab(
                label='🎯 Predict Subscription', 
                value='prediction',
                style=tab_style,
                selected_style=tab_selected_style,
                children=[
                    html.Div([
                        # Prediction Header
                        html.Div([
                            html.H2([
                                html.I(className="fas fa-brain", style={'marginRight': '15px', 'color': colors['primary']}),
                                "AI-Powered Subscription Prediction"
                            ], style={'color': colors['text_primary'], 'textAlign': 'center', 'marginBottom': '0.5rem'}),
                            html.P("Enter customer information to predict subscription likelihood using machine learning",
                                   style={'textAlign': 'center', 'color': colors['text_secondary'], 'fontSize': '1.1rem'})
                        ], style={'marginBottom': '2rem'}),

                        # Input Form
                        html.Div([
                            html.Div([
                                # Personal Information Section
                                html.Div([
                                    html.H4([
                                        html.I(className="fas fa-user", style={'marginRight': '10px', 'color': colors['info']}),
                                        "Personal Information"
                                    ], style={'color': colors['text_primary'], 'borderBottom': f'2px solid {colors["info"]}', 'paddingBottom': '0.5rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Age", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='age', type='number', required=True, 
                                                     placeholder="Enter age", min=18, max=100,
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                                        
                                        html.Div([
                                            html.Label("Job", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='job', 
                                                        options=[{'label': job.title().replace('-', ' '), 'value': job} for job in sorted(df['job'].unique())],
                                                        placeholder="Select job category",
                                                        style={'marginTop': '2px'})
                                        ], style={'width': '48%', 'display': 'inline-block'})
                                    ], style={'marginBottom': '1rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Marital Status", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='marital', 
                                                        options=[{'label': status.title(), 'value': status} for status in df['marital'].unique()],
                                                        placeholder="Select marital status")
                                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                                        
                                        html.Div([
                                            html.Label("Education", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='education', 
                                                        options=[{'label': edu.title(), 'value': edu} for edu in df['education'].unique()],
                                                        placeholder="Select education level")
                                        ], style={'width': '48%', 'display': 'inline-block'})
                                    ])
                                ], style={**card_style, 'marginBottom': '1.5rem'}),

                                # Financial Information Section
                                html.Div([
                                    html.H4([
                                        html.I(className="fas fa-euro-sign", style={'marginRight': '10px', 'color': colors['success']}),
                                        "Financial Information"
                                    ], style={'color': colors['text_primary'], 'borderBottom': f'2px solid {colors["success"]}', 'paddingBottom': '0.5rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Account Balance (€)", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='balance', type='number', 
                                                     placeholder="Enter balance", 
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                                        
                                        html.Div([
                                            html.Label("Default Status", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='default', 
                                                        options=[{'label': status.title(), 'value': status} for status in df['default'].unique()],
                                                        placeholder="Has credit in default?")
                                        ], style={'width': '48%', 'display': 'inline-block'})
                                    ], style={'marginBottom': '1rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Housing Loan", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='housing', 
                                                        options=[{'label': status.title(), 'value': status} for status in df['housing'].unique()],
                                                        placeholder="Has housing loan?")
                                        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                                        
                                        html.Div([
                                            html.Label("Personal Loan", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='loan', 
                                                        options=[{'label': status.title(), 'value': status} for status in df['loan'].unique()],
                                                        placeholder="Has personal loan?")
                                        ], style={'width': '48%', 'display': 'inline-block'})
                                    ])
                                ], style={**card_style, 'marginBottom': '1.5rem'}),

                                # Campaign Information Section
                                html.Div([
                                    html.H4([
                                        html.I(className="fas fa-phone", style={'marginRight': '10px', 'color': colors['warning']}),
                                        "Campaign Information"
                                    ], style={'color': colors['text_primary'], 'borderBottom': f'2px solid {colors["warning"]}', 'paddingBottom': '0.5rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Contact Method", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='contact', 
                                                        options=[{'label': method.title(), 'value': method} for method in df['contact'].unique()],
                                                        placeholder="Select contact method")
                                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                                        
                                        html.Div([
                                            html.Label("Day of Month", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='day', type='number', min=1, max=31,
                                                     placeholder="Day (1-31)",
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                                        
                                        html.Div([
                                            html.Label("Month", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Dropdown(id='month', 
                                                        options=[{'label': month.title(), 'value': month} for month in df['month'].unique()],
                                                        placeholder="Select month")
                                        ], style={'width': '32%', 'display': 'inline-block'})
                                    ], style={'marginBottom': '1rem'}),
                                    
                                    html.Div([
                                        html.Div([
                                            html.Label("Call Duration (seconds)", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='duration', type='number', min=0,
                                                     placeholder="Duration in seconds",
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                                        
                                        html.Div([
                                            html.Label("Campaign Contacts", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='campaign', type='number', min=1,
                                                     placeholder="Number of contacts",
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                                        
                                        html.Div([
                                            html.Label("Days Since Last Contact", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='pdays', type='number', min=-1,
                                                     placeholder="Days (-1 if never)",
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '24%', 'display': 'inline-block', 'marginRight': '1%'}),
                                        
                                        html.Div([
                                            html.Label("Previous Contacts", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                            dcc.Input(id='previous', type='number', min=0,
                                                     placeholder="Previous contacts",
                                                     style={'width': '100%', 'padding': '10px', 'borderRadius': '8px', 'border': '1px solid #d1d5db'})
                                        ], style={'width': '24%', 'display': 'inline-block'})
                                    ], style={'marginBottom': '1rem'}),
                                    
                                    html.Div([
                                        html.Label("Previous Campaign Outcome", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                                        dcc.Dropdown(id='poutcome', 
                                                    options=[{'label': outcome.title(), 'value': outcome} for outcome in df['poutcome'].unique()],
                                                    placeholder="Select previous outcome")
                                    ])
                                ], style={**card_style, 'marginBottom': '2rem'}),

                                # Predict Button
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-magic", style={'marginRight': '10px'}),
                                        "Predict Subscription"
                                    ], id='predict-button', n_clicks=0,
                                    style={
                                        'background': colors['gradient'],
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '15px 30px',
                                        'fontSize': '1.1rem',
                                        'fontWeight': 'bold',
                                        'borderRadius': '10px',
                                        'cursor': 'pointer',
                                        'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                                        'transition': 'all 0.3s ease',
                                        'width': '100%'
                                    })
                                ], style={'textAlign': 'center', 'margin': '2rem 0'}),

                                # Prediction Output
                                html.Div(id='prediction-output', style={'marginTop': '2rem'})
                            ], style={'maxWidth': '800px', 'margin': '0 auto'})
                        ])
                    ])
                ]
            )
        ])
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 1rem'})
], style=custom_style)

# Callback for filtered charts
@app.callback(
    Output('filtered-charts', 'children'),
    [Input('age-slider', 'value'),
     Input('duration-slider', 'value'),
     Input('job-filter', 'value'),
     Input('education-filter', 'value')]
)
def update_charts(age_range, duration_range, job_filter, education_filter):
    # Filter data
    filtered_df = df.copy()
    
    # Apply filters
    filtered_df = filtered_df[
        (filtered_df['age'] >= age_range[0]) & 
        (filtered_df['age'] <= age_range[1]) &
        (filtered_df['duration'] >= duration_range[0]) & 
        (filtered_df['duration'] <= duration_range[1])
    ]
    
    if job_filter != 'all' and job_filter:
        if isinstance(job_filter, list):
            if 'all' not in job_filter:
                filtered_df = filtered_df[filtered_df['job'].isin(job_filter)]
        else:
            filtered_df = filtered_df[filtered_df['job'] == job_filter]
    
    if education_filter != 'all' and education_filter:
        if isinstance(education_filter, list):
            if 'all' not in education_filter:
                filtered_df = filtered_df[filtered_df['education'].isin(education_filter)]
        else:
            filtered_df = filtered_df[filtered_df['education'] == education_filter]

    # Create charts
    charts = []
    
    # 1. Success Rate by Job (Enhanced)
    job_success = filtered_df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
    job_success = job_success.sort_values('y', ascending=True)
    
    fig1 = px.bar(job_success, x='y', y='job', orientation='h',
                  title="📊 Success Rate by Job Category",
                  color='y', color_continuous_scale='viridis',
                  labels={'y': 'Success Rate (%)', 'job': 'Job Category'})
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        title={'font': {'size': 18, 'color': colors['text_primary']}},
        height=400
    )
    
    # 2. Age Distribution by Subscription
    fig2 = px.histogram(filtered_df, x='age', color='y', barmode='overlay',
                        title="📈 Age Distribution by Subscription Status",
                        opacity=0.7, nbins=20)
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        title={'font': {'size': 18, 'color': colors['text_primary']}},
        height=400
    )
    
    # 3. Balance vs Duration Scatter
    fig3 = px.scatter(filtered_df, x='balance', y='duration', color='y',
                      title="💰 Account Balance vs Call Duration",
                      hover_data=['job', 'age'],
                      color_discrete_map={'yes': colors['success'], 'no': colors['danger']})
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        title={'font': {'size': 18, 'color': colors['text_primary']}},
        height=400
    )
    
    # 4. Monthly Campaign Performance
    monthly_stats = filtered_df.groupby('month').agg({
        'y': lambda x: (x == 'yes').mean() * 100,
        'duration': 'mean'
    }).reset_index()
    
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=monthly_stats['month'], y=monthly_stats['y'],
                             mode='lines+markers', name='Success Rate (%)',
                             line=dict(color=colors['primary'], width=3),
                             marker=dict(size=8)))
    fig4.update_layout(
        title={
            'text': "📅 Monthly Campaign Performance",
            'font': {'size': 18, 'color': colors['text_primary']}
        },
        xaxis_title="Month",
        yaxis_title="Success Rate (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        height=400
    )
    
    # 5. Education vs Housing Loan Heatmap
    edu_housing = pd.crosstab(filtered_df['education'], filtered_df['housing'], normalize='index') * 100
    
    fig5 = px.imshow(edu_housing, text_auto=True, aspect="auto",
                     title="🏠 Education Level vs Housing Loan Status",
                     color_continuous_scale='RdYlBu')
    fig5.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        title={'font': {'size': 18, 'color': colors['text_primary']}},
        height=400
    )
    
    # 6. Campaign Contact Analysis
    campaign_analysis = filtered_df.groupby('campaign').agg({
        'y': lambda x: (x == 'yes').mean() * 100,
        'duration': 'mean'
    }).reset_index()
    campaign_analysis = campaign_analysis[campaign_analysis['campaign'] <= 10]  # Limit for clarity
    
    fig6 = px.bar(campaign_analysis, x='campaign', y='y',
                  title="📞 Success Rate by Number of Campaign Contacts",
                  color='y', color_continuous_scale='plasma')
    fig6.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': colors['text_primary']},
        title={'font': {'size': 18, 'color': colors['text_primary']}},
        height=400,
        xaxis_title="Number of Contacts",
        yaxis_title="Success Rate (%)"
    )
    
    # Create layout with charts
    charts = [
        # Row 1: Success Rate by Job and Age Distribution
        html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig1)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([dcc.Graph(figure=fig2)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        # Row 2: Balance vs Duration and Monthly Performance
        html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig3)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([dcc.Graph(figure=fig4)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ]),
        
        # Row 3: Education Heatmap and Campaign Analysis
        html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig5)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.Div([dcc.Graph(figure=fig6)], style=card_style)
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ])
    ]
    
    return charts

# Prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('age', 'value'),
    State('job', 'value'),
    State('marital', 'value'),
    State('education', 'value'),
    State('default', 'value'),
    State('balance', 'value'),
    State('housing', 'value'),
    State('loan', 'value'),
    State('contact', 'value'),
    State('day', 'value'),
    State('month', 'value'),
    State('duration', 'value'),
    State('campaign', 'value'),
    State('pdays', 'value'),
    State('previous', 'value'),
    State('poutcome', 'value'),
)
def make_prediction(n_clicks, age, job, marital, education, default, balance, housing, loan,
                    contact, day, month, duration, campaign, pdays, previous, poutcome):

    if n_clicks > 0:
        # Validation
        required_fields = [age, job, marital, education, default, housing, loan, 
                          contact, day, month, duration, campaign, pdays, previous, poutcome]
        
        if any(field is None or field == '' for field in required_fields):
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", 
                          style={'fontSize': '2rem', 'color': colors['warning'], 'marginBottom': '1rem'}),
                    html.H4("⚠️ Missing Information", style={'color': colors['warning'], 'margin': '0.5rem 0'}),
                    html.P("Please fill in all required fields to make a prediction.", 
                          style={'color': colors['text_secondary']})
                ], style={'textAlign': 'center', 'padding': '2rem'})
            ], style={**card_style, 'border': f'2px solid {colors["warning"]}', 'backgroundColor': '#fefce8'})

        input_dict = {
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'contact': contact, 'day': day, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
        }

        try:
            input_df = pd.DataFrame([input_dict])
            engineered = engineer_features(input_df)
            encoded = encode_features(engineered)

            prediction = model.predict(encoded)[0]
            probability = model.predict_proba(encoded)[0][1]
            
            # Determine confidence level and styling
            if probability >= 0.7:
                confidence_level = "High"
                confidence_color = colors['success']
                confidence_icon = "fas fa-check-circle"
                bg_color = '#f0fdf4'
                border_color = colors['success']
            elif probability >= 0.4:
                confidence_level = "Medium"
                confidence_color = colors['warning']
                confidence_icon = "fas fa-exclamation-circle"
                bg_color = '#fefce8'
                border_color = colors['warning']
            else:
                confidence_level = "Low"
                confidence_color = colors['danger']
                confidence_icon = "fas fa-times-circle"
                bg_color = '#fef2f2'
                border_color = colors['danger']

            return html.Div([
                # Main Prediction Result
                html.Div([
                    html.Div([
                        html.I(className=confidence_icon, 
                              style={'fontSize': '3rem', 'color': confidence_color, 'marginBottom': '1rem'}),
                        html.H2(f"Prediction: {'✅ WILL SUBSCRIBE' if prediction == 1 else '❌ WILL NOT SUBSCRIBE'}", 
                               style={'color': confidence_color, 'margin': '0.5rem 0', 'fontWeight': 'bold'}),
                        html.H3(f"{probability * 100:.1f}% Probability", 
                               style={'color': colors['text_primary'], 'margin': '0.5rem 0'}),
                        html.P(f"Confidence Level: {confidence_level}", 
                              style={'color': colors['text_secondary'], 'fontSize': '1.1rem'})
                    ], style={'textAlign': 'center', 'padding': '2rem'})
                ], style={**card_style, 'border': f'3px solid {border_color}', 'backgroundColor': bg_color, 'marginBottom': '1.5rem'}),
                
                # Detailed Analysis
                html.Div([
                    html.H4([
                        html.I(className="fas fa-chart-bar", style={'marginRight': '10px', 'color': colors['info']}),
                        "Prediction Analysis"
                    ], style={'color': colors['text_primary'], 'borderBottom': f'2px solid {colors["info"]}', 'paddingBottom': '0.5rem'}),
                    
                    html.Div([
                        # Probability Bar
                        html.Div([
                            html.Label("Subscription Probability", style={'fontWeight': 'bold', 'color': colors['text_primary']}),
                            html.Div([
                                html.Div(
                                    style={
                                        'width': f'{probability * 100}%',
                                        'height': '25px',
                                        'background': f'linear-gradient(90deg, {colors["danger"]} 0%, {colors["warning"]} 50%, {colors["success"]} 100%)',
                                        'borderRadius': '12px',
                                        'transition': 'width 0.5s ease'
                                    }
                                )
                            ], style={
                                'width': '100%',
                                'height': '25px',
                                'backgroundColor': '#e5e7eb',
                                'borderRadius': '12px',
                                'marginTop': '5px'
                            }),
                            html.P(f"{probability * 100:.1f}%", 
                                  style={'textAlign': 'center', 'margin': '5px 0', 'fontWeight': 'bold', 'color': confidence_color})
                        ], style={'marginBottom': '1.5rem'}),
                        
                        # Key Factors
                        html.Div([
                            html.H5("📋 Key Customer Profile:", style={'color': colors['text_primary'], 'marginBottom': '1rem'}),
                            html.Div([
                                html.Div([
                                    html.Strong("Demographics: ", style={'color': colors['info']}),
                                    f"{age} years old, {job.replace('-', ' ').title()}, {marital.title()}"
                                ], style={'marginBottom': '0.5rem'}),
                                html.Div([
                                    html.Strong("Financial: ", style={'color': colors['success']}),
                                    f"€{balance} balance, Housing: {housing.title()}, Personal Loan: {loan.title()}"
                                ], style={'marginBottom': '0.5rem'}),
                                html.Div([
                                    html.Strong("Campaign: ", style={'color': colors['warning']}),
                                    f"{duration}s call duration, {campaign} contacts this campaign"
                                ], style={'marginBottom': '0.5rem'})
                            ])
                        ])
                    ])
                ], style=card_style),
                
                # Recommendations
                html.Div([
                    html.H4([
                        html.I(className="fas fa-lightbulb", style={'marginRight': '10px', 'color': colors['warning']}),
                        "Recommendations"
                    ], style={'color': colors['text_primary'], 'borderBottom': f'2px solid {colors["warning"]}', 'paddingBottom': '0.5rem'}),
                    
                    html.Div([
                        html.Ul([
                            html.Li("🎯 Focus on customers with similar profiles for higher success rates" if prediction == 1 
                                   else "🔄 Consider adjusting campaign strategy for this customer segment"),
                            html.Li(f"📞 Optimal call duration appears to be around {duration}s" if duration > 200 
                                   else "📞 Consider longer calls to build better rapport"),
                            html.Li("💼 Job category and education level are strong predictors of subscription likelihood"),
                            html.Li(f"📊 This prediction has {confidence_level.lower()} confidence - consider additional data points" 
                                   if confidence_level != "High" else "📊 High confidence prediction - reliable for decision making")
                        ], style={'color': colors['text_secondary'], 'lineHeight': '1.6'})
                    ])
                ], style=card_style)
            ])

        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle", 
                          style={'fontSize': '2rem', 'color': colors['danger'], 'marginBottom': '1rem'}),
                    html.H4("❌ Prediction Error", style={'color': colors['danger'], 'margin': '0.5rem 0'}),
                    html.P(f"An error occurred while making the prediction: {str(e)}", 
                          style={'color': colors['text_secondary']}),
                    html.P("Please check your input values and try again.", 
                          style={'color': colors['text_secondary']})
                ], style={'textAlign': 'center', 'padding': '2rem'})
            ], style={**card_style, 'border': f'2px solid {colors["danger"]}', 'backgroundColor': '#fef2f2'})

    return html.Div()

# Run the server
if __name__ == '__main__':
    app.run(debug=True)