from dash import dcc, html
import pandas as pd

# Load data for layout components
df = pd.read_csv("data/raw/bank-full.csv", delimiter=';')

def get_header_layout(colors):
    return html.Div([
        html.H1([
            html.I(className="fas fa-chart-line", style={'marginRight': '15px'}),
            "Bank Marketing Analytics Dashboard"
        ], style={'fontSize': '2.5rem', 'fontWeight': '700', 'margin': 0}),
        html.P("Comprehensive analysis and prediction tool for bank marketing campaigns", 
               style={'fontSize': '1.2rem', 'opacity': '0.9', 'margin': '0.5rem 0 0 0'})
    ])

def get_kpi_cards(colors, card_style):
    return html.Div([
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
    ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))', 'gap': '1rem', 'margin': '2rem 0'})

def get_filters_section(colors, card_style):
    return html.Div([
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
                    max=df['duration'].quantile(0.95),
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
    ], style=card_style)

def get_exploration_tab(colors, card_style):
    return dcc.Tab(
        label='📊 Data Exploration', 
        value='exploration',
        children=[
            get_kpi_cards(colors, card_style),
            get_filters_section(colors, card_style),
            html.Div(id='filtered-charts')
        ]
    )

def get_prediction_tab(colors, card_style):
    return dcc.Tab(
        label='🎯 Predict Subscription', 
        value='prediction',
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

def get_main_layout(colors, custom_style, header_style, tab_style, tab_selected_style, card_style):
    return html.Div([
        # Header
        html.Div([
            get_header_layout(colors)
        ], style=header_style),

        # Main content
        html.Div([
            dcc.Tabs(id="main-tabs", value='exploration', 
                    style=tab_style,
                    selected_style=tab_selected_style,
                    children=[
                        get_exploration_tab(colors, card_style),
                        get_prediction_tab(colors, card_style)
                    ])
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 1rem'})
    ], style=custom_style)