from dash import Dash
import dash_bootstrap_components as dbc
from layout.navbar import navbar
from layout.homepage import homepage_layout
from layout.analysis_tab import analysis_layout
from callbacks import register_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Term Deposit Dashboard"

# Define layout
app.layout = dbc.Container([
    navbar,
    homepage_layout,
    analysis_layout,
], fluid=True)

# Register callbacks
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
