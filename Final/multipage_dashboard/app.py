from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pages import home, demo_page

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Layout of the app with navigation links
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        dcc.Link('Home | ', href='/'),
        dcc.Link('Tracks Stats', href='/demo_page')
    ], className='nav-links'),
    html.Div(id='page-content')
])

# Callback to switch pages based on URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/demo_page':
        return demo_page.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)
