from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pages import home, demo_page
import pandas as pd 
import zarr


track_df = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/track_df_cleaned_final_full.pkl')
filtered_tracks = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/filtered_tracks_final.pkl')
zarr_arr = zarr.open(store = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/zarr_file/all_channels_data', mode = 'r')
z_shape = zarr_arr.shape

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
    html.Div(id='page-content'), 
    dcc.Store(id='intermediate-value', data = [], storage_type='local')
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
