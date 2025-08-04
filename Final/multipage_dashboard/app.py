from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from pages import home, demo_page
import pandas as pd 
import zarr
import os


#DO NOT CHANGE THE CODE BELOW. EXCEPT output_csv_filename
# This assumes that your notebook is inside 'MULTIPAGE_DASHBOARD', which is at the same level as 'movie_data'
base_dir = os.path.join(os.path.dirname(os.path.abspath("__file__")), 'Final/movie_data')
zarr_directory = 'zarr_file/all_channels_data'
zarr_full_path = os.path.join(base_dir, zarr_directory)

track_df_directory = 'datasets'
track_df_file_name = 'track_df_cleaned_final_full.pkl'
track_df_all_tracks_full_directory = os.path.join(base_dir,track_df_directory, track_df_file_name)

filtered_tracks_directory = 'datasets'
filtered_tracks_file_name = 'filtered_tracks_final.pkl'
filtered_tracks_all_tracks_full_directory = os.path.join(base_dir,filtered_tracks_directory, filtered_tracks_file_name)

output_csv_directory = 'datasets'

#ONLY NEED TO UPDATE THIS 
output_csv_filename = 'output_csv.csv'

output_csv_full_directory = os.path.join(base_dir,output_csv_directory , output_csv_filename)

# Check if the file exists
if os.path.isfile(output_csv_full_directory):
    # Load the file into a DataFrame
    df = pd.read_csv(output_csv_full_directory)
else:
    # Create an empty DataFrame and a file
    df = pd.DataFrame(columns=['track_id', 'quality', 'details'])
    df.to_csv(output_csv_full_directory, index=False)


track_df = pd.read_pickle(track_df_all_tracks_full_directory)
filtered_tracks = pd.read_pickle(filtered_tracks_all_tracks_full_directory)
zarr_arr = zarr.open(store = zarr_full_path, mode = 'r')
csv_file_path = output_csv_full_directory


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server



app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Tracks the URL
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", style={'fontWeight': 'bold'})),
            dbc.NavItem(dbc.NavLink("Track Stats", href="/demo_page", style={'fontWeight': 'bold'})),
        ],
        brand=html.Img(src="/assets/MatsuLab_Logo_Long.png", height="70px"),  # Logo as brand, adjust height as needed
        brand_href="/",
        color="custom-color", 
        dark=True,
        style={'backgroundColor': 'white'},  # Set the background color directly
        #brand_logo="/assets/MatsuLab_Logo_Long.png",  # Path to your logo image
    ),
    html.Div(id='page-content'),  # Content will be rendered here
    dcc.Store(id='intermediate-value', data=[], storage_type='session'),
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
    app.run_server(debug = True)
