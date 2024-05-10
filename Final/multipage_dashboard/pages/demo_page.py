import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback 
from dash.dependencies import Input, Output 
from skimage import io
import zarr 
import dash_bootstrap_components as dbc
import json



track_df = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/track_df_cleaned_final_full.pkl')
dataframe = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/filtered_tracks_final.pkl')
zarr_arr = zarr.open(store = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/zarr_file/all_channels_data', mode = 'r')
z_shape = zarr_arr.shape

def create_track_summary_table(track_id = 2, dataframe = dataframe):
    # Filter the DataFrame for the given track_id
    #track_data = dataframe[dataframe['track_id'] == track_id]
    
    # Calculate summary statistics or gather relevant track information
    summary_data = {
        'Statistic': ['Track ID', 'Total Length', 'Start Frame', 'End Frame', 'Mean Displacement', 'Mean Z', 'Mean Z Displacement', 
                     'Dynamin Positive', 'Actin Positive', 'Membrane Region'],
        'Value': [
            dataframe[dataframe['track_id'] == track_id]['track_id'].values,
            dataframe[dataframe['track_id'] == track_id]['track_length'].values,  # Total number of points for the track
            dataframe[dataframe['track_id'] == track_id]['track_start'].values,  
            dataframe[dataframe['track_id'] == track_id]['track_end'].values,  
            np.round(dataframe[dataframe['track_id'] == track_id]['mean_displacement'].values, 2),
            np.round(dataframe[dataframe['track_id'] == track_id]['mean_z'].values,2),
            np.round(dataframe[dataframe['track_id'] == track_id]['mean_z_displacement'].values,2),
            dataframe[dataframe['track_id'] == track_id]['dnm2_positive'].values,
            dataframe[dataframe['track_id'] == track_id]['actin_positive'].values,
            dataframe[dataframe['track_id'] == track_id]['membrane_region'].values,
        ]
    }

    # Convert summary data to DataFrame for easier handling in go.Table
    summary_df = pd.DataFrame(summary_data)
    
    # Create a figure with a table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Statistic', 'Value'],
            font=dict(size=14),
            align="center",
            fill_color='paleturquoise',  # Background color for header
            line_color='darkslategray'  # Border color for header cells
        ),
        cells=dict(
            values=[summary_df['Statistic'], summary_df['Value']],
            align="left",
            fill=dict(color=['white', 'white']),  # Alternating row colors
            line_color='black',  # Border color for cell
            font=dict(size=14)
        ))
    ])
    
    fig.update_layout(
        width=600,  # Adjust width according to the layout
        height=450,  # Adjust height according to the number of rows
        title={
        'text': f'Summary for Track ID {track_id}',
        'y':0.9,  # Adjust vertical position
        'x':0.5,  # Center the title horizontally
        'xanchor': 'center',  # Use 'center' to center-align the title
        'yanchor': 'top',  # Use 'top' to adjust the title from the top
        'font': dict(
            #family="Arial, sans-serif",  # Set the font family
            size=20,  # Set the font size
            color="black"  # Set the font color
        )
    }
    )
    
    return fig



# Define the layout for the Demo Page
layout = html.Div([
    dcc.Graph(id='track-summary-table'),  # Placeholder for the table
])


@callback(Output('track-summary-table', 'figure'), [Input('intermediate-value', 'data')])
def update_graph(jsonified_cleaned_data):

    # more generally, this line would be
    val = json.loads(jsonified_cleaned_data)
    #print(val['selected_track_id'])
    return create_track_summary_table(val['selected_track_id'], dataframe)

