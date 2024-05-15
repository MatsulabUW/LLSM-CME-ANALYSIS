import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback 
from dash.dependencies import Input, Output 
import dash_bootstrap_components as dbc
import json
import os


# Correctly obtain the path of the current script file
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'test_data')

track_df_directory = 'datasets'
track_df_file_name = 'track_df_cleaned_final_full.pkl'
track_df_all_tracks_full_directory = os.path.join(base_dir,track_df_directory, track_df_file_name)

filtered_tracks_directory = 'datasets'
filtered_tracks_file_name = 'filtered_tracks_final.pkl'
filtered_tracks_all_tracks_full_directory = os.path.join(base_dir,filtered_tracks_directory, filtered_tracks_file_name)

track_df = pd.read_pickle(track_df_all_tracks_full_directory)
filtered_tracks = pd.read_pickle(filtered_tracks_all_tracks_full_directory)

def create_track_summary_table(track_id = 2, dataframe = filtered_tracks):
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

def plot_track_movement(track_id, dataframe = track_df):
    # Filter the dataframe for the given track_id
    track_data = dataframe[dataframe['track_id'] == track_id]
    
    # Sort the data by frame to ensure the lines connect in the correct order
    track_data = track_data.sort_values('frame')
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=track_data['c3_mu_x'],
        y=track_data['c3_mu_y'],
        z=track_data['c3_mu_z'],
        mode='lines+markers+text',  # This uses both lines and markers to plot the points
        marker=dict(
            size=4,  # Adjust the size of markers
            color=track_data['frame'],  # Color points by frame for visual distinction of time
            colorscale='Viridis',  # Color scale can be adjusted
            colorbar=dict(title='Frame'), showscale=False
        ),
        line=dict(
            color='darkblue',  # Line color
            width=2  # Line width
        ),
        text=track_data['frame'],  # Show frame number
        textposition="top center",  # Position the text above markers
        textfont=dict(  # Adjust text font here
            color='red'  # Set the text color to white
        )
    )])
    
    # Update the layout of the plot to add titles and axes labels
    fig.update_layout(
        title=f'Movement of Track {track_id} Over Time',
        scene=dict(
            xaxis_title='mu_x',
            yaxis_title='mu_y',
            zaxis_title='mu_z', 
            #camera=dict(
                #eye=dict(x=2, y=2, z=0.1),  # Adjust the camera's position
                #up=dict(x=6, y=6, z=1)  # Set z as 'up' direction for rotation around z-axis
            #)
        ),
        scene_aspectmode='auto',  # This can be adjusted to 'cube' or other aspect ratios
        width=600,  # Width of the figure in pixels
        height=450,  # Height of the figure in pixels
        margin=dict(l=0, r=0, t=0, b=0), # Adjust margins to fit titles and labels properly
        #dragmode = True, 
        #paper_bgcolor="rgb(0,0,0,0)", 
        #font=dict(  # Adjust text font here
            #color='white'  # Set the text color to white
        #)
    )

    return fig

# Define the layout for the Demo Page
layout = html.Div([ 
    html.H1('Second Page'), 
    html.Div([
    html.Div([
        dcc.Graph(id='track-summary-table'),  # Placeholder for the table
    ], style={'flex': '3', 'padding': '5px'}), 
    html.Div([
        dcc.Graph(id='3d-track'), 
    ], style={'flex': '3', 'padding': '5px'}), 

], style={'display': 'flex', 'justify-content': 'space-between'}),
 ], style={'backgroundColor': '#d6dbdf', "padding": "50px", "font-family": "'Segoe UI', Arial, sans-serif"})


@callback(Output('track-summary-table', 'figure'), [Input('intermediate-value', 'data')])
def update_graph(jsonified_cleaned_data):

    # more generally, this line would be
    val = json.loads(jsonified_cleaned_data)
    #print(val['selected_track_id'])
    return create_track_summary_table(val['selected_track_id'], filtered_tracks)

@callback(Output('3d-track', 'figure'), [Input('intermediate-value', 'data')])
def updated_graph(jsonified_cleaned_data): 
    val = json.loads(jsonified_cleaned_data)
    return plot_track_movement(track_id = val['selected_track_id'], dataframe = track_df)

