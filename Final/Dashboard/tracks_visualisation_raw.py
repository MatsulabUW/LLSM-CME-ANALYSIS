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





##Importing the data

track_df = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/track_df_cleaned_final_full.pkl')
filtered_tracks = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/datasets/filtered_tracks_final.pkl')
zarr_arr = zarr.open(store = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/Final/test_data/zarr_file/all_channels_data', mode = 'r')
z_shape = zarr_arr.shape


##Generate the unique number of tracks 

unique_tracks = filtered_tracks['track_id'].unique()

##Functions 

#select the option to see tracks which are either dynamin positive or actin positive
def select_type_of_tracks(dataframe, only_dynamin_tracks = False, only_actin_tracks = False, all_positive_tracks = True, both_tracks_positive = False): 
        if only_dynamin_tracks == True: 
            relevant_tracks = dataframe[(dataframe['dnm2_positive'] == True) & (dataframe['actin_positive'] == False)]['track_id'].values
        elif only_actin_tracks == True: 
            relevant_tracks = dataframe[(dataframe['dnm2_positive'] == False) & (dataframe['actin_positive'] == True)]['track_id'].values
        elif all_positive_tracks == True: 
            relevant_tracks = dataframe[(dataframe['dnm2_positive'] == True) | (dataframe['actin_positive'] == True)]['track_id'].values
        elif both_tracks_positive == True: 
            relevant_tracks = dataframe[(dataframe['dnm2_positive'] == True) & (dataframe['actin_positive'] == True)]['track_id'].values

        return relevant_tracks

def select_tracks_region_wise(dataframe, tracks, only_basal, only_apical, only_lateral, all):
    updated_tracks_df = dataframe[dataframe['track_id'].isin(tracks)]
    if only_basal == True:
        updated_tracks = updated_tracks_df[updated_tracks_df['membrane_region'] == 'Basal']['track_id'].values
    elif only_apical == True: 
        updated_tracks = updated_tracks_df[updated_tracks_df['membrane_region'] == 'Apical']['track_id'].values
    elif only_lateral == True: 
        updated_tracks = updated_tracks_df[updated_tracks_df['membrane_region'] == 'Lateral']['track_id'].values
    elif all == True: 
        return updated_tracks_df['track_id'].values 
    
    return updated_tracks

def select_type_of_intensity(type, voxel_sum_col_names = ['c3_voxel_sum', 'c2_voxel_sum', 'c1_voxel_sum'], 
                             adjusted_voxel_sum_col_names = ['c3_voxel_sum_adjusted', 'c2_voxel_sum_adjusted', 'c1_voxel_sum_adjusted'], 
                             gaussian_col_names = ['c3_gaussian_amp', 'c2_gaussian_amp', 'c1_gaussian_amp'], 
                             peak_col_names = ['c3_peak_amp', 'c2_peak_amp', 'c1_peak_amp']): 
    if type == 'Adjusted Voxel Sum': 
        return adjusted_voxel_sum_col_names
    elif type == 'Voxel Sum': 
        return voxel_sum_col_names
    elif type == 'Gaussian Peaks': 
        return gaussian_col_names
    elif type == 'Peak Intensity': 
        return peak_col_names

def max_z_track_visualisation(track_of_interest,zarr_array,main_tracking_df, channel):
    

    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]

    # list to store max z slice of a spot in each frame of the track
    max_z_slice_cropped_images = []
    
    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['c3_mu_z'], track['c3_mu_y'], track['c3_mu_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 1 * sigma_z), int(mu_z + 1 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(zarr_array.shape[2], z_end)
        y_end = min(zarr_array.shape[3], y_end)
        x_end = min(zarr_array.shape[4], x_end)

        current_3d_spot = zarr_array[frame, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        # calculate the z_sum for each z slice of the current spot 
        z_sum = np.sum(current_3d_spot, axis=(1, 2))
        # Find the index of the maximum sum
        max_sum_index = np.argmax(z_sum)
        # Select the slice with the maximum sum
        max_z_slice_cropped_images.append(current_3d_spot[max_sum_index,:,:])
        
    
    return max_z_slice_cropped_images

def total_sum_track_visualisation(track_of_interest,zarr_array,main_tracking_df,channel):
    
    
    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    
    # list to store sum of each z slice
    z_slice_sum_cropped_images = []

    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['c3_mu_z'], track['c3_mu_y'], track['c3_mu_x']
        #sigma_z, sigma_y, sigma_x = track['sigma_z'], track['sigma_y'], track['sigma_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 1 * sigma_z), int(mu_z + 1 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(zarr_array.shape[2], z_end)
        y_end = min(zarr_array.shape[3], y_end)
        x_end = min(zarr_array.shape[4], x_end)


        current_3d_spot = zarr_array[frame, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        z_sum = np.sum(current_3d_spot,axis=0)
        z_slice_sum_cropped_images.append(z_sum)
        
    return z_slice_sum_cropped_images


##Maximum Intensity Projection Function 
def max_intensity_projection_track_visualisation(track_of_interest,zarr_array,main_tracking_df,channel):
    
    
    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]

    mip_movie = []

    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['c3_mu_z'], track['c3_mu_y'], track['c3_mu_x']
        #sigma_z, sigma_y, sigma_x = track['sigma_z'], track['sigma_y'], track['sigma_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 1 * sigma_z), int(mu_z + 1 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(zarr_array.shape[2], z_end)
        y_end = min(zarr_array.shape[3], y_end)
        x_end = min(zarr_array.shape[4], x_end)



        current_3d_spot = zarr_array[frame, channel, z_start:z_end, y_start:y_end, x_start:x_end]
        mip_projection = np.max(current_3d_spot,axis=0)
        mip_movie.append(mip_projection)
    
    return mip_movie



##Cropping the movie 

def crop_movie(image):
    image_array = []
    # Iterate through each frame in the time series
    for i in range(image.shape[0]):
        # Get non-zero indices in the current frame
        non_zero_indices = np.nonzero(image[i, :, :])
        if len(non_zero_indices[1]) > 0:
            image_array.append(image[i,min(non_zero_indices[0]):max(non_zero_indices[0])+1,
                                             min(non_zero_indices[1]):max(non_zero_indices[1])+1])
    return image_array

##Plotting the graphs 

def plot_raw_movie(plot_type = 'max_intensity_projection', track_number = unique_tracks[0], raw_image = zarr_arr, main_tracking_df = track_df, channel = 2):
    
    if plot_type == 'max_intensity_projection':      
        result_array = max_intensity_projection_track_visualisation(track_number,raw_image,main_tracking_df, channel)
    elif plot_type == 'max_z_slice':      
        result_array = max_z_track_visualisation(track_number,raw_image,main_tracking_df, channel)
    elif plot_type == 'total_z_sum':      
        result_array = total_sum_track_visualisation(track_number,raw_image,main_tracking_df, channel)


    length_of_track = len(result_array)
    # Set the number of rows and columns for subplots
    num_cols = 7
    num_rows = length_of_track // num_cols + 1
    
    unique_tracks = main_tracking_df[main_tracking_df['track_id'] == track_number]['frame'].unique()
    subplot_titles_tuple = tuple(f'frame_{i}' for i in unique_tracks)

    fig = make_subplots(rows=num_rows, cols=7, subplot_titles = subplot_titles_tuple, x_title = 'Frames', 
                        y_title = 'Intensity', row_titles = None, column_titles = None)
    
    
    for i in range(length_of_track):
        fig.layout.annotations[i]["font"] = {'size': 10, 'color':'black'}

    fig.layout.annotations[length_of_track]["font"] = {'size': 15, 'color':'black'}
    fig.layout.annotations[length_of_track+1]["font"] = {'size': 15, 'color':'black'}

    r = 1
    c = 1
    for i in range(len(result_array)):
        image = px.imshow(result_array[i],color_continuous_scale = 'blues')
        fig.add_trace(image.data[0], row = r, col = c)
        fig.update_xaxes(showticklabels=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, row=r, col=c)
        if i != 0 and (i+1) % (num_cols) == 0: 
            r = r + 1
            c = 1
        else: 
            c = c + 1
    r = 1 
    c = 1
    
    #fig.update_layout(title = 'Raw Image 3', title_x = 0.5, title_font=dict(size=30, color = 'black'))
    
    return fig

#Line chart plot
def plot_intensity_over_time(track_of_interest = unique_tracks[0], main_tracking_df = track_df, type_of_intensity = 'Adjusted Voxel Sum'):
    current_track_df = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    intensity_col_names = select_type_of_intensity(type_of_intensity)

    # Create Line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=current_track_df['frame'], y=current_track_df[intensity_col_names[0]], name = 'Channel 3',
                 line = dict(color = 'red', width = 4)))
    fig.add_trace(go.Scatter(x=current_track_df['frame'], y=current_track_df[intensity_col_names[1]],name = 'Channel 2', 
                            line=dict(color='green', width = 4)))
    fig.add_trace(go.Scatter(x=current_track_df['frame'], y=current_track_df[intensity_col_names[2]],name = 'Channel 1', 
                            line=dict(color='blue', width = 4)))

    # Edit the layout
    fig.update_layout(title=None, title_x = 0.5,
                       xaxis_title='Frames',  xaxis_color = 'black', title_font = dict(color = 'black', size = 40),
                       yaxis_title='Intensity', yaxis_color = 'black', legend=dict(bgcolor=None),autosize = True, plot_bgcolor = None,
                     paper_bgcolor = None)


    return fig 





app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Raw Intensity Visualization Dashboard", style={"text-align": "center"}),
    html.Br(),
    html.Div([
        # First checklist and its label
        html.Div([
            html.Label('Type of tracks to display (Select only one option)'),
            dcc.Checklist(
                id='condition-selection',
                options=[
                    {'label': 'Only Dynamin Positive Tracks', 'value': 'only_dynamin'},
                    {'label': 'Actin or Dynamin Positive Tracks', 'value': 'all_positive'}, 
                    {'label': 'Actin & Dynamin Positive Tracks', 'value': 'both_positive'},
                    {'label': 'Only Actin Positive Tracks', 'value': 'only_actin'}
                ],
                value=['all_positive'],  # Default selected value
                style={'width': '100%', 'border': '2px solid black'}
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '5px'}),

        # Second checklist and its label
        html.Div([
            html.Label('Membrane Region of Tracks'),
            dcc.Checklist(
                id='region-selection',
                options=[
                    {'label': 'All Regions', 'value': 'all'},
                    {'label': 'Basal Only', 'value': 'basal'}, 
                    {'label': 'Apical Only', 'value': 'apical'}, 
                    {'label': 'Lateral Only', 'value': 'lateral'}
                ],
                value=['all'],  # Default selected value
                style={'width': '100%', 'border': '2px solid black'}
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '5px'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    html.Br(),
    html.Label('Select the Track Number:'),
    dcc.Dropdown(
        id='track_number_dropdown',
        options=[],  # options set dynamically
        value=None,
        style={'width': '100%'}
    ),
    html.Label('Select the type of feature to display:'),
    dcc.Dropdown(
        id='display_type', 
        options=[
            {'label': 'Max Intensity Projection', 'value': 'max_intensity_projection'},
            {'label': 'Max Z Slice', 'value': 'max_z_slice'},
            {'label': 'Total Z Sum', 'value': 'total_z_sum'}
        ],
        value='max_intensity_projection',
        style={'width': '100%'}
    ),

    # New dropdown for intensity types
    html.Label('Type of intensity to view:'),
    dcc.Dropdown(
        id='intensity_type',
        options=[
            {'label': 'Voxel Sum', 'value': 'Voxel Sum'},
            {'label': 'Adjusted Voxel Sum', 'value': 'Adjusted Voxel Sum'},
            {'label': 'Peak Intensity', 'value': 'Peak Intensity'},
            {'label': 'Gaussian Peaks', 'value': 'Gaussian Peaks'}
        ],
        value='Adjusted Voxel Sum',  # Default selected value
        style={'width': '100%'}
    ),

    # Additional visualization elements as previously defined
    html.Br(),
    html.Div([
        html.Label('Channel 3 Track'),
        dcc.Graph(id='track_visualization'),  # assuming plot_raw_movie() is defined
    ], style={'border': '2px solid black', 'display': 'inline-block', 'width': '50%', 'text-align': 'center'}),
    html.Div([
        html.Label('Channel 2 Track'),
        dcc.Graph(id='track_visualization_2'),
    ], style={'border': '2px solid black', 'display': 'inline-block', 'width': '50%', 'text-align': 'center'}),
    html.Div([
        html.Label('Channel 1 Track'),
        dcc.Graph(id='track_visualization_3'),
    ], style={'border': '2px solid black', 'display': 'inline-block', 'width': '50%', 'text-align': 'center'}),
    html.Div([
        html.Label('Intensity Over time plot'),
        dcc.Graph(id='intensity_over_time')
    ], style={'border': '2px solid black', 'display': 'inline-block', 'width': '50%', 'text-align': 'center'})
], style={'backgroundColor': 'lightgray', "padding": "50px"})


#def select_tracks_region_wise(dataframe, tracks, only_basal, only_apical, only_lateral, all):
                    #{'label': 'All Regions', 'value': 'all'},
                    #{'label': 'Basal Only', 'value': 'basal'}, 
                    #{'label': 'Apical Only', 'value': 'apical'}, 
                    #{'label': 'Lateral Only', 'value': 'lateral'}

@app.callback(
    Output('track_number_dropdown', 'options'),
    Output('track_number_dropdown', 'value'),
    Input('condition-selection', 'value'), 
    Input('region-selection', 'value')
)
def update_track_dropdown(selected_conditions, selected_regions):
    all_positive_tracks = 'all_positive' in selected_conditions
    only_dynamin_tracks = 'only_dynamin' in selected_conditions
    only_actin_tracks = 'only_actin' in selected_conditions
    both_tracks_positive = 'both_positive' in selected_conditions
    only_basal_tracks = 'basal' in selected_regions 
    only_apical_tracks = 'apical' in selected_regions 
    only_lateral_tracks = 'lateral' in selected_regions 
    all_tracks = 'all' in selected_regions
    relevant_tracks = select_type_of_tracks(filtered_tracks, only_dynamin_tracks, only_actin_tracks, all_positive_tracks, both_tracks_positive)
    final_tracks = select_tracks_region_wise(filtered_tracks, relevant_tracks, only_basal_tracks, only_apical_tracks, only_lateral_tracks, all_tracks)
    options = [{'label': str(track_id), 'value': track_id} for track_id in final_tracks]
    value = options[0]['value'] if options else None
    return options, value

@callback(Output('track_visualization', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = zarr_arr):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type,track_number_dropdown, raw_image = zarr_arr, channel = 2)

@callback(Output('track_visualization_2', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = zarr_arr):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type,track_number_dropdown, raw_image = zarr_arr, channel = 1)

@callback(Output('track_visualization_3', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = zarr_arr):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type, track_number_dropdown, raw_image = zarr_arr, channel = 0)

@callback(Output('intensity_over_time', 'figure'), Input('track_number_dropdown', 'value'), Input('intensity_type', 'value'))
def update_intensity_plot(track_number_dropdown,intensity_type):
    return plot_intensity_over_time(track_of_interest = track_number_dropdown, type_of_intensity=intensity_type)

if __name__ == '__main__':
    app.run_server(debug=True)