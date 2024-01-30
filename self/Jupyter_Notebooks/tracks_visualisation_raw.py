import pandas as pd
import numpy as np 
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback 
from dash.dependencies import Input, Output 
from skimage import io





##Importing the data

track_df = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/self/files/track_df_updated.pkl')
filtered_tracks = pd.read_pickle('/Users/apple/Desktop/Akamatsu_Lab/Lap_track/self/files/filtered_tracks.pkl')
# Replace 'your_file.tif' with the path to your 4D TIFF file
file_path_3 = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/self/files/Channel3_complete.tif'
# Load the TIFF file using skimage
raw_image_3 = io.imread(file_path_3)
# Replace 'your_file.tif' with the path to your 4D TIFF file
file_path_2 = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/self/files/Channel2_complete.tif'
# Load the TIFF file using skimage
raw_image_2 = io.imread(file_path_2)
# Replace 'your_file.tif' with the path to your 4D TIFF file
file_path_1 = '/Users/apple/Desktop/Akamatsu_Lab/Lap_track/self/files/Channel1_complete.tif'
# Load the TIFF file using skimage
raw_image_1 = io.imread(file_path_1)


##Generate the unique number of tracks 

unique_tracks = filtered_tracks['track_id'].unique()

##Functions 

def max_z_track_visualisation(track_of_interest,raw_image,main_tracking_df):
    
    
    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    #A black image with all pixels set to zero
    empty_layer = np.zeros_like(raw_image_3)

    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['mu_z'], track['mu_y'], track['mu_x']
        #sigma_z, sigma_y, sigma_x = track['sigma_z'], track['sigma_y'], track['sigma_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 3 * sigma_z), int(mu_z + 3 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(raw_image.shape[1], z_end)
        y_end = min(raw_image.shape[2], y_end)
        x_end = min(raw_image.shape[3], x_end)

        # Extract the region from the raw image data
        #print(frame,z_start,z_end, y_start,y_end, x_start,x_end)
        #region_data = raw_image_3[frame, z_start:z_end, y_start:y_end, x_start:x_end]

        # Set everything outside the region to zero
        # Set the region inside the bounding box to the corresponding values in raw_image_data
        empty_layer[frame, z_start:z_end, y_start:y_end, x_start:x_end] = raw_image[frame, z_start:z_end, y_start:y_end, x_start:x_end]
        

    # Assuming 'your_4d_array' is the numpy array with dimensions (Time, z, y, x)

    # Get the shape of the array
    time_points, z_values, y_values, x_values = empty_layer.shape

    # Create an empty 3D array to store the result
    result_array = np.zeros((time_points, y_values, x_values))

    # Loop through each time point
    for t in range(time_points):
        # Calculate the sum along the z-axis
        z_sum = np.sum(empty_layer[t, :, :, :], axis=(1, 2))

        # Find the index of the maximum sum
        max_sum_index = np.argmax(z_sum)

        # Select the slice with the maximum sum
        result_array[t, :, :] = empty_layer[t, max_sum_index, :, :]
        
    
    return result_array

    # Now, 'result_array' contains the 3D array with the maximum sum along the z-axis for each time point


def total_sum_track_visualisation(track_of_interest,raw_image,main_tracking_df):
    
    
    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    #A black image with all pixels set to zero
    empty_layer = np.zeros_like(raw_image_3)

    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['mu_z'], track['mu_y'], track['mu_x']
        #sigma_z, sigma_y, sigma_x = track['sigma_z'], track['sigma_y'], track['sigma_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 3 * sigma_z), int(mu_z + 3 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(raw_image.shape[1], z_end)
        y_end = min(raw_image.shape[2], y_end)
        x_end = min(raw_image.shape[3], x_end)

        # Extract the region from the raw image data
        #print(frame,z_start,z_end, y_start,y_end, x_start,x_end)
        #region_data = raw_image_3[frame, z_start:z_end, y_start:y_end, x_start:x_end]

        # Set everything outside the region to zero
        # Set the region inside the bounding box to the corresponding values in raw_image_data
        empty_layer[frame, z_start:z_end, y_start:y_end, x_start:x_end] = raw_image[frame, z_start:z_end, y_start:y_end, x_start:x_end]
        

    # Assuming 'your_4d_array' is the numpy array with dimensions (Time, z, y, x)

    # Get the shape of the array
    time_points, z_values, y_values, x_values = empty_layer.shape
    
    # Create an empty array to store the maximum intensity projections in each time 
    
    total_sum_movie = []
    # Loop through each time point
    for t in range(time_points):
        z_sum = np.sum(empty_layer[t],axis=0)
        total_sum_movie.append(z_sum)
    
    total_sum_movie = np.array(total_sum_movie)
    return total_sum_movie


##Maximum Intensity Projection Function 
def max_intensity_projection_track_visualisation(track_of_interest,raw_image,main_tracking_df):
    
    
    current_track = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    #A black image with all pixels set to zero
    empty_layer = np.zeros_like(raw_image)

    # Loop through tracks and set values in the volume
    for index, track in current_track.iterrows():
        frame, mu_z, mu_y, mu_x = int(track['frame']), track['mu_z'], track['mu_y'], track['mu_x']
        #sigma_z, sigma_y, sigma_x = track['sigma_z'], track['sigma_y'], track['sigma_x']
        sigma_z = 4
        sigma_y = 2
        sigma_x = 2

        # Define the bounding box based on center and sigma
        z_start, z_end = int(mu_z - 3 * sigma_z), int(mu_z + 3 * sigma_z)
        y_start, y_end = int(mu_y - 3 * sigma_y), int(mu_y + 3 * sigma_y)
        x_start, x_end = int(mu_x - 3 * sigma_x), int(mu_x + 3 * sigma_x)

        # Clip the coordinates to be within the image bounds
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        z_end = min(raw_image.shape[1], z_end)
        y_end = min(raw_image.shape[2], y_end)
        x_end = min(raw_image.shape[3], x_end)

        # Extract the region from the raw image data
        #print(frame,z_start,z_end, y_start,y_end, x_start,x_end)
        #region_data = raw_image_3[frame, z_start:z_end, y_start:y_end, x_start:x_end]

        # Set everything outside the region to zero
        # Set the region inside the bounding box to the corresponding values in raw_image_data
        empty_layer[frame, z_start:z_end, y_start:y_end, x_start:x_end] = raw_image[frame, z_start:z_end, y_start:y_end, x_start:x_end]
        

    # Assuming 'your_4d_array' is the numpy array with dimensions (Time, z, y, x)

    # Get the shape of the array
    time_points, z_values, y_values, x_values = empty_layer.shape
    
    # Create an empty array to store the maximum intensity projections in each time 
    
    mip_movie = []
    # Loop through each time point
    for t in range(time_points):
        max_slice = np.max(empty_layer[t],axis=0)
        mip_movie.append(max_slice)
    
    mip_movie = np.array(mip_movie)
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

def plot_raw_movie(plot_type = 'max_intensity_projection', track_number = unique_tracks[0], raw_image = raw_image_3, main_tracking_df = track_df):
    
    if plot_type == 'max_intensity_projection':      
        result_array = max_intensity_projection_track_visualisation(track_number,raw_image,main_tracking_df)
        track_array = crop_movie(result_array)
    elif plot_type == 'max_z_slice':      
        result_array = max_z_track_visualisation(track_number,raw_image,main_tracking_df)
        track_array = crop_movie(result_array)
    elif plot_type == 'total_z_sum':      
        result_array = total_sum_track_visualisation(track_number,raw_image,main_tracking_df)
        track_array = crop_movie(result_array)
    
    length_of_track = len(track_array)
    # Set the number of rows and columns for subplots
    num_cols = 7
    num_rows = length_of_track // num_cols + 1

    fig = make_subplots(rows=num_rows, cols=7)

    r = 1
    c = 1
    for i in range(len(track_array)):
        image = px.imshow(track_array[i])
        fig.add_trace(image.data[0], row = r, col = c)
        if i != 0 and (i+1) % (num_cols) == 0: 
            r = r + 1
            c = 1
        else: 
            c = c + 1
    r = 1 
    c = 1

    return fig

#Line chart plot
def plot_intensity_over_time(track_of_interest = unique_tracks[0], main_tracking_df = track_df):
    current_track_df = main_tracking_df[main_tracking_df['track_id'] == track_of_interest]
    # Create Line plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=current_track_df['frame'], y=current_track_df['amplitude'], name = 'Channel 3',
                 line = dict(color = 'red', width = 4)))
    fig.add_trace(go.Scatter(x=current_track_df['frame'], y=current_track_df['c2_peak'],name = 'Channel 2', 
                            line=dict(color='green', width = 4)))

    # Edit the layout
    fig.update_layout(title='Intensity Over Time', title_x = 0.5,
                       xaxis_title='Frames',  xaxis_color = 'black', title_font = dict(color = 'black', size = 30),
                       yaxis_title='Amplitude', yaxis_color = 'black', legend=dict(bgcolor=None),autosize = True, plot_bgcolor = None,
                     paper_bgcolor = None)


    return fig 





app = Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Raw Intensity Visualization Dashboard", style={"text-align": "center"}),
    html.Br(),
    html.Div(
        children = [
    html.Label('Select the Track Number:'),
    dcc.Dropdown(
        id='track_number_dropdown',
        #options=[{'label': str(option), 'value': option} for option in track_number_options],
        options = filtered_tracks['track_id'].unique(),
        value= filtered_tracks['track_id'].unique()[0],
        #style={'width': '50%'}
    ),
    html.Label('Select the type of feature to display:'),
    dcc.Dropdown(
        id = 'display_type', 
        options = ['max_intensity_projection', 'max_z_slice', 'total_z_sum'], 
        value = 'max_intensity_projection', 
        #style = {'width': '50%'}
    ),
    html.Br(),html.Label('Channel 3 Track'),
    dcc.Graph(id='track_visualization', figure=plot_raw_movie()),
        ],
style={"display":"inline-block", "width":"48%",'text-align': 'center'}
    ),
    
    html.Div(children=[ 
    html.Br(),html.Label('Channel 2 Track'),
    dcc.Graph(id='track_visualization_2', figure=plot_raw_movie()),
    ],
style={"display":"inline-block", "width":"48%", "text-align":"center"}
    ),

    html.Div(children = [ 
    html.Br(), html.Label('Channel 1 Track'),
    dcc.Graph(id='track_visualization_3', figure=plot_raw_movie()),
    ],
style={ "display":"inline-block", "width":"48%", 'text-align': 'center'}    
    ),

    html.Div(children=[
        html.Br(), html.Label('Intensity Over time plot'),
    dcc.Graph(id='intensity_over_time', figure = plot_intensity_over_time())
    ],
style={ 'border': '2px solid black', "display":"inline-block","width":"48%", 'text-align': 'center'}  
    ),

],
style = {'backgroundColor': 'lightgray', "padding": "50px"}
)

@callback(Output('track_visualization', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = raw_image_3):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type,track_number_dropdown, raw_image = raw_image_3)

@callback(Output('track_visualization_2', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = raw_image_2):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type,track_number_dropdown, raw_image = raw_image_2)

@callback(Output('track_visualization_3', 'figure'),[Input('display_type', 'value'),Input('track_number_dropdown', 'value')])
def update_graph(display_type, track_number_dropdown, raw_image = raw_image_1):
    # Call your plotting function with the selected options
    return plot_raw_movie(display_type, track_number_dropdown, raw_image = raw_image_1)

@callback(Output('intensity_over_time', 'figure'), Input('track_number_dropdown', 'value'))
def update_intensity_plot(track_number_dropdown):
    return plot_intensity_over_time(track_number_dropdown)

if __name__ == '__main__':
    app.run_server(debug=True)