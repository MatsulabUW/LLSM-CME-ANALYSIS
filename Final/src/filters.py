import pandas as pd 
import numpy as np 
from skimage import io
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter


'''
The class below is used to get relevant features of the tracks in an organised manner. More features can be added in this class
'''

class Track:
    def __init__(self, track_id, frames, x, y, z, intensities):
        self.track_id = track_id
        self.frames = frames
        self.x = x
        self.y = y
        self.z = z
        self.intensities = intensities
        self.track_length = len(frames)
        self.track_start = frames.min()
        self.track_end = frames.max()
        self.peak_intensities = self.calculate_peak_intensities()
        self.peak_intensity_frames = self.calculate_peak_intensity_frame()
        self.mean_displacement_track = self.calculate_mean_displacement()
        self.mean_z_value = z.mean()
        self.mean_z_displacement = self.calculate_mean_z_displacement()
        #self.boundary = apical/basal/lateral, use mean z for this 

    def calculate_mean_displacement(self):
        displacement = ((self.x - self.x.shift())**2 + (self.y - self.y.shift())**2 + (self.z - self.z.shift())**2)**0.5
        return displacement.mean()
    
    def calculate_mean_z_displacement(self):
        displacement_z = abs(self.z - self.z.shift())
        return displacement_z.mean()
    
    def print_intensities(self):
        return self.intensities
    
    def calculate_peak_intensities(self):
        peaks_intensitiy_values = []
        for i in range(self.intensities.shape[1]):
            peaks_intensitiy_values.append(max(self.intensities[:,i]))
        return peaks_intensitiy_values
    
    def print_peak_intensities(self):
        for i in range(self.intensities.shape[1]):
            print('peaks: ', max(self.intensities[:,i]))
    
    def calculate_peak_intensity_frame(self):
        peak_frames = []
        for i in range(self.intensities.shape[1]):
            index = np.argmax(self.intensities[:,i])
            peak_frames.append(index + self.track_start)
        return peak_frames
    
    def print_peak_intensity_frame(self):
        for i in range(self.intensities.shape[1]):
            index = np.argmax(self.intensities[:,i])
            ans = index + self.track_start
            print(ans)



def create_tracks_from_dataframe(df: pd.DataFrame, track_id_col_name: str = 'track_id', frame_col_name: str = 'frame',
                                 coords: list = ['mu_x', 'mu_y', 'mu_z'], intensities_col_name: list = ['amplitude', 'c2_peak']):
    '''
    Create tracks from a pandas DataFrame.

    Parameters:
    1. df (pd.DataFrame): DataFrame containing the data.
    2. track_id_col_name (str): Name of the column containing track IDs. Default is 'track_id'.
    3. frame_col_name (str): Name of the column containing frame numbers. Default is 'frame'.
    4. coords (list): List of column names containing spatial coordinates (x, y, z). Default is ['mu_x', 'mu_y', 'mu_z'].
    5. intensities_col_name (list): List of column names containing intensities. Default is ['amplitude', 'c2_peak'].

    Output:
    1. list: A list of Track objects created from the DataFrame.
    '''
    tracks = []
    for track_id, group in df.groupby(track_id_col_name):
        track = Track(
            track_id = group[track_id_col_name],
            frames = group[frame_col_name],
            x = group[coords[0]],
            y = group[coords[1]],
            z = group[coords[2]],
            intensities = group[intensities_col_name].values
        )
        tracks.append(track)
    return tracks


def drop_tracks_below_intensity(df: pd.DataFrame, threshold: float, intensity_peak_frame: str):
    '''
    Filter out tracks with intensity values below a specified threshold.

    Parameters:
    1. df (pd.DataFrame): DataFrame containing the data.
    2. threshold (float): Threshold value for intensity.
    3. intensity_peak_frame (str): Name of the column containing intensity peak frame values.

    Outputs:
    1. pd.DataFrame: Filtered DataFrame containing tracks with intensity values above the threshold.
    '''

    return df[df[intensity_peak_frame] > threshold]

def drop_short_tracks(df: pd.DataFrame, threshold: int = 3, track_id_col_name: str = 'track_id',
                      track_length_col_name: str = 'track_length'):
    """
    Filter out tracks with lengths shorter than a specified threshold.

    Parameters:
    1. df (pd.DataFrame): DataFrame containing the data.
    2. threshold (int): Threshold value for track length. Default is 3.
    3. track_id_col_name (str): Name of the column containing track IDs. Default is 'track_id'.
    4. track_length_col_name (str): Name of the column containing track lengths. Default is 'track_length'.

    Outputs:
    1. pd.DataFrame: Filtered DataFrame containing tracks with lengths greater than or equal to the threshold.
    """
    return df[df[track_length_col_name] >= threshold]

def drop_early_peak_tracks(df: pd.DataFrame, intensity_peak_frame: str, cutoff: int, start_frame_col: str = 'track_start'): 
    """
    Filter out tracks with intensity peaks occurring too early relative to the start frame.

    Parameters:
    1. df (pd.DataFrame): DataFrame containing the data.
    2. intensity_peak_frame (str): Name of the column containing intensity peak frame values.
    3. cutoff (int): Cutoff value for early peak occurrence.
    4. start_frame_col (str): Name of the column containing start frame values. Default is 'track_start'.

    Outputs:
    1. pd.DataFrame: Filtered DataFrame containing tracks with intensity peaks occurring later than the cutoff.
    """

    df = df.copy()
    df['peak_start_relative_to_start_frame'] = df[intensity_peak_frame] - df[start_frame_col]
    df = df[df['peak_start_relative_to_start_frame'] > cutoff]
    df.drop('peak_start_relative_to_start_frame', axis =1, inplace = True)
    return df

def drop_last_frame_peak_tracks(df: pd.DataFrame, intensity_col: str, end_frame_col: str = 'track_end'):
    """
    Filter out tracks where intensity peaks occur on the last frame.

    Parameters:
    1. df (pd.DataFrame): DataFrame containing the data.
    2. intensity_col (str): Name of the column containing intensity values.
    3. end_frame_col (str): Name of the column containing end frame values. Default is 'track_end'.

    Outputs:
    1. pd.DataFrame: Filtered DataFrame containing tracks without intensity peaks on the last frame.
    """
    return df[df[intensity_col] != df[end_frame_col]]

def plot_z_sum(file_path: str):
    '''
    This function takes in an entire time series image and prints a graph of voxel sum across z slices for few selected time frames. This is an helper function
    for determining the apical basal and lateral boundaries. 

    Parameters:
    1. file_path: type(str), file path of the image 

    Output: 
    1. plots graph of sum of intensity over each z slice for different frames
    '''

    # Load the TIFF file using skimage
    c1_raw = io.imread(file_path)
    lower_frame = (c1_raw.shape[0] * 0.25) // 1
    center_frame = (c1_raw.shape[0] * 0.5) // 1
    upper_frame = (c1_raw.shape[0] * 0.75) // 1
    times_to_plot = []
    times_to_plot.append(0)
    times_to_plot.append(int(lower_frame))
    times_to_plot.append(int(center_frame))
    times_to_plot.append(int(upper_frame))
    times_to_plot.append(c1_raw.shape[0]-1)
    all_frame_sum = []

    # Generate x-axis values (index values from sum_of_pixels)
    x_values = np.arange(c1_raw.shape[1])
            
    for frame in times_to_plot:
        c1_raw_frame = c1_raw[frame]
        # Initialize an empty list to store the sum of pixel values for each z value
        sum_of_pixels = []
        # Iterate over the z-axis of the 3D numpy array
        for z in range(c1_raw_frame.shape[0]):
            # Calculate the sum of pixel values for the current z value
            sum_of_pixels_z = np.sum(c1_raw_frame[z,:, :])
            # Append the sum to the list
            sum_of_pixels.append(sum_of_pixels_z)
        all_frame_sum.append(sum_of_pixels)
    
    for i in range(len(all_frame_sum)):
        colors = ['red', 'green', 'blue', 'orange', 'pink']
        plt.plot(x_values, all_frame_sum[i], color = colors[i], label = f'frame {times_to_plot[i]}')
        # Format y-axis ticks to display whole values
    
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
    plt.legend()
    plt.xlabel('Z slice')
    plt.ylabel('Pixels sum')
    plt.title('Pixel sum over Z slices')

# Function to allocate membrane regions
def allocate_membrane_regions(df: pd.DataFrame, basal_range: list, lateral_range: list, apical_range: list,  mean_z_col: str = 'mean_z'):
    '''
    Parameters:
    1. df, type(DataFrame): contains the dataframe which has the mean z value for each track 
    2. basal_range, type(list): contains the lower and upper bound for basal part of the membrane. Where lower value is included and upper is excluded
    3. lateral_range, type(list): contains the lower and upper bound for lateral part of the membrane. Where lower value is included and upper is excluded
    4. apical_range, type(list): contains the lower and upper bound for apical part of the membrane. Where lower and upper both are included
    5. mean_z_col, type(str): name of the column which contains the mean z value for each track 

    Output: 
    1. df, type(DataFrame): modifies the original dataframe and adds a column membrane region
    '''
    membrane_regions = []

    # Iterate over each row of the DataFrame
    for index, row in df.iterrows():
        z_value = row[mean_z_col]

        
         # Check if z_value falls within basal ranges
        if basal_range[0] <= z_value < basal_range[1]:
            membrane_regions.append('Basal')

        # Check if z_value falls within lateral ranges
        elif lateral_range[0] <= z_value < lateral_range[1]:
            membrane_regions.append('Lateral')
        
                # Check if z_value falls within apical ranges
        elif apical_range[0] <= z_value <= apical_range[1]:
            membrane_regions.append('Apical')
        
        else:
            print('Inside the else function', index)
            print(row)


    df['membrane_region'] = membrane_regions
    return df