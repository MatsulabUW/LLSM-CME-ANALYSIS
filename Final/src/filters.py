import pandas as pd 
import numpy as np 


#Find track length 
#Find track start frame 
#Find track end frame 
#Find peak frame for each column provided (for different channels 1,2,3 like amplitude, c2_peak and c1_peak)
#Find peak frame with respect to start frame 
#Find mean displacement of track 

#Call all of the above functions in one function to give us a dataframe with 
#1. Track id 
#2. Track length 
#3. Track start and end frame number 
#4. Peak frame for intensity across all channels 
#5. Peak frame for intensity across all channels with respect to start frame 
#6. Mean displacement of track 
#7. Apical/Basal/Lateral towards the end 


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

    def calculate_mean_displacement(self):
        displacement = ((self.x - self.x.shift())**2 + (self.y - self.y.shift())**2 + (self.z - self.z.shift())**2)**0.5
        return displacement.mean()
    
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
    return df[df[intensity_peak_frame] > threshold]

def drop_short_tracks(df: pd.DataFrame, threshold: int = 3, track_id_col_name: str = 'track_id',
                      track_length_col_name: str = 'track_length'):
    
    return df[df[track_length_col_name] >= threshold]

def drop_early_peak_tracks(df: pd.DataFrame, intensity_peak_frame: str, cutoff: int, start_frame_col: str = 'track_start'): 
    df = df.copy()
    df['peak_start_relative_to_start_frame'] = df[intensity_peak_frame] - df[start_frame_col]
    df = df[df['peak_start_relative_to_start_frame'] > cutoff]
    df.drop('peak_start_relative_to_start_frame', axis =1, inplace = True)
    return df

def drop_last_frame_peak_tracks(df: pd.DataFrame, intensity_col: str, end_frame_col: str = 'track_end'):
    return df[df[intensity_col] != df[end_frame_col]]