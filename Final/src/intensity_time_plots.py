import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def filter_track_ids_by_length_ranges(dataframe: pd.DataFrame, track_length_buckets: list, 
                                      track_id_col_name: str = 'track_id', track_length_col_name: str = 'track_length'):

    '''
    The function takes in buckets of tracks for defined length ranges and returns the track id's within that range 

    Parameters: 
    1. dataframe: type(DataFrame), the dataframe which contains the length of each track  
    2. track_length_buckets: type(list) 2-D list, this contains a range of length of tracks on the basis of which a track id 
    will be assigned to a cohort. Both the lower and upper limit are inclusive 
    3. track_id_col_name: type(str), the column name which contains the frame number. Default value: 'track_id'
    4. track_length_col_name: type(str), the column name which contains the length of the track. Default value: 'track_length'
    
    Returns: 
    1. track_id_arrays: type(list), returns a 2-D list which contains track_ids for each bucket
    '''

    # Create an empty dictionary to store track IDs for each length range
    track_ids_dict = {f'Length_{low}_{high}': [] for low, high in track_length_buckets}
    
    # Iterate through each row in the DataFrame
    for _, row in dataframe.iterrows():
        track_id = row[track_id_col_name]
        track_length = row[track_length_col_name]
        
        # Check if track_length falls within any of the specified ranges
        for i, (low, high) in enumerate(track_length_buckets):
            if low <= track_length <= high:
                key = f'Length_{low}_{high}'
                track_ids_dict[key].append(track_id)
    
    # Convert the dictionary values to arrays
    track_id_arrays = [track_ids_dict[key] for key in track_ids_dict]
    
    return track_id_arrays


def random_track_ids(dataframe: pd.DataFrame, desired_length: list, track_length_col_name: str, 
                      track_id_col_name: str, num_to_select=16):
    '''
    This function Randomly selects track_ids of a desired length from a DataFrame.

    Parameters:
    1. dataframe: type(DataFrame), the dataframe which contains the track id and length of each track
    2. desired_length: type(list), The desired length of tracks to select.
    3. track_length_col_name: type(str), the column name which contains the length of the track. Default value: 'track_length'
    4. track_id_col_name: type(str), the column name which contains the frame number. Default value: 'track_id'
    5. num_to_select: type(int), The number of track_ids to randomly select (default is 16).

    Returns:
    1. selected_track_ids: type(array), returns the list of randomly seleted track id's 
    '''

    # Filter the DataFrame for tracks with the desired length
    tracks_of_desired_length = dataframe[(dataframe[track_length_col_name] >= desired_length[0]) & 
                                  (dataframe[track_length_col_name] <= desired_length[1]) ]

    # Determine the number of tracks available for the desired length
    num_tracks_available = len(tracks_of_desired_length)

    # Determine the number of tracks to randomly select (minimum of num_to_select or available tracks)
    num_tracks_to_select = min(num_to_select, num_tracks_available)

    if num_tracks_to_select > 0:
        # Randomly select track_ids
        selected_track_ids = np.random.choice(tracks_of_desired_length[track_id_col_name],
                                              num_tracks_to_select, replace=False)
        return selected_track_ids
    else:
        print(f"No tracks of length {desired_length} available.")
        return []
    
def intensity_time_plot(dataframe: pd.DataFrame, tracks_to_plot: np.ndarray, track_id_col_name: str, 
                       frame_col_name: str, intensity_to_plot: list, channels_to_plot: int, legend_values: list = ['Channel 3', 'Channel 2', 'Channel 1'], 
                       line_colors: list = ['red', 'green', 'blue'], graph_title = 'Intensity', normalized: bool = False):
    
    '''
    
    The function takes in the list of track ids assigned to a cohort and aligns them with respect to their 
    peak values to a specific index. The index depends on bufferZero. Whatever the value of bufferZero is the
    peaks will be aligned with that value. 
    
    Parameters:
    1. dataframe: type(DataFrame), this is the raw dataframe which contains all of the relevant intensity values
    1. tracks_to_plot: type(list), contains the tracks id's of tracks within the specified length range
    2. backgroundIntensity: type(list), the background intensity of the movie 
    4. intensity_to_plot, type(list), the name of the columns to be used for plotting intensity. Note that the first value
    in this list should be for the primary channel, second value for the secondary channel and so on. 
    Default Value: ['amplitude', 'c2_peak']
    5. track_id_col_name: type(str), the column name which contains the frame number. Default value: 'track_id'
    6. legend_values: type(list), labels for the legend. Must be in line with intensity_to_plot list. The channel that is first in intensity_to_plot
    must also appear first in legend labels. 
    7. line_colors: type(list), colors for the intensity_to_plot lines. line_colors[0] will be the color for intensity_to_plot[0] and so on. 
    8. normalized: type(bool), if the intensity values are normalized. Default value: False

    Output: 
    1. Returns the array of primary channel tracks aligned with respect to secondary channel peaks
    2. Returns the array of secondary channel tracks where tracks are aligned with all peaks of tracks being on 
    the same index. (the index is determined by bufferZero)
    
    '''


    if channels_to_plot == 3 and len(intensity_to_plot) != 3: 
        raise ValueError('length of intensity_to_plot must equal channels_to_plot')
        
    num_rows = 8
    num_cols = 8
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 18))

    handles = []
    labels = []

    for i, track_id in enumerate(tracks_to_plot):
        # Filter DataFrame for the selected track_id
        track_data = dataframe[dataframe[track_id_col_name] == track_id].copy(deep = True)

        # Set x-axis range from 1 to 20
        x_range = np.arange(1, 21)

        # Plot intensities on the same subplot
        row = i // num_cols
        col = i % num_cols
        #length = filtered_df[filtered_df['track_id'] == track_id]['track_length'].values
        length = track_data.shape[0]

        lines = []
        for i in range(channels_to_plot):
            min_val = track_data[intensity_to_plot[i]].min()
            track_data[intensity_to_plot[i]] -= min_val
            max_val = track_data[intensity_to_plot[i]].max()          

            # normalize by max_intensity
            if normalized:  
                track_data[intensity_to_plot[i]] /= max_val
            line, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[i]], label=intensity_to_plot[i], color=line_colors[i])
            lines.append(line)

        # if channels_to_plot == 2:
        #     min0 = track_data[intensity_to_plot[0]].min()
        #     min1 = track_data[intensity_to_plot[1]].min()
        #     track_data[intensity_to_plot[0]] = track_data[intensity_to_plot[0]] - min0
        #     track_data[intensity_to_plot[1]] = track_data[intensity_to_plot[1]] - min1
        #     line1, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[0]], label=intensity_to_plot[0], color = line_colors[0])
        #     line2, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[1]], label=intensity_to_plot[1], color = line_colors[1])
        # else: 
        #     min0 = track_data[intensity_to_plot[0]].min()
        #     min1 = track_data[intensity_to_plot[1]].min()
        #     min2 = track_data[intensity_to_plot[2]].min()
        #     track_data[intensity_to_plot[0]] = track_data[intensity_to_plot[0]] - min0
        #     track_data[intensity_to_plot[1]] = track_data[intensity_to_plot[1]] - min1
        #     track_data[intensity_to_plot[2]] = track_data[intensity_to_plot[2]] - min2
        #     line1, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[0]], label=intensity_to_plot[0], color = line_colors[0])
        #     line2, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[1]], label=intensity_to_plot[1],  color = line_colors[1])
        #     line3, = axes[row, col].plot(track_data[frame_col_name], track_data[intensity_to_plot[2]], label=intensity_to_plot[2], color = line_colors[2])
        axes[row, col].set_title(f'Track {track_id}, length {length}')
        axes[row, col].set_xlabel('Frame')
        axes[row, col].set_ylabel('Intensity')
        #axes[row, col].legend()

    fig.suptitle(f'{graph_title} over time', fontsize = 22, fontweight = 'bold')

    for i in range(channels_to_plot):
        handles.append(lines[i])
        labels.append(legend_values[i])
    # if (channels_to_plot == 2):
    #     # Set a single legend for all subplots
    #     handles.extend([line1, line2])
    #     labels.extend([legend_values[0], legend_values[1]])
    # else: 
    #     # Set a single legend for all subplots
    #     handles.extend([line1, line2, line3])
    #     labels.extend([legend_values[0], legend_values[1], legend_values[2]])
        
    # Set a single legend for all subplots
    fig.legend(handles, labels, loc='upper right', ncol=channels_to_plot, fontsize=18)
    # Adjust layout for better spacing
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.5)
    #fig.suptitle(f'Intensity over time \n Channel 3 color: red \n Channel 2 color: green \n Channel 1 color: blue', fontsize=18, fontweight='bold')
    #plt.subplots_adjust(top=0.90)
    plt.show()

##Below code is copied from PyLattice
def createBufferForLifetimeCohort(dataframe: pd.DataFrame ,listOfTrackIdsAssignedToCohort: list, backgroundIntensity: list, 
                                  intensity_to_plot: list = ['amplitude', 'c2_peak'], track_id_col_name: str = 'track_id'):
    
    '''
    
    The function takes in the list of track ids assigned to a cohort and aligns them with respect to their 
    peak values to a specific index. The index depends on bufferZero. Whatever the value of bufferZero is the
    peaks will be aligned with that value. 
    
    Parameters:
    1. dataframe: type(DataFrame), this is the raw dataframe which contains all of the relevant intensity values
    1. listOfTrackIdsAssignedToCohort: type(list), contains the tracks id's of tracks within the specified length range
    2. backgroundIntensity: type(list), the background intensity of the movie 
    4. intensity_to_plot, type(list), the name of the columns to be used for plotting intensity. Note that the first value
    in this list should be for the primary channel, second value for the secondary channel and so on. 
    Default Value: ['amplitude', 'c2_peak']
    5. track_id_col_name: type(str), the column name which contains the frame number. Default value: 'track_id'
    
    Output: 
    1. Returns the array of primary channel tracks aligned with respect to secondary channel peaks
    2. Returns the array of secondary channel tracks where tracks are aligned with all peaks of tracks being on 
    the same index. (the index is determined by bufferZero)
    2. Returns the array of tertiary channel tracks where tracks are aligned with all peaks of tracks being on 
    the same index. (the index is determined by bufferZero) (Only returned if intensity_to_plot is of length three)
    
    '''


    trackIdArray = listOfTrackIdsAssignedToCohort
    
    bufferSize = 200
    bufferZero = 100

    if len(backgroundIntensity) != len(intensity_to_plot):
        raise ValueError('Dimensions of intensity to plot and background intensity must be same')

    if len(intensity_to_plot) == 1:
        raise ValueError('Minimum acceptable dimensions are 2')

    elif len(intensity_to_plot) == 2:

        p_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[0],dtype=float)
        s_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[1],dtype=float)
        
        counter = 0
        
        for trackId in trackIdArray:
            track = dataframe[dataframe[track_id_col_name] == trackId]
            p_intensity = track[intensity_to_plot[0]].values.astype(float) #primary (channel 3 in our case)
            s_intensity = track[intensity_to_plot[1]].values.astype(float) #secondary  (channel 2 in our case)
            maxIdx = np.argmax(s_intensity) # was s_intensity before
        
            for i in range(0,len(track)):
                if(not np.isnan(p_intensity[i])):
                    p_buffer[counter][bufferZero-maxIdx+i]=(p_intensity[i])
                if(not np.isnan(s_intensity[i])):
                    s_buffer[counter][bufferZero-maxIdx+i]=(s_intensity[i])

                    
            counter = counter+1;
        
        return p_buffer,s_buffer
    
    elif len(intensity_to_plot) == 3:

        p_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[0],dtype=float)
        s_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[1],dtype=float)
        t_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[2],dtype=float)
        
        counter = 0
        
        for trackId in trackIdArray:
            track = dataframe[dataframe[track_id_col_name] == trackId]
            p_intensity = track[intensity_to_plot[0]].values.astype(float) #primary (channel 3 in our case)
            s_intensity = track[intensity_to_plot[1]].values.astype(float) #secondary  (channel 2 in our case)
            t_intensity = track[intensity_to_plot[2]].values.astype(float) #tertiary (channel 1 in our case)
            

            # align by peak of secondary channel
            maxIdx = np.argmax(s_intensity)
            
            # align by peak of tertiary channel
            # maxIdx = np.argmax(t_intensity)

            
        
            for i in range(0,len(track)):
                if(not np.isnan(p_intensity[i])):
                    p_buffer[counter][bufferZero-maxIdx+i]=(p_intensity[i])
                if(not np.isnan(s_intensity[i])):
                    s_buffer[counter][bufferZero-maxIdx+i]=(s_intensity[i])
                if(not np.isnan(t_intensity[i])):
                    t_buffer[counter][bufferZero-maxIdx+i]=(t_intensity[i])

            
                    
            counter = counter+1;

    
    
        return p_buffer,s_buffer, t_buffer



def cumulative_plots(buffers: list, background_intensity: list, time_shift: int, colors: list = ['magenta', 'lime', 'blue'], 
                     graph_title: str = 'Average Amplitude over time', axis_labels: list = ['Time', 'Amplitude'], framerate_msec: float = 2.3 *1000,
                     legend_vals: list = ['Primary Channel (Clathrin)', 'Secondary Channel (Dynamin)', 'Tertiary Channel (Actin)']):
    
    '''
    Parameters:

    1. buffers: type(list), this is a list of arrays. The arrays in the list are supposed to be the ones which are returned from the function 
    createBufferForLifetimeCohort. These will be used for the plots 
    2. background_intensity: type(list), this a list of ints. Must be the same as used in the function createBufferForLifetimeCohort to get normal results 
    3. time_shift: type(int), used for aligning the x-axis as needed 
    4. colors: type(list), color for each channel. must be passed in the order and dimension matching buffers. default ['magenta', 'lime', 'blue']
    5. graph_title: type(str), title for the graph. default 'Average Amplitude over time'
    6. axis_labels: type(list), order ['x-axis label', 'y-axis label']. default ['Time', 'Amplitude']
    7. framerate_msec: type(float), the framerate of the original movie in milli seconds. default 2.3 
    8. legend_vals: type(list), values to be used for the legend. default ['Primary Channel (Clathrin)', 'Secondary Channel (Dynamin)', 'Tertiary Channel (Actin)']

    Note: 
    buffers, background_intensity, colors, legend vals, all must have the same dimensions

    Output: 
    Prints graph 
    
    '''
    
    

    plt.figure(dpi=300)

    bufferSize = 200
    bufferZero = 100

    timeShift = np.array([0,30,70,120]) + time_shift
    alph = 0.05
    liwi = 0.5

    if len(buffers) != len(colors):
        raise ValueError("dimensions of buffers list and colors list are not equal")

    if len(buffers) == 1:
        raise ValueError("minimum length 2 and maximum length 3 needed for buffers for plotting")

    elif len(buffers) == 2:
        p_buffer = buffers[0]
        s_buffer = buffers[1]
        p_buffer_average = np.nanmean(p_buffer,axis=0)-background_intensity[0]
        s_buffer_average = np.nanmean(s_buffer,axis=0)-background_intensity[1]
        p_buffer_std = np.nanstd(p_buffer,axis=0)
        s_buffer_std = np.nanstd(s_buffer,axis=0)
        time = framerate_msec/1000*(np.array(range(0,bufferSize))-bufferZero)+timeShift[0]

        plt.plot(time, p_buffer_average, c='black', lw=liwi+2, zorder=1)  # Border line
        plt.plot(time,p_buffer_average,c=colors[0],lw=liwi+1, label = legend_vals[0], zorder =2)
        plt.fill_between(time,p_buffer_average-p_buffer_std,p_buffer_average+p_buffer_std,facecolor=colors[0],alpha=0.08)

        plt.plot(time, s_buffer_average, c='black', lw=liwi+2, zorder=1)  # Border line
        plt.plot(time,s_buffer_average,c=colors[1],lw=liwi+1, label = legend_vals[1], zorder =2)
        plt.fill_between(time,s_buffer_average-s_buffer_std,s_buffer_average+s_buffer_std,facecolor=colors[1],alpha=0.08)

    elif len(buffers) == 3:
        p_buffer = buffers[0]
        s_buffer = buffers[1]
        t_buffer = buffers[2]
        p_buffer_average = np.nanmean(p_buffer,axis=0)-background_intensity[0]
        s_buffer_average = np.nanmean(s_buffer,axis=0)-background_intensity[1]
        t_buffer_average = np.nanmean(t_buffer,axis=0)-background_intensity[2]
        p_buffer_std = np.nanstd(p_buffer,axis=0)
        s_buffer_std = np.nanstd(s_buffer,axis=0)
        t_buffer_std = np.nanstd(t_buffer,axis=0)
        time = framerate_msec/1000*(np.array(range(0,bufferSize))-bufferZero)+timeShift[0]

        plt.plot(time, p_buffer_average, c='black', lw=liwi+2, zorder=1)  # Border line
        plt.plot(time,p_buffer_average,c=colors[0],lw=liwi+1, label = legend_vals[0], zorder =2)
        plt.fill_between(time,p_buffer_average-p_buffer_std,p_buffer_average+p_buffer_std,facecolor=colors[0],alpha=0.08)

        plt.plot(time, s_buffer_average, c='black', lw=liwi+2, zorder=1)  # Border line
        plt.plot(time,s_buffer_average,c=colors[1],lw=liwi+1, label = legend_vals[1], zorder =2)
        plt.fill_between(time,s_buffer_average-s_buffer_std,s_buffer_average+s_buffer_std,facecolor=colors[1],alpha=0.08)

        plt.plot(time, t_buffer_average, c='black', lw=liwi+2, zorder=1)  # Border line
        plt.plot(time,t_buffer_average,c=colors[2],lw=liwi+1, label = legend_vals[2], zorder =2)
        plt.fill_between(time,t_buffer_average-t_buffer_std,t_buffer_average+t_buffer_std,facecolor=colors[2],alpha=0.08)



    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(graph_title)
    plt.legend(fontsize=6)
    
    plt.xlim(-5,200)



def createBufferForLifetimeCohort_normalized(dataframe: pd.DataFrame ,listOfTrackIdsAssignedToCohort: list, backgroundIntensity: list, 
                                  intensity_to_plot: list, track_id_col_name: str = 'track_id'):
    
    """
    Normalizes the intensity values for each track and creates a buffer for lifetime cohorts.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing the track data.
    listOfTrackIdsAssignedToCohort : list
        A list of track IDs assigned to the cohort.
    backgroundIntensity : list
        A list of background intensity values for each channel.
    intensity_to_plot : list
        A list of column names for the intensity values to be plotted.
    track_id_col_name : str, optional
        The column name for the track IDs in the DataFrame, by default 'track_id'.

    Returns
    -------
    tuple
        If the length of intensity_to_plot is 2, returns a tuple containing two buffers for the primary and secondary intensities.
        If the length of intensity_to_plot is 3, returns a tuple containing three buffers for the primary, secondary, and tertiary intensities.

    Raises
    ------
    ValueError
        If the dimensions of intensity_to_plot and backgroundIntensity are not the same.
        If the length of intensity_to_plot is less than 2.
    """
    
    trackIdArray = listOfTrackIdsAssignedToCohort
    
    bufferSize = 200
    bufferZero = 100

    if len(backgroundIntensity) != len(intensity_to_plot):
        raise ValueError('Dimensions of intensity to plot and background intensity must be same')

    if len(intensity_to_plot) == 1:
        raise ValueError('Minimum acceptable dimensions are 2')

    elif len(intensity_to_plot) == 2:

        p_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[0],dtype=float)
        s_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[1],dtype=float)
        
        counter = 0
        
        for trackId in trackIdArray:
            track = dataframe[dataframe[track_id_col_name] == trackId]
            p_intensity = track[intensity_to_plot[0]].values.astype(float) #primary (channel 3 in our case)
            s_intensity = track[intensity_to_plot[1]].values.astype(float) #secondary  (channel 2 in our case)
            maxIdx = np.argmax(s_intensity) # was s_intensity before
            p_maxIntensity = np.nanmax(p_intensity)
            s_maxIntensity = np.nanmax(s_intensity)
        
            for i in range(0,len(track)):
                if(not np.isnan(p_intensity[i])):
                    p_buffer[counter][bufferZero-maxIdx+i]=(p_intensity[i]) / p_maxIntensity
                if(not np.isnan(s_intensity[i])):
                    s_buffer[counter][bufferZero-maxIdx+i]=(s_intensity[i]) / s_maxIntensity

                    
            counter = counter+1;
        
        return p_buffer,s_buffer
    
    elif len(intensity_to_plot) == 3:

        p_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[0],dtype=float)
        s_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[1],dtype=float)
        t_buffer = np.full(( len(trackIdArray),bufferSize), backgroundIntensity[2],dtype=float)
        
        counter = 0
        
        for trackId in trackIdArray:
            track = dataframe[dataframe[track_id_col_name] == trackId]
            p_intensity = track[intensity_to_plot[0]].values.astype(float) #primary (channel 3 in our case)
            s_intensity = track[intensity_to_plot[1]].values.astype(float) #secondary  (channel 2 in our case)
            t_intensity = track[intensity_to_plot[2]].values.astype(float) #tertiary (channel 1 in our case)

            # align by peak of secondary channel
            maxIdx = np.argmax(s_intensity)

            # align by peak of tertiary channel
            # maxIdx = np.argmax(t_intensity)

            p_maxIntensity = np.nanmax(p_intensity)
            s_maxIntensity = np.nanmax(s_intensity)
            t_maxIntensity = np.nanmax(t_intensity)

            
        
            for i in range(0,len(track)):
                if(not np.isnan(p_intensity[i])):
                    p_buffer[counter][bufferZero-maxIdx+i]=(p_intensity[i]) / p_maxIntensity
                if(not np.isnan(s_intensity[i])):
                    s_buffer[counter][bufferZero-maxIdx+i]=(s_intensity[i]) / s_maxIntensity
                if(not np.isnan(t_intensity[i])):
                    t_buffer[counter][bufferZero-maxIdx+i]=(t_intensity[i]) / t_maxIntensity

            
                    
            counter = counter+1;

    
    
        return p_buffer,s_buffer, t_buffer
