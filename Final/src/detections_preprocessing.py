import numpy as np
import pandas as pd 
import napari
import os


def combine_dataframes(number_of_files: int ,channel: int, directory_path: str):
    '''
    The function takes in all pickle files of the dataframes returned from detection for each time frame 
    and combines them into a single large dataframe. The function is provided with a specific folder path where it
    goes and extracts all the pickle files. 
    
    Note: 
    The naming convention of the files is fixed to a specific naming convention. For example for channel 2 frame 0
    the name of the file should be df_c2_t0. This must be followed for this function to work properly. 
    
    Inputs: 
    1. number_of_files: type(int), This is equal to the number of time frames/files in the specified folder 
    2. file_path: type(int), the path of the folder where all the files for a specific channel is stored
    3. channel: type(str), The channel number (e.g 1,2,3)
    
    Returns: 
    1. combined_df: type(Dataframe), this is a combined dataframe for all detections for a single channel 
    '''
    
    # Create an empty list to store the individual DataFrames
    dataframes = []

    # Load pickle files, add "frame" column, and store DataFrames
    for i in range(number_of_files):
        file_name = f"df_c{channel}_t{i}.pkl"
        file_path = os.path.join(directory_path, file_name)

        # Load the DataFrame from the pickle file
        df = pd.read_pickle(file_path)

        # Extract the frame number from the file name
        frame = int(file_name.split("_t")[1].split(".")[0])

        # Add the "frame" column to the DataFrame
        df["frame"] = frame

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate the list of DataFrames into a single larger DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df
    

