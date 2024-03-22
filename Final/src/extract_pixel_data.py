from os import path
import pandas as pd
from skimage import io
import numpy as np 
from typing import Optional, Union

#Fixed radius version
def extract_pixels_data_fixed(target_image: np.ndarray, frame_centers: list, radii: list, offset: list = [0,0]):
    '''
    The function above uses a volume of pixels from one channel and extracts data for the same pixels from the 
    other channel. It uses a fixed radii and ignores all the zero pixel values which exist while doing calculations. 

    Inputs: 
    1. target_image: type(array), this is a 4-D array with dimensions as (t,z,y,x). The target image is the channel for which 
    we want to extract the data for not the one for which we have already have the values for
    2. frame_centers: type(list), this is a list of list. Each time frame has its z,y,x coords stored in it. It has the center's
    of the entire track. 
    3. radii: type(list), the radius of a spot in each dimension. Passed in as [z,y,x]
    4. offset: type(list), this is an optional argument, it can be used to adjust for offset if there is any between 
    two channels. Must be in the form [y,x]

    Outputs: 
    1. mean: type(list), returns the mean pixel value for the volume for each time frame of a specific track 
    2. maximum: type(list), returns the max pixel value for the volume for each time frame of a specific track
    3. minimum: type(list), returns the min pixel value for the volume for each time frame of a specific track
    4. pixel_values: type(list), this is a list of arrays, returns all the non zero pixel values for the selected volume
    5. max_loc: type(list), this is a list of arrays, returns the coordinates of the pixel with maximum value in the volume 
    6. voxel_sum_array: type(list), returns the pixel values sum of the selected volume(does not include any zero values)
    '''


    mean = []
    maximum = []
    minimum = []
    pixel_values = []
    max_loc = []
    voxel_sum_array = []
    max_z, max_y, max_x = target_image[0].shape  # Assuming image is a 3D numpy array
    radius_z = radii[0]
    radius_y = radii[1]
    radius_x = radii[2]

    for coords in frame_centers:
        #print('current coordinates are: ', coords)
        frame = int(coords[0])
        z = coords[1]
        y = max(0,coords[2] - offset[0])
        x = max(0,coords[3] - offset[1])
        

        # Ensure lower bounds
        z_start = int(max(0, z - radius_z))
        y_start = int(max(0, y - radius_y))
        x_start = int(max(0, x - radius_x))

        # Ensure upper bounds
        z_end = int(min(max_z, z + radius_z + 1))
        y_end = int(min(max_y, y + radius_y + 1))
        x_end = int(min(max_x, x + radius_x + 1))

        # Extract relevant pixels
        extracted_pixels = target_image[frame,z_start:z_end, y_start:y_end, x_start:x_end]
        #print('shape of extracted pixels is ', extracted_pixels.shape)
        
        # Exclude pixels with value 0 before calculating mean
        non_zero_pixels = extracted_pixels[extracted_pixels != 0]
        
        if non_zero_pixels.size > 0:
            # Calculate statistics
            mean_value = np.mean(non_zero_pixels)
            max_value = np.max(non_zero_pixels)
            voxel_sum = np.sum(non_zero_pixels)
            
            # Get coordinates of the maximum value
            max_index = np.unravel_index(np.argmax(extracted_pixels), extracted_pixels.shape)
            max_coords = (z_start + max_index[0], y_start + max_index[1], x_start + max_index[2])
            min_value = np.min(non_zero_pixels)
            
            mean.append(mean_value)
            maximum.append(max_value)
            max_loc.append(max_coords)
            minimum.append(min_value)
            pixel_values.append(extracted_pixels)
            voxel_sum_array.append(voxel_sum)
        else:
            # If all pixels are 0, handle this case as needed
            mean.append(np.nan)  # Use NaN or any other suitable value
            maximum.append(0)
            minimum.append(0)
            pixel_values.append(extracted_pixels)
            print('zero pixels here')
        
    return mean,maximum,minimum,pixel_values,max_loc,voxel_sum_array


#Variable radii(sigma values) version
def extract_pixels_data_variable(raw_image: np.ndarray, mean_col_names: list, dataframe: pd.DataFrame
                                 ,radi_col_names: list, frame_col_name: str, offset: list = [0,0]):
    '''
    The function above uses a volume of pixels from one channel and extracts data for the same pixels from the 
    other channel. It uses a variable radii and ignores all the zero pixel values which exist while doing calculations. 

    **Inputs**: 
    1. raw_image: type(array), this is a 4-D array with dimensions as (t,z,y,x). The target image is the channel for which 
    we want to extract the data for not the one for which we have already have the values for
    2. mean_col_names: type(list), this is a list of strings. This contains the names of the columns which contain the center
    coordinates. It must be passed in the form [z,y,x]
    3. dataframe: type(dataframe), this is the main dataframe returned from tracking of our primary channel  and contains 
    the frame number along with radi/sigma values so that they can be used for constructing the volume 
    4. radi_col_names: type(list), this is a list of strings. This contains the name of the columns which will act as a 
    radius to construct volume around the coordinates. It must be passed in the form [sigma_z,sigma_y,sigma_x]
    5. frame_col_name: type(str), the column name which contains the frame number
    6. offset: type(list), this is an optional argument, it can be used to adjust for offset if there is any between 
    two channels. Must be in the form [y,x]

    **Outputs**: 
    1. mean: type(list), returns the mean pixel value for the volume for each time frame of a specific track 
    2. maximum: type(list), returns the max pixel value for the volume for each time frame of a specific track
    3. minimum: type(list), returns the min pixel value for the volume for each time frame of a specific track
    4. pixel_values: type(list), this is a list of arrays, returns all the non zero pixel values for the selected volume
    5. max_loc: type(list), this is a list of arrays, returns the coordinates of the pixel with maximum value in the volume 
    '''

    if not isinstance(raw_image, np.ndarray):
        raise TypeError("Input must be a NumPy array")
    if raw_image.ndim != 4:
        raise ValueError("Input array must be 4-D")
    
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")
    
    mean = []
    maximum = []
    minimum = []
    pixel_values = []
    max_loc = []
    max_z, max_y, max_x = raw_image[0].shape  # Assuming image is a 3D numpy array

    for i in range(len(dataframe)):
        #print('current coordinates are: ', coords)
        frame = int(dataframe.loc[i,frame_col_name])
        z = dataframe.loc[i, mean_col_names[0]]
        y = max(0,dataframe.loc[i, mean_col_names[1]] - offset[0])
        x = max(0,dataframe.loc[i, mean_col_names[2]] - offset[1])
        radius_z = dataframe.loc[i,radi_col_names[0]]
        radius_y = dataframe.loc[i,radi_col_names[1]]
        radius_x = dataframe.loc[i,radi_col_names[2]]
        

        # Ensure lower bounds
        z_start = int(max(0, z - radius_z))
        y_start = int(max(0, y - radius_y))
        x_start = int(max(0, x - radius_x))

        # Ensure upper bounds
        z_end = int(min(max_z, z + radius_z + 1))
        y_end = int(min(max_y, y + radius_y + 1))
        x_end = int(min(max_x, x + radius_x + 1))

        # Extract relevant pixels
        extracted_pixels = raw_image[frame,z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Exclude pixels with value 0 before calculating mean
        non_zero_pixels = extracted_pixels[extracted_pixels != 0]
        
        if non_zero_pixels.size > 0:
            # Calculate statistics
            mean_value = np.mean(non_zero_pixels)
            max_value = np.max(non_zero_pixels)
            voxel_sum = np.sum(non_zero_pixels)
            
            # Get coordinates of the maximum value
            max_index = np.unravel_index(np.argmax(extracted_pixels), extracted_pixels.shape)
            max_coords = (z_start + max_index[0], y_start + max_index[1], x_start + max_index[2])
            min_value = np.min(non_zero_pixels)
            
            mean.append(mean_value)
            maximum.append(max_value)
            max_loc.append(max_coords)
            minimum.append(min_value)
            pixel_values.append(extracted_pixels)
        else:
            # If all pixels are 0, handle this case as needed
            mean.append(np.nan)  # Use NaN or any other suitable value
            maximum.append(np.nan)
            minimum.append(np.nan)
            temp = (np.nan, np.nan, np.nan)
            max_loc.append(temp)
            pixel_values.append(extracted_pixels)
        
    return mean,maximum,minimum,pixel_values,max_loc


#Variable radi(sigma) variant
def voxel_sum_variable(dataframe: pd.DataFrame, mean_col_names: list, raw_image: np.ndarray, radi_col_names: list, 
                       frame_col_name: str):
    '''
    This function calculates the voxel sum for a volume constructed using center coords and variable sigma values. It ignores 
    all pixels which have value zero

    Inputs: 
    1. dataframe: type(dataframe), this is the main tracking dataframe which contains the center coords and sigma/radi values 
    2. mean_col_names: type(list), this is a list of strings. This contains the names of the columns which contain the center
    coordinates. It must be passed in the form [z,y,x]
    3. raw_image: type(array), this is a 4-D array with dimensions as (t,z,y,x). The image should be the one for which 
    coordinates are being used. 
    4. radi_col_names: type(list), this is a list of strings. This contains the name of the columns which will act as a 
    radius to construct volume around the coordinates. It must be passed in the form [sigma_z,sigma_y,sigma_x]
    5. frame_col_name: type(str), the column name which contains the frame number

    Outputs: 
    1. voxel_sum_array: type(list), returns the voxel sum
    2. pixel_vales: type(list), list of array, returns all the values for the pixels used for voxel sum 
    '''
    max_z, max_y, max_x = raw_image[0].shape  # Assuming image is a 3D numpy array
    voxel_sum_array = []
    pixel_values = []
    
    for i in range(len(dataframe)):
        frame = int(dataframe.loc[i,frame_col_name])
        z = dataframe.loc[i,mean_col_names[0]]
        y = dataframe.loc[i,mean_col_names[1]]
        x = dataframe.loc[i,mean_col_names[2]]
        radius_z = dataframe.loc[i,radi_col_names[0]]
        radius_y = dataframe.loc[i,radi_col_names[1]]
        radius_x = dataframe.loc[i,radi_col_names[2]]
        

        # Ensure lower bounds
        z_start = int(max(0, z - radius_z))
        y_start = int(max(0, y - radius_y))
        x_start = int(max(0, x - radius_x))

        # Ensure upper bounds
        z_end = int(min(max_z, z + radius_z + 1))
        y_end = int(min(max_y, y + radius_y + 1))
        x_end = int(min(max_x, x + radius_x + 1))

        # Extract relevant pixels
        extracted_pixels = raw_image[frame,z_start:z_end, y_start:y_end, x_start:x_end]
        #print('shape of extracted pixels is ', extracted_pixels.shape)
        
        # Exclude pixels with value 0 before calculating mean
        non_zero_pixels = extracted_pixels[extracted_pixels != 0]
        
        if non_zero_pixels.size > 0:
            # Calculate statistics
            voxel_sum = np.sum(non_zero_pixels)
            
            # Get coordinates of the maximum value
            voxel_sum_array.append(voxel_sum)
            pixel_values.append(non_zero_pixels)
        else:
            # If all pixels are 0, handle this case as needed
            voxel_sum_array.append(np.nan)  # Use NaN or any other suitable value
        
    return voxel_sum_array,pixel_values

#Fixed radi/sigma variant
def voxel_sum_fixed(dataframe: pd.DataFrame ,col_names: list, raw_image: np.ndarray, radii: list, frame_col_name: str):
    '''
    This function calculates the voxel sum for a volume constructed using center coords and fixed sigma/radii values. It ignores 
    all pixels which have value zero

    Inputs: 
    1. dataframe: type(dataframe), this is the main tracking dataframe which contains the center coords and sigma/radi values 
    2. col_names: type(list), this is a list of strings. This contains the names of the columns which contain the center
    coordinates. It must be passed in the form [z,y,x]
    3. raw_image: type(array), this is a 4-D array with dimensions as (t,z,y,x). The image should be the one for which 
    coordinates are being used. 
    4. radii: type(list), the radius of a spot in each dimension. Passed in as [z,y,x]
    5. frame_col_name: type(str), the column name which contains the frame number

    Outputs: 
    1. voxel_sum_array: type(list), returns the voxel sum
    2. pixel_vales: type(list), list of array, returns all the values for the pixels used for voxel sum 
    '''

    max_z, max_y, max_x = raw_image[0].shape  # Assuming image is a 3D numpy array
    voxel_sum_array = []
    pixel_values = []
    radius_z = radii[0]
    radius_y = radii[1]
    radius_x = radii[2]
    
    for i in range(len(dataframe)):
        frame = int(dataframe.loc[i,frame_col_name])
        z = dataframe.loc[i,col_names[0]]
        y = dataframe.loc[i,col_names[1]]
        x = dataframe.loc[i,col_names[2]]
        

        # Ensure lower bounds
        z_start = int(max(0, z - radius_z))
        y_start = int(max(0, y - radius_y))
        x_start = int(max(0, x - radius_x))

        # Ensure upper bounds
        z_end = int(min(max_z, z + radius_z + 1))
        y_end = int(min(max_y, y + radius_y + 1))
        x_end = int(min(max_x, x + radius_x + 1))

        # Extract relevant pixels
        extracted_pixels = raw_image[frame,z_start:z_end, y_start:y_end, x_start:x_end]
        #print('shape of extracted pixels is ', extracted_pixels.shape)
        
        # Exclude pixels with value 0 before calculating mean
        non_zero_pixels = extracted_pixels[extracted_pixels != 0]
        
        if non_zero_pixels.size > 0:
            # Calculate statistics
            voxel_sum = np.sum(non_zero_pixels)
            
            # Get coordinates of the maximum value
            voxel_sum_array.append(voxel_sum)
            pixel_values.append(non_zero_pixels)
        else:
            # If all pixels are 0, handle this case as needed
            voxel_sum_array.append(np.nan)  # Use NaN or any other suitable value
        
    return voxel_sum_array,pixel_values


#Variable radii(sigma values) version for handling bigger files which do not fit in memory 
def extract_pixels_data_variable_bd(raw_image, mean_col_names: list, dataframe: pd.DataFrame
                                 ,radi_col_names: list, frame_col_name: str, offset: list = [0,0]):
    '''
    The function above uses a volume of pixels from one channel and extracts data for the same pixels from the 
    other channel. It uses a variable radii and ignores all the zero pixel values which exist while doing calculations. 

    **Inputs**: 
    1. raw_image: this raw image is the img object from AICSImageio tool. Its not imported into memory and will use dask_arrays to work
    for a single time frame at a time
    2. mean_col_names: type(list), this is a list of strings. This contains the names of the columns which contain the center
    coordinates. It must be passed in the form [z,y,x]
    3. dataframe: type(dataframe), this is the main dataframe returned from tracking of our primary channel  and contains 
    the frame number along with radi/sigma values so that they can be used for constructing the volume 
    4. radi_col_names: type(list), this is a list of strings. This contains the name of the columns which will act as a 
    radius to construct volume around the coordinates. It must be passed in the form [sigma_z,sigma_y,sigma_x]
    5. frame_col_name: type(str), the column name which contains the frame number
    6. offset: type(list), this is an optional argument, it can be used to adjust for offset if there is any between 
    two channels. Must be in the form [y,x]

    **Outputs**: 
    1. mean: type(list), returns the mean pixel value for the volume for each time frame of a specific track 
    2. maximum: type(list), returns the max pixel value for the volume for each time frame of a specific track
    3. minimum: type(list), returns the min pixel value for the volume for each time frame of a specific track
    4. pixel_values: type(list), this is a list of arrays, returns all the non zero pixel values for the selected volume
    5. max_loc: type(list), this is a list of arrays, returns the coordinates of the pixel with maximum value in the volume 
    '''

    #if not isinstance(raw_image, np.ndarray):
        #raise TypeError("Input must be a NumPy array")
    #if raw_image.ndim != 4:
        #raise ValueError("Input array must be 4-D")
    
    #if not isinstance(dataframe, pd.DataFrame):
        #raise TypeError("Input must be a DataFrame")
    
    mean = []
    maximum = []
    minimum = []
    pixel_values = []
    max_loc = []
    max_z = raw_image.dims.Z
    max_y = raw_image.dims.Y
    max_x = raw_image.dims.X
    frames = raw_image.dims.T
    
    for frame in range(frames):
        print(f'current frame number is {frame}')
        current_df = dataframe[dataframe['frame'] == frame].reset_index()
        lazy_single_frame_input = raw_image.get_image_dask_data("ZYX", T=frame, C=0)
        current_image = lazy_single_frame_input.compute()
        
        for i in range(len(current_df)):
            #print('current coordinates are: ', coords)
            z = current_df.loc[i, mean_col_names[0]]
            y = max(0,current_df.loc[i, mean_col_names[1]] - offset[0])
            x = max(0,current_df.loc[i, mean_col_names[2]] - offset[1])
            radius_z = current_df.loc[i,radi_col_names[0]]
            radius_y = current_df.loc[i,radi_col_names[1]]
            radius_x = current_df.loc[i,radi_col_names[2]]


            # Ensure lower bounds
            z_start = int(max(0, z - radius_z))
            y_start = int(max(0, y - radius_y))
            x_start = int(max(0, x - radius_x))

            # Ensure upper bounds
            z_end = int(min(max_z, z + radius_z + 1))
            y_end = int(min(max_y, y + radius_y + 1))
            x_end = int(min(max_x, x + radius_x + 1))

            # Extract relevant pixels
            extracted_pixels = current_image[z_start:z_end, y_start:y_end, x_start:x_end]

            # Exclude pixels with value 0 before calculating mean
            non_zero_pixels = extracted_pixels[extracted_pixels != 0]

            if non_zero_pixels.size > 0:
                # Calculate statistics
                mean_value = np.mean(non_zero_pixels)
                max_value = np.max(non_zero_pixels)
                voxel_sum = np.sum(non_zero_pixels)

                # Get coordinates of the maximum value
                max_index = np.unravel_index(np.argmax(extracted_pixels), extracted_pixels.shape)
                max_coords = (z_start + max_index[0], y_start + max_index[1], x_start + max_index[2])
                min_value = np.min(non_zero_pixels)

                mean.append(mean_value)
                maximum.append(max_value)
                max_loc.append(max_coords)
                minimum.append(min_value)
                pixel_values.append(extracted_pixels)
            else:
                # If all pixels are 0, handle this case as needed
                mean.append(np.nan)  # Use NaN or any other suitable value
                maximum.append(np.nan)
                minimum.append(np.nan)
                temp = (np.nan, np.nan, np.nan)
                max_loc.append(temp)
                pixel_values.append(extracted_pixels)

    return mean,maximum,minimum,pixel_values,max_loc

#Fixed radi/sigma variant for large files which do not fit in memory 
def voxel_sum_fixed_bd(dataframe: pd.DataFrame ,col_names: list, raw_image, 
                    radii: list, frame_col_name: str):
    '''
    This function calculates the voxel sum for a volume constructed using center coords and fixed sigma/radii values. It ignores 
    all pixels which have value zero

    Inputs: 
    1. dataframe: type(dataframe), this is the main tracking dataframe which contains the center coords and sigma/radi values 
    2. col_names: type(list), this is a list of strings. This contains the names of the columns which contain the center
    coordinates. It must be passed in the form [z,y,x]
    3. raw_image: this raw image is the img object from AICSImageio tool. Its not imported into memory and will use dask_arrays to work
    for a single time frame at a time
    4. radii: type(list), the radius of a spot in each dimension. Passed in as [z,y,x]
    5. frame_col_name: type(str), the column name which contains the frame number

    Outputs: 
    1. voxel_sum_array: type(list), returns the voxel sum
    2. pixel_vales: type(list), list of array, returns all the values for the pixels used for voxel sum 
    '''

    max_z = raw_image.dims.Z
    max_y = raw_image.dims.Y
    max_x = raw_image.dims.X
    frames = raw_image.dims.T
    voxel_sum_array = []
    pixel_values = []
    radius_z = radii[0]
    radius_y = radii[1]
    radius_x = radii[2]
    
    for frame in range(frames): 
        print(f'current frame is {frame}')
        current_df = dataframe[dataframe[frame_col_name] == frame].reset_index()
        lazy_single_frame_input = raw_image.get_image_dask_data("ZYX", T=frame, C=0)
        current_image = lazy_single_frame_input.compute()
        
        for i in range(len(current_df)):
            z = dataframe.loc[i,col_names[0]]
            y = dataframe.loc[i,col_names[1]]
            x = dataframe.loc[i,col_names[2]]


            # Ensure lower bounds
            z_start = int(max(0, z - radius_z))
            y_start = int(max(0, y - radius_y))
            x_start = int(max(0, x - radius_x))

            # Ensure upper bounds
            z_end = int(min(max_z, z + radius_z + 1))
            y_end = int(min(max_y, y + radius_y + 1))
            x_end = int(min(max_x, x + radius_x + 1))

            # Extract relevant pixels
            extracted_pixels = current_image[z_start:z_end, y_start:y_end, x_start:x_end]
            #print('shape of extracted pixels is ', extracted_pixels.shape)

            # Exclude pixels with value 0 before calculating mean
            non_zero_pixels = extracted_pixels[extracted_pixels != 0]

            if non_zero_pixels.size > 0:
                # Calculate statistics
                voxel_sum = np.sum(non_zero_pixels)

                # Get coordinates of the maximum value
                voxel_sum_array.append(voxel_sum)
                pixel_values.append(non_zero_pixels)
            else:
                # If all pixels are 0, handle this case as needed
                voxel_sum_array.append(np.nan)  # Use NaN or any other suitable value
        
    return voxel_sum_array,pixel_values


# Helper function
# Function to check if coordinates are within the range
def check_range(row):
    x_range = (row['mu_x'] - 2 * row['sigma_x'], row['mu_x'] + 2 * row['sigma_x'])
    y_range = (row['mu_y'] - 2 * row['sigma_y'], row['mu_y'] + 2 * row['sigma_y'])
    z_range = (row['mu_z'] - 2 * row['sigma_z'], row['mu_z'] + 2 * row['sigma_z'])

    return (
        x_range[0] <= row['c2_peak_x'] <= x_range[1] and
        y_range[0] <= row['c2_peak_y'] <= y_range[1] and
        z_range[0] <= row['c2_peak_z'] <= z_range[1]
    )
