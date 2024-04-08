from os import path
import pandas as pd
from skimage import io
import numpy as np 
from typing import Optional, Union
import zarr
from gaussian_fitting import fit_multiple_gaussians
from gaussian_fitting import check_fitting_error
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os


class Extractor:
    
    def __init__(self, zarr_obj: zarr.array, dataframe: pd.DataFrame, radii: list, frame_col_name: str, radi_col_name: list, 
                n_jobs: int):
        self.zarr_obj = zarr_obj
        self.channels = zarr_obj.shape[1]
        self.frames = zarr_obj.shape[0]
        self.z = zarr_obj.shape[2]
        self.y = zarr_obj.shape[3]
        self.x = zarr_obj.shape[4]
        self.dataframe = dataframe
        self.radii = radii
        self.frame_col_name = frame_col_name
        self.parallel_process = n_jobs
        self.radi_col_names = radi_col_name

        
    

    #Fixed radi/sigma variant for large files which do not fit in memory 
    def voxel_sum_fixed_bd(self,center_col_names: list,  channel: int):

        current_channel = channel - 1 
        frames = self.dataframe[self.frame_col_name].nunique()
        max_z = self.z
        max_y = self.y
        max_x = self.x
        voxel_sum_array = []
        pixel_values = []

        radius_z = self.radii[0]
        radius_y = self.radii[1]
        radius_x = self.radii[2]
        
        for frame in range(frames): 
            print(f'current frame is {frame}')
            current_df = self.dataframe[self.dataframe[self.frame_col_name] == frame].reset_index()
            current_image = self.zarr_obj[frame,current_channel,:,:,:]
        
            
            for i in range(len(current_df)):

                z = current_df.loc[i,center_col_names[0]]
                y = current_df.loc[i,center_col_names[1]]
                x = current_df.loc[i,center_col_names[2]]


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

    #Variable radi/sigma variant for large files which do not fit in memory 
    def voxel_sum_variable_bd(self ,center_col_names: list, channel: int):

        current_channel = channel - 1 
        frames = self.dataframe[self.frame_col_name].nunique()
        max_z = self.z
        max_y = self.y
        max_x = self.x
        voxel_sum_array = []
        pixel_values = []
        
        for frame in range(frames): 
            print(f'current frame is {frame}')
            current_df = self.dataframe[self.dataframe[self.frame_col_name] == frame].reset_index()
            current_image = self.zarr_obj[frame,current_channel,:,:,:]
        
            
            for i in range(len(current_df)):
                radius_z = current_df.loc[i,self.radi_col_names[0]]
                radius_y = current_df.loc[i,self.radi_col_names[1]]
                radius_x = current_df.loc[i,self.radi_col_names[2]]
                z = current_df.loc[i,center_col_names[0]]
                y = current_df.loc[i,center_col_names[1]]
                x = current_df.loc[i,center_col_names[2]]


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

    #Variable radii(sigma values) version for handling bigger files which do not fit in memory 
    def extract_pixels_data_variable_bd(self, center_col_names: list, channel: int, offset: list = [0,0]):
        
        current_channel = channel - 1
        mean = []
        maximum = []
        minimum = []
        pixel_values = []
        max_loc = []
        frames = self.dataframe[self.frame_col_name].nunique()
        max_z = self.z
        max_y = self.y
        max_x = self.x
        
        
        for frame in range(frames):
            print(f'current frame number is {frame}')
            current_df = self.dataframe[self.dataframe['frame'] == frame].reset_index()
            current_image = self.zarr_obj[frame,current_channel,:,:,:]
            
            for i in range(len(current_df)):
                #print('current coordinates are: ', coords)
                z = current_df.loc[i, center_col_names[0]]
                y = max(0,current_df.loc[i, center_col_names[1]] - offset[0])
                x = max(0,current_df.loc[i, center_col_names[2]] - offset[1])
                radius_z = current_df.loc[i,self.radi_col_names[0]]
                radius_y = current_df.loc[i,self.radi_col_names[1]]
                radius_x = current_df.loc[i,self.radi_col_names[2]]


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
    
    #Fixed radii(sigma values) version for handling bigger files which do not fit in memory 
    def extract_pixels_data_fixed_bd(self, center_col_names: list, channel: int, offset: list = [0,0]):
        
        current_channel = channel - 1
        mean = []
        maximum = []
        minimum = []
        pixel_values = []
        max_loc = []
        frames = self.dataframe[self.frame_col_name].nunique()
        max_z = self.z
        max_y = self.y
        max_x = self.x
        radius_z = self.radii[0]
        radius_y = self.radii[1]
        radius_x = self.radii[2]
        
        
        for frame in range(frames):
            print(f'current frame number is {frame}')
            current_df = self.dataframe[self.dataframe['frame'] == frame].reset_index()
            current_image = self.zarr_obj[frame,current_channel,:,:,:]
            
            for i in range(len(current_df)):
                #print('current coordinates are: ', coords)
                z = current_df.loc[i, center_col_names[0]]
                y = max(0,current_df.loc[i, center_col_names[1]] - offset[0])
                x = max(0,current_df.loc[i, center_col_names[2]] - offset[1])


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

    def gaussian_fitting_single_frame(self, expected_sigma: list, center_col_names: list, frame: int, channel: int, dist_between_spots: int):
        
        current_channel = channel - 1
        sigmaExpected_x__pixels = expected_sigma[2]
        sigmaExpected_y__pixels = expected_sigma[1]
        sigmaExpected_z__pixels = expected_sigma[0]

        center_col_names_adjusted = [center_col_names[0], center_col_names[2], center_col_names[1]]

        sigmas_guesses = []
        current_df = self.dataframe[self.dataframe[self.frame_col_name]==frame]
        maximas = current_df[center_col_names_adjusted].values
        single_frame_input = self.zarr_obj[frame,current_channel,:,:,:]
        single_frame_input = np.transpose(single_frame_input,axes =(0,2,1))


        for i in range(len(maximas)):
            sigmas_guesses.append([sigmaExpected_z__pixels,sigmaExpected_x__pixels,sigmaExpected_y__pixels])
            
        #last parameter in the fit_multiple_gaussians is similar to min_distance above, we should give half of the 
        #value here of min_distance   
        half_dist = dist_between_spots / 2
        gaussians, gaussians_popt = fit_multiple_gaussians(single_frame_input,maximas,sigmas_guesses,half_dist)
            
        accumulator = []
        for gaussian in gaussians:

            if(gaussian!=-1):
                amplitude = gaussian[0]
                mu_x     = int(gaussian[1][1]) 
                mu_y     = int(gaussian[1][2]) 
                mu_z     = int(gaussian[1][0]) 
                sigma_x  = int(gaussian[2][1]) 
                sigma_y  = int(gaussian[2][2])
                sigma_z  = int(gaussian[2][0])
                accumulator.append(np.array([amplitude,mu_x,mu_y,mu_z,sigma_x,sigma_y,sigma_z]))
            else: 

                amplitude = np.nan
                mu_x     = np.nan
                mu_y     = np.nan
                mu_z     = np.nan
                sigma_x  = np.nan
                sigma_y  = np.nan
                sigma_z  = np.nan
                accumulator.append(np.array([amplitude,mu_x,mu_y,mu_z,sigma_x,sigma_y,sigma_z]))

                
        accumulator = np.array(accumulator)
        print(accumulator.shape)
        df = pd.DataFrame()
        df['amplitude'] = accumulator[:,0]
        df['mu_x'] = accumulator[:,1]
        df['mu_y'] = accumulator[:,2]
        df['mu_z'] = accumulator[:,3]
        df['sigma_x'] = accumulator[:,4]
        df['sigma_y'] = accumulator[:,5]
        df['sigma_z'] = accumulator[:,6]
        
        error_list, index_list = check_fitting_error(single_frame_input,maximas,gaussians,sigmas_guesses)

        return df

    def cores_to_use(self):
        """
        Determines the optimal number of cores to use for parallel processing.
        
        Returns:
            int: The number of cores to use. Raises a ValueError if the specified number of cores exceeds available cores.
        """

        if self.parallel_process == -1:
            return os.cpu_count() - 1
        elif self.parallel_process > os.cpu_count():
            raise ValueError(f"Error: You specified {self.parallel_process} cores, but only {os.cpu_count()} cores are available.")
        else: 
            return self.parallel_process
            
    def run_parallel_frame_processing(self, expected_sigma: list, center_col_name: list,
                                    dist_between_spots: int, channel: int,  max_frames: int = 2, all_frames: bool = False):
        """
        Processes multiple frames in parallel using the single_frame_segmentation method and returns the combined results.
        
        Parameters:
            max_frames (int, optional): The maximum number of frames to process. Defaults to 2.
            all_frames (bool, optional): If true all the frames will be processed regardless of max_frames
            
        Returns:
            DataFrame: A pandas DataFrame containing the combined analysis results from all processed frames.
        """

        if all_frames == True: 
            frames_to_process = self.dataframe[self.frame_col_name].nunique()
        else:
            frames_to_process = max_frames

        num_of_parallel_process = self.cores_to_use()
        futures_to_frame = {}

        # Initialize a ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers = num_of_parallel_process) as executor:
        
            # Submit tasks for each frame to be processed in parallel
            for frame in range(frames_to_process):
                future = executor.submit(self.gaussian_fitting_single_frame,  expected_sigma, center_col_name, frame, channel, dist_between_spots)
                futures_to_frame[future] = frame  # Map future to frame number

            
            frame_results = []
            # Use tqdm to show progress as tasks complete
            for future in tqdm(as_completed(futures_to_frame), total=frames_to_process, desc="Processing frames"):
                frame = futures_to_frame[future]
                try:
                    # If you need the result for anything, or to catch exceptions:
                    result = future.result()
                    if result is not None: 
                        # Append a tuple of (frame, result) to frame_results
                        frame_results.append((frame, result))
                except Exception as e:
                    # Handle exceptions (if any) from your processed function
                    print(f"Error processing frame: {e}")
            
            # Initialize an empty DataFrame
            final_df = pd.DataFrame()

            for frame, result_df in sorted(frame_results):
                result_df['frame'] = frame  # Add a column with the frame number
                final_df = pd.concat([final_df, result_df], ignore_index=True)
            
            # Construct the filename based on the loop index (time_frame)
            #filename_pkl = f'all_detections_channel{self.target_channel + 1}.pkl'

            # Construct the full file path by joining the directory and filename
            #file_path = os.path.join(self.save_directory, filename_pkl)

            #final_df.to_pickle(file_path)
            
            # Return the combined dataframe instead of saving
            return final_df