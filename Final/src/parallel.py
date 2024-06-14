from peak_local_max_3d import peak_local_max_3d 
from gaussian_fitting import fit_multiple_gaussians
from extract_data import extract_data_from_filename
from gaussian_visualization import visualize_3D_gaussians
from gaussian_fitting import check_fitting_error
import pandas as pd
import os 
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import zarr

class Detector:
    """
    A class designed to detect and analyze spots in image frames from a zarr file using parallel processing.
    
    Attributes:
        zarr_obj (zarr.array): The zarr array object containing image data.
        save_directory (str): Directory path where results are saved.
        min_intensity (float): Minimum intensity threshold for spot detection.
        dist_between_spots (int): Minimum distance between detected spots.
        sigma_z (float): Estimated standard deviation of spots in the z dimension.
        sigma_y (float): Estimated standard deviation of spots in the y dimension.
        sigma_x (float): Estimated standard deviation of spots in the x dimension.
        frames (int): Total number of frames in the zarr object.
        channels (int): Number of channels in the image data.
        z (int): Depth of the image data.
        y (int): Height of the image data.
        x (int): Width of the image data.
        parallel_process (int): Number of parallel processes to use for processing.
        target_channel(int): Number of the channel to perfrom detection on

    Methods:
        single_frame_segmentation(frame):
            Processes a single frame to detect and analyze spots.
        
        cores_to_use():
            Determines the number of cores to use for parallel processing based on user input and system capabilities.
        
        run_parallel_frame_processing(max_frames=2):
            Runs spot detection and analysis on multiple frames in parallel and returns the results in a combined DataFrame.
    """

    def __init__(self, zarr_obj: zarr.array, save_directory: str, spot_intensity: float, dist_between_spots: int, sigma_estimations: list, n_jobs: int,
                 channel_to_detect: int):
        """
        Initializes the Detector with the specified parameters.
        
        Parameters:
            zarr_obj (zarr.array): The zarr array object containing image data.
            save_directory (str): Directory path where results are to be saved.
            spot_intensity (float): Minimum intensity threshold for spot detection.
            dist_between_spots (int): Minimum distance between detected spots.
            sigma_estimations (list of float): List containing estimated standard deviations of spots in z, y, and x dimensions.
            n_jobs (int): Number of parallel processes to use. If -1, uses all available cores minus one.
            channel_to_detect (int): Determines which channel to perform detection on. 
            since the detector can take input of multi channel movie stored as zarr so determing which channel to perform detection on is important. 
        """

        self.zarr_obj = zarr_obj
        self.save_directory = save_directory
        self.min_intensity = spot_intensity
        self.dist_between_spots = dist_between_spots
        self.sigma_z = sigma_estimations[0]
        self.sigma_y = sigma_estimations[1]
        self.sigma_x = sigma_estimations[2]
        self.frames = zarr_obj.shape[0]
        self.channels = zarr_obj.shape[1]
        self.z = zarr_obj.shape[2]
        self.y = zarr_obj.shape[3]
        self.x = zarr_obj.shape[4]
        self.parallel_process = n_jobs
        self.target_channel = channel_to_detect - 1
    

    def single_frame_segmentation(self,frame: int):
        """
        Processes a single frame to detect and analyze spots using specified detection parameters.
        
        Parameters:
            frame (int): The index of the frame to process.
            
        Returns:
            DataFrame: A pandas DataFrame containing the analysis results for the frame.
        """
        
        
        #get the number of frames for our original data for automated analysis for all frames         
        single_frame_input = self.zarr_obj[frame,self.target_channel,:,:,:]
        single_frame_input = np.transpose(single_frame_input,axes =(0,2,1))

        #define threshold: a value(intensity) for the pixel below which all values would be considered noise and dropped 
        #define min_distance: min_distance/2 is the radius within which we will keep the peak with max value/intensity or 
        #if two peaks have the same value they will be kept 
        
        maximas = peak_local_max_3d(single_frame_input,min_distance=self.dist_between_spots,threshold=self.min_intensity)


        #give the expected std dev/radius of our particles for x,y,z 
        sigmaExpected_x__pixels = self.sigma_x
        sigmaExpected_y__pixels = self.sigma_y
        sigmaExpected_z__pixels = self.sigma_z

        sigmas_guesses = []
        for i in range(len(maximas)):
            sigmas_guesses.append([sigmaExpected_z__pixels,sigmaExpected_x__pixels,sigmaExpected_y__pixels])
            
        #last parameter in the fit_multiple_gaussians is similar to min_distance above, we should give half of the 
        #value here of min_distance   
        half_dist = self.dist_between_spots / 2
        # gaussians, gaussians_popt = fit_multiple_gaussians(single_frame_input,maximas,sigmas_guesses,half_dist)
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
                
        accumulator = np.array(accumulator)
        df = pd.DataFrame()
        df['amplitude'] = accumulator[:,0]
        df['mu_x'] = accumulator[:,1]
        df['mu_y'] = accumulator[:,2]
        df['mu_z'] = accumulator[:,3]
        df['sigma_x'] = accumulator[:,4]
        df['sigma_y'] = accumulator[:,5]
        df['sigma_z'] = accumulator[:,6]
        
        # save the errors in the dataframe
        error_list, index_list = check_fitting_error(single_frame_input,maximas,gaussians,sigmas_guesses)

        df['errors'] = error_list


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
    

    def run_parallel_frame_processing(self, max_frames: int = 2, all_frames: bool = False):
        """
        Processes multiple frames in parallel using the single_frame_segmentation method and returns the combined results.
        
        Parameters:
            max_frames (int, optional): The maximum number of frames to process. Defaults to 2.
            all_frames (bool, optional): If true all the frames will be processed regardless of max_frames
            
        Returns:
            DataFrame: A pandas DataFrame containing the combined analysis results from all processed frames.
        """

        if all_frames == True: 
            frames_to_process = self.frames
        else:
            frames_to_process = max_frames

        num_of_parallel_process = self.cores_to_use()
        futures_to_frame = {}

        # Initialize a ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers = num_of_parallel_process) as executor:
        
            # Submit tasks for each frame to be processed in parallel
            for frame in range(frames_to_process):
                future = executor.submit(self.single_frame_segmentation, frame)
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
            filename_pkl = f'all_detections_channel{self.target_channel + 1}.pkl'

            # Construct the full file path by joining the directory and filename
            file_path = os.path.join(self.save_directory, filename_pkl)

            final_df.to_pickle(file_path)
            
            # Return the combined dataframe instead of saving
            return final_df