from peak_local_max_3d import peak_local_max_3d 
from gaussian_fitting import fit_multiple_gaussians
from extract_data import extract_data_from_filename
from gaussian_visualization import visualize_3D_gaussians
from gaussian_fitting import check_fitting_error
import pandas as pd
import os 
import numpy as np


def single_frame_segmentation(frame, save_directory,zarr_file,spot_intensity,dist_between_spots):
    
    
    #get the number of frames for our original data for automated analysis for all frames 
    print('frame number is', frame)
    
    single_frame_input = zarr_file[frame]
    single_frame_input = single_frame_input[0,:,:,:]
    print(single_frame_input.shape)
    single_frame_input = np.transpose(single_frame_input,axes =(0,2,1))
    print(single_frame_input.shape)

    #define threshold: a value(intensity) for the pixel below which all values would be considered noise and dropped 
    #define min_distance: min_distance/2 is the radius within which we will keep the peak with max value/intensity or 
    #if two peaks have the same value they will be kept 
    
    maximas = peak_local_max_3d(single_frame_input,min_distance=dist_between_spots,threshold=spot_intensity)
    print('local_maximas detected are', maximas.shape[0])


    #give the expected std dev/radius of our particles for x,y,z 
    sigmaExpected_x__pixels = 2
    sigmaExpected_y__pixels = 2
    sigmaExpected_z__pixels = 4

    sigmas_guesses = []
    for i in range(len(maximas)):
        sigmas_guesses.append([sigmaExpected_z__pixels,sigmaExpected_x__pixels,sigmaExpected_y__pixels])
        
    #last parameter in the fit_multiple_gaussians is similar to min_distance above, we should give half of the 
    #value here of min_distance   
    gaussians, gaussians_popt = fit_multiple_gaussians(single_frame_input,maximas,sigmas_guesses,5)
        
    accumulator = []
    for gaussian in gaussians:

        if(gaussian!=-1):
            amplitude = gaussian[0]

            #print(gaussian)
            mu_x     = int(gaussian[1][1]) ##this is going to be mu_z, previous code [1][0]
            mu_y     = int(gaussian[1][2]) ##need to finalise what this is (x or y) [1][1]
            mu_z     = int(gaussian[1][0]) ##need to finalise what this is (x or y) [1][2]
            ##sigmas will also change due to the above 
            sigma_x  = int(gaussian[2][1]) 
            sigma_y  = int(gaussian[2][2])
            sigma_z  = int(gaussian[2][0])
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
    df.head()
    
    error_list, index_list = check_fitting_error(single_frame_input,maximas,gaussians,sigmas_guesses)
    
    # Construct the filename
    '''
    filename_csv = f'df_c2_t{frame}.csv'
    file_path_csv = os.path.join(csv_save_dir, filename_csv)
    df.to_csv(file_path_csv)
    '''

    # Construct the filename based on the loop index (time_frame)
    filename_pkl = f'df_c3_t{frame}.pkl'

    # Construct the full file path by joining the directory and filename
    file_path = os.path.join(save_directory, filename_pkl)

    # Save the DataFrame to a pickle file with the specified path
    df.to_pickle(file_path)