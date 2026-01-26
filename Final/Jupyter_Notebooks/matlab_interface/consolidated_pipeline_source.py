#### Part 1: Load MATLAB Detections in dataframes for all channels and view in Napari ####
import dask.array as da 
import hdf5storage
# import mat73
import matplotlib.pyplot as plt
import napari
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree
from scipy.stats import anderson
from scipy.stats import t as t_dist
import sys
import time
from tqdm import tqdm
import zarr
import warnings
warnings.filterwarnings('ignore')

llsmtools_path= r'C:\Users\Lab admin\Desktop\llsmtools\psdetect3d'

# MATLAB Engine setup
try:
    import matlab.engine
    print("Initializing MATLAB engine...")
    eng = matlab.engine.start_matlab()
    # Add path to fitGaussian3D MEX file
    # Adjust this path to your llsmtools location
    eng.addpath(llsmtools_path, nargout=0) # type: ignore
    print("MATLAB engine initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize MATLAB engine: {e}")
    eng = None


def load_channel_detections(mat_filepath, sigma_xy, sigma_z):
    """
    Load detection data from MATLAB .mat file and convert to DataFrame.
    
    Parameters
    ----------
    mat_filepath : str or Path
        Path to channel_X.mat file
    sigma_xy : float
        Sigma value in x/y dimensions used for detection
    sigma_z : float
        Sigma value in z dimension used for detection
        
    Returns
    -------
    pd.DataFrame
        Detections with columns: mu_x, mu_y, mu_z, amplitude, intensity,
        x_conf, y_conf, z_conf, amp_conf, int_conf, 
        sigmaX, sigmaY, sigmaZ, bkg, bkg_conf, sigmaRes, sigmaRes_conf,
        RSS, pval_Ar, hval_Ar, hval_AD
        Index: frame (0-based)
    """

    # Load MATLAB file
    mat_data = hdf5storage.loadmat(mat_filepath)
    detection_data = mat_data['movieInfo']
    
    all_detections = []
    
    # Process each frame
    for frame_idx in range(detection_data.shape[0]): # This makes the frame_idx 0-based
        frame_data = detection_data[frame_idx, 0]
        
        # Skip empty frames
        if frame_data.size == 0 or 'xCoord' not in frame_data.dtype.names:
            continue
        
        # Extract coordinate and amplitude arrays
        x_coords = frame_data['xCoord'] - 1 # Convert to 0-based indexing from Matlab to Python
        y_coords = frame_data['yCoord'] - 1 # Convert to 0-based indexing from Matlab to Python
        z_coords = frame_data['zCoord'] - 1 # Convert to 0-based indexing from Matlab to Python
        amp_vals = frame_data['amp']
        int_vals = frame_data['int']
        
        # Extract new fields from expanded movieInfo
        sigma_x_vals = frame_data['sigmaX'] if 'sigmaX' in frame_data.dtype.names else None
        sigma_y_vals = frame_data['sigmaY'] if 'sigmaY' in frame_data.dtype.names else None
        sigma_z_vals = frame_data['sigmaZ'] if 'sigmaZ' in frame_data.dtype.names else None
        bkg_vals = frame_data['bkg'] if 'bkg' in frame_data.dtype.names else None
        sigma_res_vals = frame_data['sigmaRes'] if 'sigmaRes' in frame_data.dtype.names else None
        rss_vals = frame_data['RSS'] if 'RSS' in frame_data.dtype.names else None
        pval_ar_vals = frame_data['pval_Ar'] if 'pval_Ar' in frame_data.dtype.names else None
        hval_ar_vals = frame_data['hval_Ar'] if 'hval_Ar' in frame_data.dtype.names else None
        hval_ad_vals = frame_data['hval_AD'] if 'hval_AD' in frame_data.dtype.names else None
        
        # Skip if any arrays are empty
        if any(arr.size == 0 for arr in [x_coords, y_coords, z_coords, amp_vals, int_vals]):
            print(f"Skipping frame {frame_idx} due to empty detection arrays")
            continue

        # Ensure arrays are 2D (add second dimension if missing)
        if x_coords.ndim == 1:
            x_coords = x_coords.reshape(-1, 1)
            y_coords = y_coords.reshape(-1, 1)
            z_coords = z_coords.reshape(-1, 1)
            amp_vals = amp_vals.reshape(-1, 1)
            int_vals = int_vals.reshape(-1, 1)
        
        num_detections = x_coords.shape[0]
        
        # Process each detection
        for i in range(num_detections):
            # Skip NaN coordinates
            if np.isnan(x_coords[i, 0]) or np.isnan(y_coords[i, 0]) or np.isnan(z_coords[i, 0]):
                print(f"Frame {frame_idx}, Detection {i}: NaN value found in coordinates")
                continue
            
            detection = {
                'frame': frame_idx,
                'x': x_coords[i, 0],
                'y': y_coords[i, 0],
                'z': z_coords[i, 0],
                'A': amp_vals[i, 0],
                'intensity': int_vals[i, 0],
                'x_conf': x_coords[i, 1] if x_coords.shape[1] > 1 else np.nan,
                'y_conf': y_coords[i, 1] if y_coords.shape[1] > 1 else np.nan,
                'z_conf': z_coords[i, 1] if z_coords.shape[1] > 1 else np.nan,
                'A_pstd': amp_vals[i, 1] if amp_vals.shape[1] > 1 else np.nan,
                'int_conf': int_vals[i, 1] if int_vals.shape[1] > 1 else np.nan,
            }
            
            # Add new fields if available
            if sigma_x_vals is not None:
                if sigma_x_vals.ndim == 1:
                    sigma_x_vals = sigma_x_vals.reshape(-1, 1)
                detection['sigma_x'] = sigma_x_vals[i, 0]
                detection['sigma_x_conf'] = sigma_x_vals[i, 1] if sigma_x_vals.shape[1] > 1 else np.nan
                
            if sigma_y_vals is not None:
                if sigma_y_vals.ndim == 1:
                    sigma_y_vals = sigma_y_vals.reshape(-1, 1)
                detection['sigma_y'] = sigma_y_vals[i, 0]
                detection['sigma_y_conf'] = sigma_y_vals[i, 1] if sigma_y_vals.shape[1] > 1 else np.nan
                
            if sigma_z_vals is not None:
                if sigma_z_vals.ndim == 1:
                    sigma_z_vals = sigma_z_vals.reshape(-1, 1)
                detection['sigma_z'] = sigma_z_vals[i, 0]
                detection['sigma_z_conf'] = sigma_z_vals[i, 1] if sigma_z_vals.shape[1] > 1 else np.nan
                
            if bkg_vals is not None:
                if bkg_vals.ndim == 1:
                    bkg_vals = bkg_vals.reshape(-1, 1)
                detection['c'] = bkg_vals[i, 0]
                detection['c_pstd'] = bkg_vals[i, 1] if bkg_vals.shape[1] > 1 else np.nan
                
            if sigma_res_vals is not None:
                if sigma_res_vals.ndim == 1:
                    sigma_res_vals = sigma_res_vals.reshape(-1, 1)
                detection['sigma_r'] = sigma_res_vals[i, 0]
                detection['SE_sigma_r'] = sigma_res_vals[i, 1] if sigma_res_vals.shape[1] > 1 else np.nan
                
            if rss_vals is not None and rss_vals.size > 0:
                if rss_vals.ndim == 0:  # scalar
                    detection['RSS'] = float(rss_vals)
                else:
                    detection['RSS'] = rss_vals[i, 0]
                
            if pval_ar_vals is not None and pval_ar_vals.size > 0:
                if pval_ar_vals.ndim == 0:  # scalar
                    detection['pval_Ar'] = float(pval_ar_vals)
                else:
                    detection['pval_Ar'] = pval_ar_vals[i, 0] 
                
            if hval_ar_vals is not None and hval_ar_vals.size > 0:
                if hval_ar_vals.ndim == 0:  # scalar
                    hval = hval_ar_vals
                else:
                    hval =  hval_ar_vals[i, 0]
                detection['hval_Ar'] = hval
                
            if hval_ad_vals is not None and hval_ad_vals.size > 0:
                if hval_ad_vals.ndim == 0:  # scalar
                    hval = hval_ad_vals
                else:
                    hval = hval_ad_vals[i, 0]
                detection['hval_AD'] = hval
            
            all_detections.append(detection)
    
    # Create DataFrame
    df = pd.DataFrame(all_detections).set_index('frame')
    
    # Add detection sigma parameters (for reference)
    df['detection_sigma_xy'] = sigma_xy
    df['detection_sigma_z'] = sigma_z
    
    return df


def import_all_detections(channel_config):
    """
    Import detections from all specified channels.
    
    Parameters
    ----------
    channel_config : dict
        Configuration for each channel. Keys are channel numbers (1, 2, 3),
        values are dicts with keys:
        - 'enabled': bool
        - 'sigma_xy': float
        - 'sigma_z': float
        - 'input_file': str or Path (path to channel_X.mat file)
        - 'output_file': str or Path (path to save output file)
        
        Example: {
            1: {
                'enabled': True, 
                'sigma_xy': 1.8, 
                'sigma_z': 2.3,
                'input_file': '/path/to/detection/channel_1.mat',
                'output_file': '/path/to/output/detections_channel_1.pkl'
            },
            2: {
                'enabled': True, 
                'sigma_xy': 1.4, 
                'sigma_z': 2.0,
                'input_file': '/path/to/detection/channel_2.mat',
                'output_file': '/path/to/output/detections_channel_2.pkl'
            },
            3: {
                'enabled': False,  # This channel will be skipped
                'sigma_xy': 1.38, 
                'sigma_z': 1.94,
                'input_file': '/path/to/detection/channel_3.mat',
                'output_file': '/path/to/output/detections_channel_3.pkl'
            },
        }
        
    Returns
    -------
    dict
        Dictionary mapping channel numbers to DataFrames (only enabled channels)
    """
    results = {}
    
    for channel_num, config in channel_config.items():
        if not config['enabled']:
            print(f"Channel {channel_num}: Skipped (disabled)")
            continue
        
        # Get paths
        input_file = Path(config['input_file'])
        output_file = Path(config['output_file'])
        
        # Check input exists
        if not input_file.exists():
            print(f"Channel {channel_num}: File not found ({input_file})")
            continue
        
        # Load detections
        print(f"Channel {channel_num}: Loading detections from {input_file}")
        df = load_channel_detections(
            input_file,
            sigma_xy=config['sigma_xy'],
            sigma_z=config['sigma_z']
        )
        
        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to pickle
        df.to_pickle(output_file)
        print(f"Channel {channel_num}: Saved {len(df)} detections to {output_file}")
        
        results[channel_num] = df
    
    return results

def visualize_3D_gaussians(zarr_obj, gaussians_df):
    """
    Visualizes 3D Gaussians based on the parameters extracted from a DataFrame 
    and overlays them onto a 3D array. Uses sub-voxel precision for accurate 
    centering of Gaussian blobs.

    Parameters
    ----------
    zarr_obj : zarr.core.Array
        The raw 3D image data from which the Gaussians have been segmented and fitted.
        Expected shape: (T, C, Z, Y, X) where T=time, C=channel.
    gaussians_df : pd.DataFrame
        A DataFrame containing the Gaussian parameters with columns:
        - 'A': amplitude
        - 'x', 'y', 'z': sub-voxel center coordinates
        - 'sigma_x', 'sigma_y', 'sigma_z': Gaussian widths

    Returns
    -------
    np.ndarray
        A 3D array (Z, Y, X) with the visualized Gaussians.
    
    Notes
    -----
    - Uses np.round() instead of int() for converting sub-voxel centers to integer 
      voxel indices, reducing maximum positioning error from ~1 voxel to ~0.5 voxel.
    - Sub-voxel precision is preserved in the Gaussian distance calculation for 
      accurate rendering, while integer indices are only used for neighborhood bounds.
    - Neighborhood size extends int(3*sigma) + 1 voxels in each direction, ensuring
      the full Gaussian tail (>99.7% of intensity) is captured.
    """
    
    # Initialize output array with shape (Z, Y, X)
    image_gaussians = np.zeros((zarr_obj.shape[2], zarr_obj.shape[3], zarr_obj.shape[4]))

    # Avoid modifying the original DataFrame
    gaussians_df = gaussians_df.copy()
    
    # # Handle zero sigmas to prevent division by zero
    # gaussians_df.loc[gaussians_df['sigma_x'] == 0, 'sigma_x'] = 1
    # gaussians_df.loc[gaussians_df['sigma_y'] == 0, 'sigma_y'] = 1
    # gaussians_df.loc[gaussians_df['sigma_z'] == 0, 'sigma_z'] = 1
    
    # Extract Gaussian parameters - keep sub-voxel precision for centers
    amplitudes = gaussians_df['A'].values * 100  # Scale for visualization
    mu_xs_subvoxel = gaussians_df['x'].values
    mu_ys_subvoxel = gaussians_df['y'].values
    mu_zs_subvoxel = gaussians_df['z'].values
    sigma_xs = gaussians_df['sigma_x'].values
    sigma_ys = gaussians_df['sigma_y'].values
    sigma_zs = gaussians_df['sigma_z'].values
    
    # Convert to integer indices using round() for neighborhood bounds only
    # round() gives max error of ±0.5 voxels vs int() which can give up to -0.99 voxels
    mu_xs_int = np.round(mu_xs_subvoxel).astype(int)
    mu_ys_int = np.round(mu_ys_subvoxel).astype(int)
    mu_zs_int = np.round(mu_zs_subvoxel).astype(int)

    for i in range(len(amplitudes)):
        # Get parameters for this Gaussian
        amplitude = amplitudes[i]
        sigma_x, sigma_y, sigma_z = sigma_xs[i], sigma_ys[i], sigma_zs[i]
        
        # Sub-voxel centers for accurate Gaussian rendering
        mu_x, mu_y, mu_z = mu_xs_subvoxel[i], mu_ys_subvoxel[i], mu_zs_subvoxel[i]
        
        # Integer centers for neighborhood bounds
        mu_x_int, mu_y_int, mu_z_int = mu_xs_int[i], mu_ys_int[i], mu_zs_int[i]
        
        # Calculate neighborhood size: int(3*sigma) + 1 captures >99.7% of Gaussian
        n_neighbors_x = int(3 * sigma_x) + 1
        n_neighbors_y = int(3 * sigma_y) + 1
        n_neighbors_z = int(3 * sigma_z) + 1

        # Define voxel ranges, clipped to image boundaries
        z_range = np.arange(max(0, mu_z_int - n_neighbors_z), 
                           min(image_gaussians.shape[0], mu_z_int + n_neighbors_z + 1))
        y_range = np.arange(max(0, mu_y_int - n_neighbors_y), 
                           min(image_gaussians.shape[1], mu_y_int + n_neighbors_y + 1))
        x_range = np.arange(max(0, mu_x_int - n_neighbors_x), 
                           min(image_gaussians.shape[2], mu_x_int + n_neighbors_x + 1))

        # Create 3D meshgrid for vectorized distance calculation
        zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        
        # Calculate squared Mahalanobis distance using SUB-VOXEL centers
        # This gives accurate Gaussian shape even when center is between voxels
        distances = (
            ((zz - mu_z) ** 2) / (2 * sigma_z ** 2) +
            ((yy - mu_y) ** 2) / (2 * sigma_y ** 2) +
            ((xx - mu_x) ** 2) / (2 * sigma_x ** 2)
        )
        
        # Compute Gaussian values and add to output (handles overlapping Gaussians)
        gaussian_values = amplitude * np.exp(-distances)
        np.add.at(image_gaussians, (zz, yy, xx), gaussian_values)
    
    return image_gaussians

# Visualizing the detections in Napari
def napari_visualization(df, zarr_obj, plot_frame, channel_to_detect):
    df_reset = df.reset_index()
    masks = visualize_3D_gaussians(zarr_obj, gaussians_df = df_reset[df_reset['frame'] == plot_frame])
    # Create a napari viewer
    viewer = napari.Viewer()
    #open the zarr file in read mode
    dask_array = da.from_zarr(zarr_obj)
    dask_array_slice = dask_array[plot_frame,channel_to_detect-1,:,:,:]
    # Add the 3D stack to the viewer
    layer_raw = viewer.add_image(dask_array_slice, name='fluorescence', interpolation3d = 'nearest', blending = 'additive', colormap = 'magenta')
    # layer_mask = viewer.add_image(masks, name = 'detections mask')
    layer_mask = viewer.add_image(masks, name = 'detections', interpolation3d = 'nearest', blending = 'additive', colormap = 'green')
    layer_raw.bounding_box.visible = True # type: ignore


#### Part 2: Import tracks from the main channel (AP2) ####
def convert_tracks_to_dataframe(track_paths):
    """
    Convert MATLAB tracking data to a pandas DataFrame.
    
    Parameters
    ----------
    track_paths : str or Path
        Path to the MATLAB .mat file containing tracking results
    
    Returns
    -------
    track_df : pd.DataFrame
        DataFrame with columns:
        - 'frame': frame number (1-based, from MATLAB)
        - 'x', 'y', 'z': coordinates (0-based, converted from MATLAB)
        - 'A': amplitude
        - 'track_id': unique track identifier
        - 'segment_id': segment identifier (only for split/merge tracks)
    regular_tracks : list
        List of track_ids for regular (non-split/merge) tracks
    split_merge_tracks : list
        List of track_ids for tracks with split or merge events
    
    Notes
    -----
    - Coordinates are converted from MATLAB 1-based to Python 0-based indexing.
    - Split/merge tracks contain a 'segment_id' column to distinguish segments.
    - DataFrame is sorted by frame and track_id.
    """


    mat_data = hdf5storage.loadmat(track_paths)
    
    
    # Extract the tracks structure
    tracks = mat_data['tracksFinal'].flatten()
    
    # Create an empty list to store all track data
    all_tracks = []
    regular_tracks = []
    split_merge_tracks = []
    
    # Process each track
    for track_id in range(len(tracks)):

        # Handle numpy.void objects by accessing elements with field names
        track = tracks[track_id]
        # Access fields using dictionary-like indexing for numpy.void objects
        track_coords = track['tracksCoordAmpCG']
        seq_events = track['seqOfEvents']
        
        # Handle different shapes of seqOfEvents
        if seq_events.size == 0:
            continue  # Skip empty tracks
            
        # Reshape if necessary to ensure consistent format
        if len(seq_events.shape) == 1:
            seq_events = seq_events.reshape(1, -1)

        if np.isnan(seq_events[:, -1]).all():

            track_coords = track_coords[0]

            # save the track as a regular track
            regular_tracks.append(track_id)

            # Find start and end frames
            start_frame = int(seq_events[0, 0])
            end_frame = int(seq_events[-1, 0])

            # Determine the total number of frames from the size of tracksCoordAmpCG
            # Each frame has 8 columns [x y z a dx dy dz da]
            num_cols = track_coords.shape[0]
            num_frames = num_cols // 8

            # For each frame in the track's lifespan
            for frame_idx in range(num_frames):
                frame_number = start_frame + frame_idx - 1 # Convert to 0-based indexing from MATLAB to Python. This makes the frame index 0-based.
                # Extract x, y, z coordinates for current frame
                col_idx = frame_idx * 8

                x = track_coords[col_idx] - 1 # Convert to 0-based indexing from Matlab to Python
                y = track_coords[col_idx + 1] - 1 # Convert to 0-based indexing from Matlab to Python
                z = track_coords[col_idx + 2] - 1 # Convert to 0-based indexing from Matlab to Python
                amplitude = track_coords[col_idx + 3]

                track_data = {
                    'frame': frame_number,
                    'x': (x) if not np.isnan(x) else None,
                    'y': (y) if not np.isnan(y) else None,
                    'z': (z) if not np.isnan(z) else None,
                    'A': amplitude,
                    'track_id': track_id  
                }

                all_tracks.append(track_data)

        else:
            # save the track as a split/merge track
            split_merge_tracks.append(track_id)

            # Determine if the track is split or merged (this seems a bit rudimentary, probably could be better)
            # Abhishek Raghunathan, 04/14/25
            if not np.isnan(seq_events[1, -1]): # This is assuming that we have only a single split event, it will fail otherwise.
                # Also that all split events are position 1 in seq_events
                track_flag = 'split'
                # print(f'Track {track_id} is split.')

            if not np.isnan(seq_events[2, -1]): # This is assuming that we have only a single merge event, it will fail otherwise.
                # Also that all merge events are position 2 in seq_events
                track_flag = 'merge'
                # print(f'Track {track_id} is merged.')
            

            # Process each segment in the track
            segments = np.unique(seq_events[:, 2]).astype(int)
            segments_min = np.min(segments) # Assuming the lowest segment ID is the first one (might not be true).
            start_frame_original = [] #To store the original start frame
            
            for segment_id in segments:
                # Find events related to this segment
                segment_events = seq_events[seq_events[:, 2] == segment_id]
                
                # Get start and end frames for this segment
                start_events = segment_events[segment_events[:, 1] == 1]
                end_events = segment_events[segment_events[:, 1] == 2]

                if segment_id == segments_min: # Assuming the lowest segment ID has the track which started first (lower value of first frame). CHECK THIS.
                    start_frame_original.append(int(start_events[0,0])) # This is the original start frame for the split or merge track
                    # print(start_frame_original)
                
                if start_events.size > 0 and end_events.size > 0:
                    start_frame = int(start_events[0, 0])
                    end_frame = int(end_events[0, 0])
                    
                    # Get row index for this segment (0-indexed)
                    segment_idx = segment_id - 1

                    # Get the row data for this segment
                    if segment_idx < len(track_coords):
                        segment_data = track_coords[segment_idx]
                        
                        # Calculate number of frames in this segment
                        segment_frames = end_frame - start_frame + 1
                        
                        # Process each frame in this segment
                        for frame_offset in range(segment_frames):
                            frame_number = start_frame + frame_offset - 1 # Convert to 0-based indexing from MATLAB to Python

                            if (track_flag == 'split') or (track_flag == 'merge'): # type: ignore # This flag is unnecessary here, have it for legacy reasons.
                                col_idx = (start_frame + frame_offset - start_frame_original[0]) * 8 # This will handle cases where start_frame_original is not 1
                            else:
                                col_idx = frame_offset * 8
                            
                            # if track_id == 4370:
                            #     print(col_idx)
                            
                            
                            # Check if indices are within bounds
                            if col_idx + 3 < len(segment_data):
                                x = segment_data[col_idx] - 1 # Convert to 0-based indexing from Matlab to Python
                                y = segment_data[col_idx + 1] - 1 # Convert to 0-based indexing from Matlab to Python
                                z = segment_data[col_idx + 2] - 1 # Convert to 0-based indexing from Matlab to Python
                                amplitude = segment_data[col_idx + 3]
                                
                                track_data = {
                                    'frame': frame_number,
                                    'x': (x) if not np.isnan(x) else None,
                                    'y': (y) if not np.isnan(y) else None,
                                    'z': (z) if not np.isnan(z) else None,
                                    'A': amplitude,
                                    'track_id': track_id,
                                    'segment_id': segment_id
                                }
                                all_tracks.append(track_data)
        
    
    track_df = pd.DataFrame(all_tracks)

    # Sort by frame and track_id
    track_df = track_df.sort_values(['frame', 'track_id'])

    return track_df, regular_tracks, split_merge_tracks

def enrich_tracks_with_detections(track_df, detection_df, decimal_precision=4):
    """
    Enrich track DataFrame with additional columns from detection DataFrame.
    
    Matches track points to detections based on exact frame and coordinate match,
    then adds all detection columns (uncertainties, sigmas, background, etc.)
    to the track DataFrame.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame with columns: frame, x, y, z, A, track_id
    detection_df : pd.DataFrame
        Detection DataFrame with columns: x, y, z, A, and additional fields
        Index: frame
    decimal_precision : int, optional
        Number of decimal places to round coordinates for matching.
        Default is 4 decimal places (~0.1 nm precision at typical voxel sizes).
    
    Returns
    -------
    pd.DataFrame
        Track DataFrame with all detection columns added.
        Unmatched points will have NaN for detection-specific columns.
    
    Notes
    -----
    - Matching is done by exact (frame, x, y, z) match after rounding.
    - Track amplitude 'A' is preserved; detection amplitude added as 'A_det'.
    - If match rate is low, this indicates a data pipeline issue.
    - Both DataFrames are copied to avoid modifying originals.
    """
    
    # Copy DataFrames to avoid modifying originals
    track_df = track_df.copy()
    det_df = detection_df.reset_index().copy()
    
    # Round coordinates to avoid floating-point precision issues
    for col in ['x', 'y', 'z']:
        track_df[col] = track_df[col].round(decimal_precision)
        det_df[col] = det_df[col].round(decimal_precision)
    
    # Rename detection 'A' to avoid conflict with track 'A'
    det_df = det_df.rename(columns={'A': 'A_det'})
    
    # Merge on exact (frame, x, y, z) match
    track_df_enriched = track_df.merge(
        det_df,
        on=['frame', 'x', 'y', 'z'],
        how='left',
        suffixes=('', '_from_det')
    )
    
    return track_df_enriched

#### Part 3: Filter valid tracks in the main channel (AP2) ####
def classify_tracks_by_lifetime(track_df, n_frames, buffer_frames=5, min_track_length=3):
    """
    Classify tracks based on their temporal extent and position relative to movie boundaries.
    
    Separates tracks into four categories:
    - Valid: Complete tracks with sufficient buffer at both ends
    - Invalid: Very short tracks (< min_track_length frames)
    - Partial: Tracks at movie boundaries without sufficient buffer
    - Persistent: Tracks spanning the entire movie (present at both boundaries)
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame with columns: frame, x, y, z, A, track_id, and enriched detection columns
    n_frames : int
        Total number of frames in the movie
    buffer_frames : int, optional
        Number of frames required at movie boundaries for valid tracks. Default is 5.
    min_track_length : int, optional
        Minimum number of frames for a track to be considered valid. Default is 3.
    
    Returns
    -------
    dict
        Dictionary with keys 'valid', 'invalid', 'partial', 'persistent',
        each containing a list of track_ids belonging to that category.
    
    Notes
    -----
    - A track is 'persistent' if it starts within buffer_frames of frame 0 
      AND ends within buffer_frames of the last frame.
    - A track is 'partial' if it starts OR ends within buffer_frames of a boundary
      (but not both, which would make it persistent).
    - A track is 'invalid' if it has fewer than min_track_length frames.
    - A track is 'valid' if it passes all the above criteria.
    
    Classification order matters: invalid → persistent → partial → valid
    """
    
    # Initialize classification lists
    valid_tracks = []
    invalid_tracks = []
    partial_tracks = []
    persistent_tracks = []
    
    # Get unique track IDs
    track_ids = track_df['track_id'].unique()
    
    for track_id in track_ids:
        # Get all frames for this track
        track_frames = track_df[track_df['track_id'] == track_id]['frame'].values
        
        track_start = np.min(track_frames)
        track_end = np.max(track_frames)
        track_length = len(track_frames)
        
        # 1. Check for very short tracks (invalid)
        if track_length < min_track_length:
            invalid_tracks.append(track_id)
            continue
        
        # 2. Check for persistent tracks (spans entire movie)
        at_start_boundary = (track_start < buffer_frames)
        at_end_boundary = (track_end >= n_frames - buffer_frames)
        
        if at_start_boundary and at_end_boundary:
            persistent_tracks.append(track_id)
            continue
        
        # 3. Check for partial tracks (at one boundary only)
        if at_start_boundary or at_end_boundary:
            partial_tracks.append(track_id)
            continue
        
        # 4. Track passes all criteria - it's valid
        valid_tracks.append(track_id)
    
    # Create results dictionary
    classification = {
        'valid': valid_tracks,
        'invalid': invalid_tracks,
        'partial': partial_tracks,
        'persistent': persistent_tracks
    }
    
    # Print summary
    total = len(track_ids)
    print(f"Track classification summary (n_frames={n_frames}, buffer={buffer_frames}):")
    print(f"  Valid:      {len(valid_tracks):6d} ({100*len(valid_tracks)/total:.1f}%)")
    print(f"  Invalid:    {len(invalid_tracks):6d} ({100*len(invalid_tracks)/total:.1f}%) - fewer than {min_track_length} frames")
    print(f"  Partial:    {len(partial_tracks):6d} ({100*len(partial_tracks)/total:.1f}%) - at one boundary")
    print(f"  Persistent: {len(persistent_tracks):6d} ({100*len(persistent_tracks)/total:.1f}%) - spans movie")
    print(f"  Total:      {total:6d}")
    
    return classification

# =============================================================================
# MATLAB ENGINE SETUP
# =============================================================================
# Global variable for MATLAB engine - initialized once per session


def get_matlab_engine(llsmtools_path):
    """
    Lazily initialize and return a singleton MATLAB engine instance.

    The MATLAB engine is started at most once per Python process. If
    initialization fails, subsequent calls return None without retrying.
    This allows callers to reliably fall back to a Python implementation.

    Parameters
    ----------
    llsmtools_path : str
        Path to the llsmtools `psdetect3d` directory containing the
        `fitGaussian3D` MEX files.

    Returns
    -------
    matlab.engine.MatlabEngine or None
        An active MATLAB engine instance if initialization succeeds,
        otherwise None.
    """
    global matlab_engine, matlab_init_attempted

    if matlab_init_attempted:
        return matlab_engine

    try:
        import matlab.engine
        import os

        if not os.path.isdir(llsmtools_path):
            raise ValueError(f"Invalid llsmtools path: {llsmtools_path}")

        matlab_init_attempted = True
        print("Initializing MATLAB engine...")

        engine = matlab.engine.start_matlab()
        engine.addpath(llsmtools_path, nargout=0) # type: ignore

        matlab_engine = engine
        print("MATLAB engine initialized successfully")
        return matlab_engine

    except Exception as e:
        matlab_init_attempted = True
        matlab_engine = None
        print(f"Warning: Could not initialize MATLAB engine: {e}")
        print("Will use Python fallback for Gaussian fitting")
        return None

def cleanup_matlab_engine():
    """
    Shut down the MATLAB engine and reset initialization state.

    After calling this function, a subsequent call to
    `get_matlab_engine()` may attempt to start MATLAB again.
    """
    global matlab_engine, matlab_init_attempted

    if matlab_engine is not None:
        try:
            matlab_engine.quit()  # type: ignore
            print("MATLAB engine closed")
        except Exception as e:
            print(f"Warning: Error while closing MATLAB engine: {e}")

    matlab_engine = None
    matlab_init_attempted = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_binary_segment_lengths(binary_array):
    """
    Get lengths and values of consecutive segments in a binary array.
    
    Used to find runs of consecutive non-significant (False) frames in buffer.
    
    Parameters
    ----------
    binary_array : array-like
        Boolean or binary (0/1) array where True = significant, False = non-significant
        
    Returns
    -------
    lengths : ndarray
        Length of each consecutive segment
    values : ndarray  
        Value (0 or 1) of each segment. 0 = non-significant run.
        
    Examples
    --------
    >>> binary_array = [True, True, False, False, False, True]
    >>> lengths, values = get_binary_segment_lengths(binary_array)
    >>> # Returns: lengths = [2, 3, 1], values = [1, 0, 1]
    >>> # The middle segment has 3 consecutive non-significant frames
    """
    if len(binary_array) == 0:
        return np.array([]), np.array([])
    
    # Convert to numpy array of integers (True->1, False->0)
    arr = np.array(binary_array, dtype=int)
    
    # Find where value changes
    changes = np.where(np.diff(arr) != 0)[0] + 1
    
    # Add start and end positions
    boundaries = np.concatenate([[0], changes, [len(arr)]])
    
    # Calculate segment lengths
    lengths = np.diff(boundaries)
    
    # Get values of each segment
    values = arr[boundaries[:-1]]
    
    return lengths, values

def classify_tracks_by_buffer(track_df_valid, zarr_data, channel_config, 
                                tracking_channel, n_frames, p_threshold, k_level, 
                                buffer_frames=3, min_consecutive=2,
                                fit_mode='xyzAc', verbose=True):
    """
    Classify tracks based on Aguet buffer frame criteria.
    
    For each valid track (already filtered by lifetime), performs Gaussian
    fitting in buffer frames before/after the track to determine if the
    signal was absent (non-significant) in the buffer region.
    
    Aguet Criteria (simplified):
    - At least `min_consecutive` (default 2) consecutive non-significant 
      frames in BOTH start and end buffers
    
    Parameters
    ----------
    track_df_valid : pd.DataFrame
        DataFrame of valid tracks with columns: frame, x, y, z, A, track_id,
        and enriched detection columns.
        Should be pre-filtered to include only lifetime-valid tracks.
    zarr_data : zarr.Array
        Movie data with shape (T, C, Z, Y, X)
    channel_config : dict
        Channel configuration from notebook with sigma values.
        Example: {3: {'sigma_xy': 1.38, 'sigma_z': 1.94, ...}}
    tracking_channel : int
        Channel number for tracking (1-based, e.g., 3)
    n_frames : int
        Total number of frames in the movie
    p_threshold : float
        P-value threshold for significance (e.g., 0.05)
    k_level : float
        Z-score for significance threshold (e.g., 2.807 for α=0.005)
    llsmtools_path : str
        Path to llsmtools psdetect3d directory for MATLAB engine
    buffer_frames : int
        Number of buffer frames to check before/after each track (default 3)
    min_consecutive : int
        Minimum consecutive non-significant frames required (default 2)
    fit_mode : str
        Fitting mode: 'xyzAc' for position refinement (recommended),
        'Ac' for fixed position
    verbose : bool
        If True, print progress and summary statistics
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'buffer_valid': list of track_ids that pass buffer criteria
        - 'buffer_invalid': list of track_ids that fail buffer criteria
        - 'buffer_results': DataFrame with per-track buffer summary
        - 'track_df_with_buffer': DataFrame with original tracks plus buffer 
          frame fitting results, includes 'spot_type' column ('track' or 'buffer')
        
    Notes
    -----
    - Uses tracking channel sigma from channel_config
    - Calls MATLAB fitGaussian3D via fit_buffer_frame for each buffer frame
    - Position refinement (xyzAc) follows Aguet methodology
    - Tracks without sufficient buffer frames are marked invalid
    - Buffer frame results are appended to track_df_valid with spot_type='buffer'
    """
    # Get sigma values for tracking channel from channel_config
    sigma_xy = channel_config[tracking_channel]['sigma_xy']
    sigma_z = channel_config[tracking_channel]['sigma_z']
    sigma = [sigma_xy, sigma_z]
    
    # Channel index is 0-based (tracking_channel is 1-based)
    channel_idx = tracking_channel - 1
    
    if verbose:
        print(f"=" * 70)
        print(f"BUFFER-BASED TRACK CLASSIFICATION")
        print(f"=" * 70)
        print(f"Tracking channel: {tracking_channel} (index {channel_idx})")
        print(f"Sigma: [σ_xy={sigma_xy}, σ_z={sigma_z}]")
        print(f"Buffer frames: {buffer_frames}")
        print(f"Min consecutive non-significant: {min_consecutive}")
        print(f"P-value threshold: {p_threshold}")
        print(f"K-level: {k_level}")
        print(f"Fit mode: {fit_mode}")
        print(f"=" * 70)
    
    # Get unique track IDs
    track_ids = track_df_valid['track_id'].unique()
    
    if verbose:
        print(f"Processing {len(track_ids)} tracks...")
    
    # Results storage
    buffer_valid_tracks = []
    buffer_invalid_tracks = []
    buffer_results_list = []
    buffer_frame_rows = []  # Store buffer frame fitting results
    
    # Process each track
    for track_id in tqdm(track_ids, desc="Classifying tracks", disable=not verbose):
        # Get track data
        track_data = track_df_valid[track_df_valid['track_id'] == track_id].sort_values('frame')
        
        track_frames = track_data['frame'].values
        track_start = int(np.min(track_frames))
        track_end = int(np.max(track_frames))
        
        # Get first and last positions
        first_row = track_data.iloc[0]
        last_row = track_data.iloc[-1]
        
        x_start, y_start, z_start = first_row['x'], first_row['y'], first_row['z']
        x_end, y_end, z_end = last_row['x'], last_row['y'], last_row['z']
        
        # Initialize buffer results for this track
        track_buffer_result = {
            'track_id': track_id,
            'track_start': track_start,
            'track_end': track_end,
            'start_buffer_frames': [],
            'start_buffer_pvals': [],
            'end_buffer_frames': [],
            'end_buffer_pvals': [],
            'start_buffer_valid': False,
            'end_buffer_valid': False,
            'buffer_valid': False
        }
        
        # ===== PROCESS START BUFFER =====
        # Frames before track start: [track_start - buffer_frames, track_start - 1]
        start_buffer_frame_indices = range(max(0, track_start - buffer_frames), track_start)
        start_pvals = []
        
        for frame_idx in start_buffer_frame_indices:
            # Load frame data for tracking channel
            frame_data = zarr_data[frame_idx, channel_idx]
            
            # Fit Gaussian at track's first position
            fit_result = fit_buffer_frame(x_start, y_start, z_start, 
                                          frame_data, sigma, k_level, 
                                          fit_mode,
                                          debug=False)
            
            if fit_result is not None:
                pval = fit_result['pval_Ar']
                start_pvals.append(pval)
            
                # Create buffer frame row for DataFrame
                buffer_row = _create_buffer_row(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    fit_result=fit_result,
                    position=(x_start, y_start, z_start),
                    buffer_type='start'
                )
                buffer_frame_rows.append(buffer_row)

            else:
                print(f"⚠️  Warning: Gaussian fitting failed for track {track_id}, "
                          f"start buffer frame {frame_idx}. Skipping this frame.")
        
        track_buffer_result['start_buffer_frames'] = list(start_buffer_frame_indices)
        track_buffer_result['start_buffer_pvals'] = start_pvals
        
        # ===== PROCESS END BUFFER =====
        # Frames after track end: [track_end + 1, track_end + buffer_frames]
        end_buffer_frame_indices = range(track_end + 1, min(n_frames, track_end + buffer_frames + 1))
        end_pvals = []
        
        for frame_idx in end_buffer_frame_indices:
            # Load frame data for tracking channel
            frame_data = zarr_data[frame_idx, channel_idx]
            
            # Fit Gaussian at track's last position
            fit_result = fit_buffer_frame(x_end, y_end, z_end,
                                          frame_data, sigma, k_level,
                                          fit_mode,
                                          debug=False)
            
            if fit_result is not None:
                pval = fit_result['pval_Ar']       
                end_pvals.append(pval)
            
                # Create buffer frame row for DataFrame
                buffer_row = _create_buffer_row(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    fit_result=fit_result,
                    position=(x_end, y_end, z_end),
                    buffer_type='end'
                )
                buffer_frame_rows.append(buffer_row)

            else:
                print(f"⚠️  Warning: Gaussian fitting failed for track {track_id}, "
                          f"end buffer frame {frame_idx}. Skipping this frame.")
        
        track_buffer_result['end_buffer_frames'] = list(end_buffer_frame_indices)
        track_buffer_result['end_buffer_pvals'] = end_pvals
        
        # ===== APPLY AGUET CRITERIA =====
        # Check for min_consecutive non-significant frames in each buffer
        
        # Start buffer: significant if pval < threshold
        # print("Start pvals: ", start_pvals)
        start_significant = np.array([p < p_threshold for p in start_pvals])
        start_lengths, start_values = get_binary_segment_lengths(start_significant)
        
        # Look for segment with value=0 (non-significant) and length >= min_consecutive
        has_start_buffer_segment = np.any((start_lengths >= min_consecutive) & (start_values == 0))
        track_buffer_result['start_buffer_valid'] = bool(has_start_buffer_segment)
        
        # End buffer: same criteria
        end_significant = np.array([p < p_threshold for p in end_pvals])
        end_lengths, end_values = get_binary_segment_lengths(end_significant)
        
        has_end_buffer_segment = np.any((end_lengths >= min_consecutive) & (end_values == 0))
        track_buffer_result['end_buffer_valid'] = bool(has_end_buffer_segment)
        
        # Track is valid only if BOTH buffers pass
        track_valid = has_start_buffer_segment and has_end_buffer_segment
        track_buffer_result['buffer_valid'] = bool(track_valid)
        
        # Classify track
        if track_valid:
            buffer_valid_tracks.append(track_id)
        else:
            buffer_invalid_tracks.append(track_id)
        
        buffer_results_list.append(track_buffer_result)
    
    # Create results DataFrame for buffer summary
    buffer_results_df = pd.DataFrame(buffer_results_list)
    
    # Create combined DataFrame with tracks and buffer frames
    # Add spot_type and buffer_type columns to original track data
    track_df_copy = track_df_valid.copy()
    track_df_copy['spot_type'] = 'track'
    track_df_copy['buffer_type'] = np.nan  # 'start'/'end' only applies to buffer spots
    
    # Create buffer DataFrame
    if buffer_frame_rows:
        buffer_df = pd.DataFrame(buffer_frame_rows)
        buffer_df['spot_type'] = 'buffer'
        
        # Combine track and buffer DataFrames
        # Use concat with all columns; missing columns will be NaN
        track_df_with_buffer = pd.concat([track_df_copy, buffer_df], ignore_index=True)
        
        # Sort by track_id and frame for easy viewing
        track_df_with_buffer = track_df_with_buffer.sort_values(
            ['track_id', 'frame']
        ).reset_index(drop=True)
    else:
        track_df_with_buffer = track_df_copy
    
    # Print summary
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"BUFFER CLASSIFICATION RESULTS")
        print(f"{'=' * 70}")
        total = len(track_ids)
        n_valid = len(buffer_valid_tracks)
        n_invalid = len(buffer_invalid_tracks)
        print(f"  Buffer-valid tracks:   {n_valid:6d} ({100*n_valid/total:.1f}%)")
        print(f"  Buffer-invalid tracks: {n_invalid:6d} ({100*n_invalid/total:.1f}%)")
        print(f"  Total processed:       {total:6d}")
        
        # Breakdown of invalid reasons
        start_fail = buffer_results_df[~buffer_results_df['start_buffer_valid']].shape[0]
        end_fail = buffer_results_df[~buffer_results_df['end_buffer_valid']].shape[0]
        both_fail = buffer_results_df[
            (~buffer_results_df['start_buffer_valid']) & 
            (~buffer_results_df['end_buffer_valid'])
        ].shape[0]
        
        print(f"\nInvalid breakdown:")
        print(f"  Failed start buffer: {start_fail}")
        print(f"  Failed end buffer:   {end_fail}")
        print(f"  Failed both:         {both_fail}")
        
        # DataFrame summary
        n_track_rows = (track_df_with_buffer['spot_type'] == 'track').sum()
        n_buffer_rows = (track_df_with_buffer['spot_type'] == 'buffer').sum()
        print(f"\nCombined DataFrame:")
        print(f"  Track spots:  {n_track_rows}")
        print(f"  Buffer spots: {n_buffer_rows}")
        print(f"  Total rows:   {len(track_df_with_buffer)}")
        print(f"{'=' * 70}\n")
    
    return {
        'buffer_valid': buffer_valid_tracks,
        'buffer_invalid': buffer_invalid_tracks,
        'buffer_results': buffer_results_df,
        'track_df_with_buffer': track_df_with_buffer
    }


def _create_buffer_row(track_id, frame_idx, fit_result, position, buffer_type):
    """
    Create a dictionary row for a buffer frame fitting result.
    
    Parameters
    ----------
    track_id : int
        Track identifier
    frame_idx : int
        Frame index (0-based)
    fit_result : dict or None
        Fitting results from fit_buffer_frame, or None if fitting failed.
        Expected keys: x, y, z, A, c, A_pstd, c_pstd, sigma_r, SE_sigma_r,
        pval_Ar, hval_AD, sigma_xy, sigma_z
    position : tuple
        (x, y, z) position used for fitting (track start or end position)
    buffer_type : str
        'start' for start buffer, 'end' for end buffer
        
    Returns
    -------
    dict
        Row dictionary with all columns matching track_df_valid structure:
        frame, x, y, z, A, track_id, A_det, intensity, x_conf, y_conf, z_conf,
        A_pstd, int_conf, sigma_x, sigma_x_conf, sigma_y, sigma_y_conf, 
        sigma_z, sigma_z_conf, c, c_pstd, sigma_r, SE_sigma_r, RSS, pval_Ar,
        hval_Ar, hval_AD, detection_sigma_xy, detection_sigma_z, buffer_type
    """
    x_pos, y_pos, z_pos = position
    
    # Row with all columns from track_df_valid
    row = {
        # Core identification
        'frame': frame_idx,
        'track_id': track_id,
        
        # Position - use fitted position if available, otherwise input position
        'x': fit_result['x'] if fit_result is not None else x_pos,
        'y': fit_result['y'] if fit_result is not None else y_pos,
        'z': fit_result['z'] if fit_result is not None else z_pos,
        
        # Amplitude
        'A': fit_result['A'] if fit_result is not None else np.nan,
        'A_det': np.nan,  # Not applicable for buffer frames
        
        # Intensity (not available from fitting)
        'intensity': np.nan,
        
        # Position confidence intervals (not available from fitting)
        'x_conf': np.nan,
        'y_conf': np.nan,
        'z_conf': np.nan,
        
        # Amplitude uncertainty
        'A_pstd': fit_result['A_pstd'] if fit_result is not None else np.nan,
        
        # Intensity confidence (not available from fitting)
        'int_conf': np.nan,
        
        # Sigma values - use fitted sigma_xy for x and y, sigma_z for z
        'sigma_x': fit_result['sigma_xy'] if fit_result is not None else np.nan,
        'sigma_x_conf': np.nan,
        'sigma_y': fit_result['sigma_xy'] if fit_result is not None else np.nan,
        'sigma_y_conf': np.nan,
        'sigma_z': fit_result['sigma_z'] if fit_result is not None else np.nan,
        'sigma_z_conf': np.nan,
        
        # Background
        'c': fit_result['c'] if fit_result is not None else np.nan,
        'c_pstd': fit_result['c_pstd'] if fit_result is not None else np.nan,
        
        # Residual statistics
        'sigma_r': fit_result['sigma_r'] if fit_result is not None else np.nan,
        'SE_sigma_r': fit_result['SE_sigma_r'] if fit_result is not None else np.nan,
        
        # RSS (not returned by fit_buffer_frame)
        'RSS': np.nan,
        
        # Statistical tests
        'pval_Ar': fit_result['pval_Ar'] if fit_result is not None else 1.0,
        'hval_Ar': np.nan,  # Not computed in buffer fitting
        'hval_AD': fit_result['hval_AD'] if fit_result is not None else np.nan,
        
        # Detection sigma (not applicable for buffer frames)
        'detection_sigma_xy': np.nan,
        'detection_sigma_z': np.nan,
        
        # Buffer identification
        'buffer_type': buffer_type,  # 'start' or 'end'
    }
    
    return row

def fit_buffer_frame(x, y, z, frame_data, sigma, k_level, fit_mode, 
                    debug=False):
    """
    Fit Gaussian at track position in a single buffer frame.
    
    Extracts a local window around the track position and performs
    Gaussian fitting with position refinement (xyzAc mode) as per Aguet.
    
    Parameters
    ----------
    x, y, z : float
        Track position (sub-voxel coordinates, 0-based Python indexing)
    frame_data : ndarray
        3D image frame (z, y, x)
    sigma : list
        [sigma_xy, sigma_z] PSF parameters in pixels
    fit_mode : str
        'xyzAc' for position refinement (default, recommended)
        'Ac' for fixed position fitting
    matlab_engine : matlab.engine, optional
        MATLAB engine instance
    debug : bool
        If True, print debugging information
        
    Returns
    -------
    dict or None
        Fitting results including 'A', 'c', 'pval_Ar', etc.
        Returns None if fitting fails.
        
    Notes
    -----
    - Window size is determined by sigma: ±ceil(2*sigma) in each dimension
    - Coordinates are converted to window-local coordinates for fitting
    - Position bounds checking prevents out-of-window fits
    """
    nz, ny, nx = frame_data.shape
    
    # Convert to integer coordinates for window extraction
    xi = int(np.round(np.clip(x, 0, nx-1)))
    yi = int(np.round(np.clip(y, 0, ny-1)))
    zi = int(np.round(np.clip(z, 0, nz-1)))
    
    # Define window boundaries based on sigma
    # Window extends ±ceil(2*sigma) in each direction
    w_xy = int(np.ceil(2 * sigma[0]))  # Half-width in x,y
    w_z = int(np.ceil(2 * sigma[1]))   # Half-width in z
    
    # Extract window with boundary checking
    x_start = max(0, xi - w_xy)
    x_end = min(nx, xi + w_xy + 1)
    y_start = max(0, yi - w_xy)
    y_end = min(ny, yi + w_xy + 1)
    z_start = max(0, zi - w_z)
    z_end = min(nz, zi + w_z + 1)
    
    window = frame_data[z_start:z_end, y_start:y_end, x_start:x_end].copy()
    
    # Calculate position relative to window origin
    ox = xi - x_start  # x offset within window
    oy = yi - y_start  # y offset within window
    oz = zi - z_start  # z offset within window
    
    # Get initial amplitude and background estimates
    A_est, c_est = estimate_gaussian_amplitude_3d(window, sigma) # type: ignore
    
    # Get initial values at center position
    if not np.isnan(A_est[oz, oy, ox]):
        ai = A_est[oz, oy, ox]
    else:
        print("   Warning: Amplitude estimate is NaN, using max-min of window")
        ai = np.nanmax(window) - np.nanmin(window)
        
    if not np.isnan(c_est[oz, oy, ox]):
        ci = c_est[oz, oy, ox]
    else:
        print("   Warning: Background estimate is NaN, using min of window")
        ci = np.nanmin(window)
    
    # Prepare initial parameters
    # Use sub-voxel offset within window for precise positioning
    x_local = x - x_start  # Sub-voxel x position in window coordinates
    y_local = y - y_start  # Sub-voxel y position in window coordinates
    z_local = z - z_start  # Sub-voxel z position in window coordinates
    
    initial_params = [x_local, y_local, z_local, ai, sigma, ci]
    
    # Perform Gaussian fitting
    fit_result = fit_gaussian_3d_matlab(window, initial_params, sigma, k_level, fit_mode, # type: ignore
                                        debug=debug) # type: ignore
    
    if fit_result is None:
        print("   Warning: Gaussian fitting failed, returning None")
        return None
    
    # For xyzAc mode, check if fitted position is within reasonable bounds
    if fit_mode == 'xyzAc':
        w1_xy = int(np.ceil(sigma[0]))  # Inner bound (1 sigma)
        w1_z = int(np.ceil(sigma[1]))
        
        dx = fit_result['x'] - x_local
        dy = fit_result['y'] - y_local
        dz = fit_result['z'] - z_local
        
        if abs(dx) > w1_xy or abs(dy) > w1_xy or abs(dz) > w1_z:
            # Position fit moved too far - retry with Ac mode
            if debug:
                print(f"   Position out of bounds (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f})")
                print(f"   Retrying with Ac mode...")
            
            fit_result = fit_gaussian_3d_matlab(window, initial_params, sigma, k_level, # type: ignore
                                                fit_mode ='Ac', # type: ignore
                                                debug=debug) # type: ignore
            
            if fit_result is not None:
                # Keep original global position for Ac mode
                fit_result['x'] = x
                fit_result['y'] = y
                fit_result['z'] = z
        else:
            # Convert fitted position back to global coordinates
            fit_result['x'] = x_start + fit_result['x']
            fit_result['y'] = y_start + fit_result['y']
            fit_result['z'] = z_start + fit_result['z']
    
    return fit_result

def estimate_gaussian_amplitude_3d(frame, sigma):
    """
    Estimate initial amplitude and background for Gaussian fitting.
    
    Implements the linear least-squares estimation from Aguet et al.
    Uses FFT convolution for computational efficiency.
    
    Parameters
    ----------
    frame : ndarray
        3D image frame (z, y, x)
    sigma : list
        [sigma_xy, sigma_z] PSF parameters in pixels
    window_size : int
        Size of local window for background estimation (not used directly,
        window determined by sigma)
        
    Returns
    -------
    A_est : ndarray
        Estimated amplitude at each voxel (same shape as frame)
    c_est : ndarray
        Estimated background at each voxel (same shape as frame)
        
    Notes
    -----
    From Aguet Supplemental Procedures:
    - Solves linear system: [Σg²  Σg ] [A] = [Σgf]
                           [Σg   n  ] [c]   [Σf ]
    - Uses FFT convolution for speed
    """
    # Handle sigma input - convert scalar to [sigma_xy, sigma_z]
    if np.isscalar(sigma):
        sigma = [sigma, sigma]

    # Window size based on sigma (matches MATLAB: ws = ceil(2*sigma))
    ws_xy = int(np.ceil(2 * sigma[0])) # type: ignore
    ws_z = int(np.ceil(2 * sigma[1])) # type: ignore

    # Create 3D Gaussian kernel
    z, y, x = np.ogrid[-ws_z:ws_z+1, -ws_xy:ws_xy+1, -ws_xy:ws_xy+1]
    
    # Anisotropic Gaussian kernel
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma[0]**2)) * np.exp(-z**2 / (2 * sigma[1]**2)) # type: ignore

    # Number of elements in the kernel
    n = kernel.size

    # Pre-compute sums for linear system
    Σg = np.sum(kernel)        # Σg
    Σg2 = np.sum(kernel**2)    # Σg²
       
    # Compute convolutions (vectorized sums at each voxel)
    Σf  = fftconvolve(frame, np.ones_like(kernel), mode="same")
    Σgf = fftconvolve(frame, kernel, mode="same")

    # Solve linear system at each voxel
    # From: [Σg²  Σg ] [A] = [Σgf]
    #       [Σg   n  ] [c]   [Σf ]
    denominator = n * Σg2 - Σg**2  # Determinant of 2x2 system
    
    # Avoid division by zero
    if abs(denominator) < 1e-10:
        print(f"  Warning: Singular system - determinant is near zero")
        A_nan = np.full_like(frame, np.nan, dtype=float)
        c_nan = np.full_like(frame, np.nan, dtype=float)
        return A_nan, c_nan
    
    # Solve for amplitude A at each voxel
    A = (n * Σgf - Σg * Σf) / denominator

    # Solve for background c at each voxel
    c = (Σf - Σg * A) / n
    
    return A, c

def fit_gaussian_3d_matlab(window, initial_params, sigma_fixed, k_level, fit_mode, matlab_engine = None, debug=False):
    """
    Wrapper for MATLAB fitGaussian3D MEX function.
    
    Calls MATLAB's fitGaussian3D for precise Gaussian fitting with proper
    uncertainty estimation. Falls back to Python if MATLAB unavailable.
    
    Parameters
    ----------
    window : ndarray
        3D data window to fit (z, y, x order in Python)
    initial_params : list
        [x0, y0, z0, amplitude, sigma, background] - initial guesses
        Note: sigma in initial_params is ignored; sigma_fixed is used
    sigma_fixed : list
        [sigma_xy, sigma_z] - fixed PSF sigma values
    fit_mode : str
        'xyzAc' - fit position, amplitude, background (recommended by Aguet)
        'Ac' - fit only amplitude and background (position fixed)
    matlab_engine : matlab.engine, optional
        MATLAB engine instance. If None, uses global engine.
    k_level : float
        Z-score for significance threshold (default 2.807 for α=0.005)
    debug : bool
        If True, print debugging information
        
    Returns
    -------
    dict or None
        Dictionary with fitted parameters:
        - 'x', 'y', 'z': fitted position (or initial if mode='Ac')
        - 'A': fitted amplitude
        - 'c': fitted background
        - 'A_pstd': uncertainty on amplitude
        - 'c_pstd': uncertainty on background
        - 'sigma_r': residual standard deviation
        - 'SE_sigma_r': standard error of sigma_r
        - 'pval_Ar': p-value for amplitude significance test
        - 'hval_AD': Anderson-Darling test statistic
        - 'npx': number of pixels used in fit
        Returns None if fitting fails.
        
    Notes
    -----
    - MATLAB fitGaussian3D expects data in (y, x, z) order
    - Python window is (z, y, x), so we transpose with (1, 2, 0)
    - Uses one-sided t-test for amplitude significance (Aguet Eq. 2)
    """
    
    engine = matlab_engine if matlab_engine is not None else eng
    
    if engine is None:
        if debug: # type: ignore
            print("⚠️  MATLAB engine not available, using Python fallback")
        # return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, 
                                                # fit_mode, k_level, debug)
    
    try:
        # Convert numpy array to MATLAB format
        # CRITICAL: MATLAB expects (y, x, z) order, Python has (z, y, x)
        # Transpose (1, 2, 0) converts [z,y,x] -> [y,x,z]
        window_transposed = np.transpose(window, (1, 2, 0))
        window_clean = np.ascontiguousarray(window_transposed, dtype=np.float64)
        window_matlab = matlab.double(window_clean.tolist()) # type: ignore

        # Prepare initial parameters for MATLAB
        # MATLAB expects: [x, y, z, A, sigma_xy, sigma_z, c]
        init_matlab = matlab.double([ # type: ignore
            float(initial_params[0]),  # x0
            float(initial_params[1]),  # y0
            float(initial_params[2]),  # z0
            float(initial_params[3]),  # A0
            float(sigma_fixed[0]),     # sigma_xy (fixed)
            float(sigma_fixed[1]),     # sigma_z (fixed)
            float(initial_params[5])   # c0 (background)
        ])

        # Call MATLAB fitGaussian3D
        # Returns: [prm, prmStd, C, res]
        result = engine.fitGaussian3D(window_matlab, init_matlab, fit_mode, nargout=4) # type: ignore

        prm = np.array(result[0]).flatten() # type: ignore
        prmStd = np.array(result[1]).flatten() # type: ignore
        res = result[3] # type: ignore

        # Extract residual statistics from MATLAB dictionary
        try:
            sigma_r = float(res.get('std', np.nan))
        except (TypeError, ValueError):
            sigma_r = np.nan
        
        try:
            hval_AD_raw = res.get('hAD', np.nan)
            hval_AD = float(hval_AD_raw) if isinstance(hval_AD_raw, bool) else float(hval_AD_raw)
        except (TypeError, ValueError):
            hval_AD = np.nan

        # Calculate statistics for significance test
        npx = np.sum(~np.isnan(window.flatten()))
        
        if np.isnan(sigma_r) or sigma_r == 0:
            se_sigma_r = np.nan
            se_r = np.nan
        else:
            # Standard error of variance (Aguet Supplemental)
            se_sigma_r = sigma_r / np.sqrt(2 * (npx - 1))
            se_r = se_sigma_r * k_level
        
        if debug: # type: ignore
            print(f"npx: {npx}, sigma_r: {sigma_r:.4f}, se_sigma_r: {se_sigma_r:.4f}")

        # Extract parameters based on fit_mode
        # MATLAB returns full 7-parameter array for both modes
        if fit_mode == 'xyzAc': # type: ignore
            # Full fit: [x, y, z, A, sigma_xy, sigma_z, c]
            x = prm[0]
            y = prm[1]
            z = prm[2]
            A = prm[3]
            c = prm[6]
            A_pstd = prmStd[3]
            c_pstd = prmStd[4]
            
        else:  # fit_mode == 'Ac'
            # Amplitude-only fit: positions fixed, but MATLAB still returns full array
            A = prm[3]
            c = prm[6]
            x = initial_params[0]
            y = initial_params[1]
            z = initial_params[2]
            A_pstd = prmStd[0]
            c_pstd = prmStd[1]

        # Calculate p-value for amplitude significance (Aguet Eq. 2)
        # One-sided t-test: H0: A <= k*sigma_r
        if not np.isnan(se_r) and not np.isnan(A_pstd):
            df2 = (npx - 1) * (A_pstd**2 + se_r**2)**2 / (A_pstd**4 + se_r**4)
            scomb = np.sqrt((A_pstd**2 + se_r**2) / npx)
            T = (A - sigma_r * k_level) / scomb
            pval_Ar = t_dist.cdf(-T, df2) # type: ignore
        else:
            pval_Ar = np.nan

        if debug: # type: ignore
            print(f"✓ MATLAB fitting ({fit_mode}): A={A:.2f}, c={c:.2f}, pval={pval_Ar:.4f}") # type: ignore

        return {
            'x': x, 'y': y, 'z': z,
            'A': A, 'c': c,
            'sigma_xy': sigma_fixed[0],
            'sigma_z': sigma_fixed[1],
            'A_pstd': A_pstd,
            'c_pstd': c_pstd,
            'sigma_r': sigma_r,
            'SE_sigma_r': se_sigma_r,
            'pval_Ar': pval_Ar,
            'hval_AD': hval_AD,
            'npx': npx
        }

    except Exception as e:
        if debug: # type: ignore
            print(f"✗ MATLAB fitGaussian3D failed: {e}")

#### Part 4: Search for spots in secondary channels around the tracking channel; classify tracks as positive/negative for secondary channel proteins e.g. ARPC3 +/- ####
def find_colocalized_detections(track_df, detection_df, channel_num, tracking_channel,
                                 search_radius_xy=2, search_radius_z=2,
                                 verbose=True):
    """
    Find detections from a secondary channel near tracking channel positions.
    
    For each spot in the tracking channel, searches for detections in the
    secondary channel within a cubic region. Uses Chebyshev distance (L∞ norm)
    to define exact voxel boundaries.
    
    Search region: ±search_radius_xy voxels in x/y, ±search_radius_z voxels in z
    Example: search_radius_xy=2, search_radius_z=2 creates a 5×5×5 voxel cube.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame with columns: frame, x, y, z, track_id, spot_type, ...
        Can include both 'track' and 'buffer' spot_types.
    detection_df : pd.DataFrame
        Detection DataFrame for secondary channel with columns: x, y, z, A, c, ...
        Index: frame (0-based)
    channel_num : int
        Channel number (1, 2, or 3) for labeling output columns
    search_radius_xy : int
        Search radius in x/y dimensions (voxels). Default 2.
    search_radius_z : int
        Search radius in z dimension (voxels). Default 2.
    verbose : bool
        If True, show progress bar. Default True.
        
    Returns
    -------
    pd.DataFrame
        Copy of track_df with additional columns for this channel:
        - 'ch{N}_detected': bool, True if detection found within search radius
        - 'ch{N}_x', 'ch{N}_y', 'ch{N}_z': coordinates of closest detection
        - 'ch{N}_A', 'ch{N}_c': amplitude and background of closest detection
        - 'ch{N}_A_pstd', 'ch{N}_c_pstd': uncertainties
        - 'ch{N}_pval_Ar': p-value for amplitude significance
        - 'ch{N}_distance': Euclidean distance to closest detection
        - 'ch{N}_multi_detect': bool, True if multiple detections equidistant
        
    Notes
    -----
    - Uses anisotropic search radius: different for xy vs z
    - For anisotropic search, coordinates are scaled before KD-tree query
    - When multiple detections are equidistant, closest is chosen arbitrarily
      and ch{N}_multi_detect is set to True
    - Only processes 'track' spot_type rows; 'buffer' rows get NaN
    """
    
    # Create copy to avoid modifying original
    result_df = track_df.copy()
    
    # Column prefix for this channel
    prefix = f'ch{channel_num}'
    
    # Initialize output columns
    result_df[f'{prefix}_detected'] = False
    result_df[f'{prefix}_x'] = np.nan
    result_df[f'{prefix}_y'] = np.nan
    result_df[f'{prefix}_z'] = np.nan
    result_df[f'{prefix}_A'] = np.nan
    result_df[f'{prefix}_A_pstd'] = np.nan
    result_df[f'{prefix}_sigma_x'] = np.nan
    result_df[f'{prefix}_sigma_y'] = np.nan
    result_df[f'{prefix}_sigma_z'] = np.nan
    result_df[f'{prefix}_c'] = np.nan
    result_df[f'{prefix}_c_pstd'] = np.nan
    result_df[f'{prefix}_sigma_r'] = np.nan
    result_df[f'{prefix}_SE_sigma_r'] = np.nan
    result_df[f'{prefix}_pval_Ar'] = np.nan
    result_df[f'{prefix}_hval_AD'] = np.nan
    result_df[f'{prefix}_distance'] = np.nan
    result_df[f'{prefix}_multi_detect'] = False
    
    # Only process 'track' spots, not 'buffer' spots
    if 'spot_type' in result_df.columns:
        track_mask = result_df['spot_type'] == 'track'
    else:
        track_mask = pd.Series(True, index=result_df.index)
    
    # Get frames present in detection data
    detection_df_reset = detection_df.reset_index()
    detection_frames = set(detection_df_reset['frame'].unique())
    
    # Group track data by frame
    track_spots = result_df[track_mask]
    track_grouped = track_spots.groupby('frame')
    
    # Group detection data by frame
    detection_grouped = detection_df_reset.groupby('frame')
    
    # Calculate scale factors for anisotropic search
    # We scale coordinates so that Chebyshev distance works correctly
    # Scale z so that search_radius_z maps to search_radius_xy
    if search_radius_z != search_radius_xy and search_radius_z > 0:
        z_scale = search_radius_xy / search_radius_z
    else:
        z_scale = 1.0
    
    # Use the xy radius for the scaled KD-tree query
    search_radius = search_radius_xy
    
    # Process each frame
    frames_to_process = sorted(track_grouped.groups.keys())
    
    for frame_idx in tqdm(frames_to_process, desc=f"Colocalization ch{channel_num}", 
                          disable=not verbose):
        
        # Skip if no detections in this frame
        if frame_idx not in detection_frames:
            continue
        
        # Get track spots for this frame
        track_frame_df = track_grouped.get_group(frame_idx)
        
        # Get detections for this frame
        detection_frame_df = detection_grouped.get_group(frame_idx)
        
        # Extract coordinates
        cols_channel = [f'ch{tracking_channel}_x', f'ch{tracking_channel}_y', f'ch{tracking_channel}_z']
        cols_fallback = ['x', 'y', 'z']
        cols = cols_channel if all(c in track_frame_df.columns for c in cols_channel) else cols_fallback
        track_coords = track_frame_df[cols].values.copy()
        detection_coords = detection_frame_df[['x', 'y', 'z']].values.copy()
        
        # Scale z coordinates for anisotropic search
        track_coords_scaled = track_coords.copy()
        track_coords_scaled[:, 2] *= z_scale
        
        detection_coords_scaled = detection_coords.copy()
        detection_coords_scaled[:, 2] *= z_scale
        
        # Build KD-tree for detections (scaled coordinates)
        detection_tree = cKDTree(detection_coords_scaled)
        
        # Query for all detections within search radius (Chebyshev distance)
        neighbors_lists = detection_tree.query_ball_point(
            track_coords_scaled,
            r=search_radius,
            p=np.inf  # Chebyshev distance (L∞ norm) for cubic region
        )
        
        # Process each track spot
        for i, (track_idx, neighbor_indices) in enumerate(
            zip(track_frame_df.index, neighbors_lists)
        ):
            # No detections found within search radius
            if len(neighbor_indices) == 0:
                continue
            
            # Get original (unscaled) coordinates for distance calculation
            track_coord = track_coords[i]
            neighbor_coords = detection_coords[neighbor_indices]
            
            # Calculate Euclidean distances (unscaled, true distances)
            distances = np.sqrt(np.sum((neighbor_coords - track_coord)**2, axis=1))
            
            # Find minimum distance
            min_distance = np.min(distances)
            
            # Find all detections at minimum distance (handles ties)
            closest_mask = np.abs(distances - min_distance) < 1e-10
            closest_indices = np.where(closest_mask)[0]
            
            # Check for multiple equidistant detections
            multi_detect = len(closest_indices) > 1
            
            # Select the first closest detection (arbitrary choice for ties)
            closest_local_idx = closest_indices[0]
            closest_global_idx = neighbor_indices[closest_local_idx]
            
            # Get detection row
            closest_detection = detection_frame_df.iloc[closest_global_idx]
            
            # Update result DataFrame
            result_df.loc[track_idx, f'{prefix}_detected'] = True
            result_df.loc[track_idx, f'{prefix}_x'] = closest_detection['x']
            result_df.loc[track_idx, f'{prefix}_y'] = closest_detection['y']
            result_df.loc[track_idx, f'{prefix}_z'] = closest_detection['z']
            result_df.loc[track_idx, f'{prefix}_A'] = closest_detection.get('A', np.nan)
            result_df.loc[track_idx, f'{prefix}_A_pstd'] = closest_detection.get('A_pstd', np.nan)
            result_df.loc[track_idx, f'{prefix}_sigma_x'] = closest_detection.get('sigma_x', np.nan)
            result_df.loc[track_idx, f'{prefix}_sigma_y'] = closest_detection.get('sigma_y', np.nan)
            result_df.loc[track_idx, f'{prefix}_sigma_z'] = closest_detection.get('sigma_z', np.nan)
            result_df.loc[track_idx, f'{prefix}_c'] = closest_detection.get('c', np.nan)
            result_df.loc[track_idx, f'{prefix}_c_pstd'] = closest_detection.get('c_pstd', np.nan)
            result_df.loc[track_idx, f'{prefix}_sigma_r'] = closest_detection.get('sigma_r', np.nan)
            result_df.loc[track_idx, f'{prefix}_SE_sigma_r'] = closest_detection.get('SE_sigma_r', np.nan)
            result_df.loc[track_idx, f'{prefix}_pval_Ar'] = closest_detection.get('pval_Ar', np.nan)
            result_df.loc[track_idx, f'{prefix}_hval_AD'] = closest_detection.get('hval_AD', np.nan)
            result_df.loc[track_idx, f'{prefix}_distance'] = min_distance
            result_df.loc[track_idx, f'{prefix}_multi_detect'] = multi_detect
    
    return result_df


def classify_tracks_by_colocalization(track_df, channel_num, min_consecutive_frames,
                                       second_half_only=True, verbose=True):
    """
    Classify tracks as channel-positive based on consecutive colocalized frames.
    
    A track is classified as positive for a channel if it has at least
    `min_consecutive_frames` consecutive frames with a detection in that channel.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame with colocalization columns from find_colocalized_detections.
        Must have 'ch{N}_detected' column for the specified channel.
    channel_num : int
        Channel number (1, 2, or 3) to classify
    min_consecutive_frames : int
        Minimum consecutive frames with detection required for positive classification
    second_half_only : bool
        If True, only count consecutive frames in the second half of each track.
        Default True (matches biological expectation for recruitment timing).
    verbose : bool
        If True, print summary statistics. Default True.
        
    Returns
    -------
    pd.DataFrame
        Copy of track_df with additional column:
        - 'ch{N}_positive_track': bool, True if track meets consecutive frame threshold
        
    Notes
    -----
    - Only processes rows with spot_type='track' (ignores buffer frames)
    - second_half_only=True is biologically motivated: for endocytosis,
      DNM2/ARPC3 recruitment typically occurs in the second half of pit lifetime
    """
    
    result_df = track_df.copy()
    prefix = f'ch{channel_num}'
    
    # Initialize track-level classification column
    result_df[f'{prefix}_positive_track'] = False
    
    # Only process track spots
    if 'spot_type' in result_df.columns:
        track_spots = result_df[result_df['spot_type'] == 'track']
    else:
        track_spots = result_df
    
    # Get unique track IDs
    track_ids = track_spots['track_id'].unique()
    
    positive_track_ids = []
    
    for track_id in track_ids:
        # Get track data sorted by frame
        track_data = track_spots[track_spots['track_id'] == track_id].sort_values('frame')
        track_length = len(track_data)
        
        # Get detection status array
        detected = track_data[f'{prefix}_detected'].values
        
        # If second_half_only, only consider second half of track
        if second_half_only:
            midpoint = track_length // 2
            detected = detected[midpoint:]
        
        # Find maximum consecutive run of True values
        max_consecutive = 0
        current_consecutive = 0
        
        for is_detected in detected:
            if is_detected:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # Check if track meets threshold
        if max_consecutive >= min_consecutive_frames:
            positive_track_ids.append(track_id)
    
    # Update classification column for positive tracks
    result_df.loc[result_df['track_id'].isin(positive_track_ids), 
                  f'{prefix}_positive_track'] = True
    
    # Print summary
    if verbose:
        n_total = len(track_ids)
        n_positive = len(positive_track_ids)
        pct = 100 * n_positive / n_total if n_total > 0 else 0
        half_str = "second half" if second_half_only else "full track"
        print(f"Channel {channel_num} track classification ({half_str}, "
              f"min_consecutive={min_consecutive_frames}):")
        print(f"  Positive: {n_positive:6d} ({pct:.1f}%)")
        print(f"  Negative: {n_total - n_positive:6d} ({100-pct:.1f}%)")
        print(f"  Total:    {n_total:6d}")
    
    return result_df


def run_colocalization_analysis(track_df, detection_dfs, channel_config,
                                 tracking_channel, search_radius_xy=2,
                                 search_radius_z=2, second_half_only=True,
                                 verbose=True):
    """
    Run colocalization analysis for all enabled secondary channels.
    
    Performs spatial search and track classification for each enabled channel
    that is not the tracking channel. Handles both 2-channel and 3-channel
    experiments based on channel_config enabled flags.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame from tracking channel with buffer classification.
        Should contain columns: frame, x, y, z, track_id, spot_type, ...
    detection_dfs : dict
        Dictionary mapping channel numbers to detection DataFrames.
        Example: {1: df_ch1, 2: df_ch2, 3: df_ch3}
    channel_config : dict
        Channel configuration dictionary with keys:
        - 'enabled': bool
        - 'sigma_xy', 'sigma_z': float
        - 'min_consecutive': int (for colocalization classification)
        Example: {
            1: {'enabled': True, 'min_consecutive': 2, ...},
            2: {'enabled': True, 'min_consecutive': 3, ...},
            3: {'enabled': True, ...}  # tracking channel
        }
    tracking_channel : int
        Channel number used for tracking (will be skipped in colocalization)
    search_radius_xy : int
        Search radius in x/y dimensions (voxels). Default 2.
    search_radius_z : int
        Search radius in z dimension (voxels). Default 2.
    second_half_only : bool
        If True, only count consecutive frames in second half for classification.
        Default True.
    verbose : bool
        If True, print progress and summary statistics. Default True.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'track_df_colocalized': DataFrame with all colocalization columns
        - 'summary': dict with classification counts per channel
        
    Notes
    -----
    - Skips disabled channels (channel_config[N]['enabled'] = False)
    - Skips tracking channel (no self-colocalization)
    - Each channel adds columns: ch{N}_detected, ch{N}_x/y/z, ch{N}_A, etc.
    - Track classification uses channel-specific min_consecutive from config
    """
    
    if verbose:
        print("=" * 70)
        print("COLOCALIZATION ANALYSIS")
        print("=" * 70)
        print(f"Tracking channel: {tracking_channel}")
        print(f"Search radius: xy=±{search_radius_xy}, z=±{search_radius_z} voxels")
        print(f"Second half only: {second_half_only}")
        print("=" * 70)
    
    result_df = track_df.copy()
    summary = {}
    
    # Process each channel
    for channel_num, config in channel_config.items():
        # Skip disabled channels
        if not config.get('enabled', False):
            if verbose:
                print(f"\nChannel {channel_num}: Skipped (disabled)")
            continue
        
        # Skip tracking channel
        if channel_num == tracking_channel:
            if verbose:
                print(f"\nChannel {channel_num}: Skipped (tracking channel)")
            continue
        
        # Check if detection DataFrame exists
        if channel_num not in detection_dfs or detection_dfs[channel_num] is None:
            if verbose:
                print(f"\nChannel {channel_num}: Skipped (no detection data)")
            continue
        
        if verbose:
            print(f"\nProcessing channel {channel_num}...")
        
        # Get detection DataFrame
        detection_df = detection_dfs[channel_num]
        
        # Step 1: Find colocalized detections
        result_df = find_colocalized_detections(
            result_df, detection_df, channel_num, tracking_channel,
            search_radius_xy=search_radius_xy,
            search_radius_z=search_radius_z,
            verbose=verbose
        )
        
        # Step 2: Classify tracks
        min_consecutive = config.get('min_consecutive', 2)
        result_df = classify_tracks_by_colocalization(
            result_df, channel_num, min_consecutive,
            second_half_only=second_half_only,
            verbose=verbose
        )
        
        # Collect summary statistics
        if 'spot_type' in result_df.columns:
            track_spots = result_df[result_df['spot_type'] == 'track']
        else:
            track_spots = result_df
        
        n_tracks = track_spots['track_id'].nunique()
        n_positive = track_spots[track_spots[f'ch{channel_num}_positive_track']]['track_id'].nunique()
        
        summary[channel_num] = {
            'total_tracks': n_tracks,
            'positive_tracks': n_positive,
            'negative_tracks': n_tracks - n_positive,
            'positive_pct': 100 * n_positive / n_tracks if n_tracks > 0 else 0
        }
    
    # Print overall summary
    if verbose:
        print("\n" + "=" * 70)
        print("COLOCALIZATION SUMMARY")
        print("=" * 70)
        for ch_num, stats in summary.items():
            print(f"Channel {ch_num}: {stats['positive_tracks']}/{stats['total_tracks']} "
                  f"positive ({stats['positive_pct']:.1f}%)")
        print("=" * 70 + "\n")
    
    return {
        'track_df_colocalized': result_df,
        'summary': summary
    }


def create_track_classification_summary(track_df, channel_config, tracking_channel):
    """
    Create summary DataFrame of track classifications across all channels.
    
    Generates a per-track summary showing colocalization status for each
    enabled secondary channel, useful for downstream analysis and filtering.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        Track DataFrame with colocalization columns from run_colocalization_analysis.
    channel_config : dict
        Channel configuration dictionary.
    tracking_channel : int
        Tracking channel number.
        
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with one row per track_id and columns:
        - 'track_id': unique track identifier
        - 'track_length': number of frames in track
        - 'ch{N}_positive': bool for each enabled secondary channel
        - 'classification': string label (e.g., 'ch2+_ch1+', 'ch2+_ch1-', 'ch2-')
        
    Notes
    -----
    - Only includes tracks with spot_type='track' (excludes buffer frames)
    - Classification string shows positive channels in ascending order
    """
    
    # Only process track spots
    if 'spot_type' in track_df.columns:
        track_spots = track_df[track_df['spot_type'] == 'track']
    else:
        track_spots = track_df
    
    # Get secondary channels (enabled and not tracking channel)
    secondary_channels = [
        ch for ch, cfg in channel_config.items()
        if cfg.get('enabled', False) and ch != tracking_channel
    ]
    
    # Build summary per track
    summary_rows = []
    
    for track_id in track_spots['track_id'].unique():
        track_data = track_spots[track_spots['track_id'] == track_id]
        
        row = {
            'track_id': track_id,
            'track_length': len(track_data)
        }
        
        # Get classification for each secondary channel
        classification_parts = []
        for ch_num in sorted(secondary_channels):
            col_name = f'ch{ch_num}_positive_track'
            if col_name in track_data.columns:
                is_positive = track_data[col_name].iloc[0]
                row[f'ch{ch_num}_positive'] = is_positive
                sign = '+' if is_positive else '-'
                classification_parts.append(f'ch{ch_num}{sign}')
        
        row['classification'] = '_'.join(classification_parts)
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)

#### Part 5: Intensity fitting in secondary channels ####
def fit_intensity_secondary_channels(track_df, zarr_data, channel_config,
                                      tracking_channel, k_level,
                                      fit_mode='xyzAc', verbose=True):
    """
    Fit Gaussian intensity in secondary channels for frames without detections.
    
    For each track frame where a secondary channel detection was NOT found
    during colocalization (ch{N}_detected == False), performs Gaussian fitting
    at the tracking channel position. For buffer frames, uses the first/last
    track position.
    
    Optimized to batch processing by frame, loading each frame's data only once.
    
    Parameters
    ----------
    track_df : pd.DataFrame
        DataFrame with colocalization results. Must have columns:
        - 'frame', 'track_id', 'spot_type', 'buffer_type'
        - 'ch{tracking_channel}_x/y/z': tracking channel positions
        - 'ch{N}_detected': bool for each secondary channel
        - 'ch{N}_x/y/z', 'ch{N}_A', etc.: colocalization results
    zarr_data : zarr.Array
        Movie data with shape (T, C, Z, Y, X)
    channel_config : dict
        Channel configuration with sigma values per channel.
        Example: {1: {'enabled': True, 'sigma_xy': 1.8, 'sigma_z': 2.3}, ...}
    tracking_channel : int
        Channel number used for tracking (1-based)
    k_level : float
        Z-score for significance threshold (e.g., 2.807 for α=0.005)
    fit_mode : str
        Fitting mode: 'xyzAc' for position refinement, 'Ac' for fixed position.
        Default 'xyzAc' (recommended by Aguet).
    verbose : bool
        If True, print progress and summary statistics. Default True.
        
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with fitted values populated for previously
        undetected frames. Columns ch{N}_A, ch{N}_c, ch{N}_pval_Ar, etc.
        are filled in for frames where ch{N}_detected remains False.
        
    Notes
    -----
    - Uses channel-specific sigma from channel_config
    - Skips disabled channels and tracking channel
    - For 'track' spots: uses tracking channel position from same row
    - For 'buffer' spots: uses first (start buffer) or last (end buffer) 
      track position
    - Position refinement uses same bounds as buffer fitting
    - Optimized: batches by frame to minimize zarr reads
    """
    
    result_df = track_df.copy()
    
    # Get tracking channel column prefix
    track_prefix = f'ch{tracking_channel}'
    
    if verbose:
        print("=" * 70)
        print("INTENSITY FITTING IN SECONDARY CHANNELS")
        print("=" * 70)
        print(f"Tracking channel: {tracking_channel}")
        print(f"Fit mode: {fit_mode}")
        print(f"K-level: {k_level}")
        print("=" * 70)
    
    # Pre-compute first/last track positions for buffer frames (all channels need this)
    track_positions = {}
    for track_id in result_df['track_id'].unique():
        track_data = result_df[
            (result_df['track_id'] == track_id) & 
            (result_df['spot_type'] == 'track')
        ].sort_values('frame')
        
        if len(track_data) > 0:
            first_row = track_data.iloc[0]
            last_row = track_data.iloc[-1]
            track_positions[track_id] = {
                'first': (first_row[f'{track_prefix}_x'], 
                          first_row[f'{track_prefix}_y'], 
                          first_row[f'{track_prefix}_z']),
                'last': (last_row[f'{track_prefix}_x'], 
                         last_row[f'{track_prefix}_y'], 
                         last_row[f'{track_prefix}_z'])
            }
    
    # Process each secondary channel
    for channel_num, config in channel_config.items():
        # Skip disabled channels
        if not config.get('enabled', False):
            if verbose:
                print(f"\nChannel {channel_num}: Skipped (disabled)")
            continue
        
        # Skip tracking channel
        if channel_num == tracking_channel:
            continue
        
        prefix = f'ch{channel_num}'
        
        # Check if colocalization columns exist
        if f'{prefix}_detected' not in result_df.columns:
            if verbose:
                print(f"\nChannel {channel_num}: Skipped (no colocalization data)")
            continue
        
        # Get channel-specific sigma
        sigma_xy = config['sigma_xy']
        sigma_z = config['sigma_z']
        sigma = [sigma_xy, sigma_z]
        
        # Channel index is 0-based
        channel_idx = channel_num - 1
        
        if verbose:
            print(f"\nProcessing channel {channel_num}...")
            print(f"  Sigma: [σ_xy={sigma_xy}, σ_z={sigma_z}]")
        
        # Find frames needing fitting (not detected during colocalization)
        needs_fitting_mask = result_df[f'{prefix}_detected'] == False
        n_to_fit = needs_fitting_mask.sum()
        
        if verbose:
            print(f"  Frames to fit: {n_to_fit}")
        
        if n_to_fit == 0:
            continue
        
        # Group indices by frame for batch processing
        indices_to_fit = result_df[needs_fitting_mask].index
        frame_groups = result_df.loc[indices_to_fit].groupby('frame').groups
        n_frames_to_process = len(frame_groups)
        
        if verbose:
            print(f"  Unique frames: {n_frames_to_process}")
        
        n_fitted = 0
        n_failed = 0
        
        # Process frame by frame (load each frame only once)
        for frame_idx, row_indices in tqdm(frame_groups.items(), 
                                            desc=f"Fitting ch{channel_num}",
                                            disable=not verbose):
            # Load frame data ONCE for all spots in this frame
            frame_data = zarr_data[int(frame_idx), channel_idx]
            
            # Process all spots in this frame
            for idx in row_indices:
                row = result_df.loc[idx]
                track_id = row['track_id']
                spot_type = row['spot_type']
                buffer_type = row.get('buffer_type', np.nan)
                
                # Determine position to use for fitting
                if spot_type == 'track':
                    # Use tracking channel position from same row
                    x = row[f'{track_prefix}_x']
                    y = row[f'{track_prefix}_y']
                    z = row[f'{track_prefix}_z']
                elif spot_type == 'buffer':
                    # Use first or last track position
                    if track_id not in track_positions:
                        n_failed += 1
                        continue
                    if buffer_type == 'start':
                        x, y, z = track_positions[track_id]['first']
                    else:  # 'end'
                        x, y, z = track_positions[track_id]['last']
                else:
                    n_failed += 1
                    continue
                
                # Fit Gaussian at tracking channel position
                fit_result = fit_buffer_frame(x, y, z, frame_data, sigma, k_level,
                                              fit_mode, debug=False)
                
                if fit_result is not None:
                    # Populate DataFrame columns with fit results
                    result_df.loc[idx, f'{prefix}_x'] = fit_result['x']
                    result_df.loc[idx, f'{prefix}_y'] = fit_result['y']
                    result_df.loc[idx, f'{prefix}_z'] = fit_result['z']
                    result_df.loc[idx, f'{prefix}_A'] = fit_result['A']
                    result_df.loc[idx, f'{prefix}_c'] = fit_result['c']
                    result_df.loc[idx, f'{prefix}_sigma_x'] = sigma_xy
                    result_df.loc[idx, f'{prefix}_sigma_y'] = sigma_xy
                    result_df.loc[idx, f'{prefix}_sigma_z'] = sigma_z
                    result_df.loc[idx, f'{prefix}_A_pstd'] = fit_result['A_pstd']
                    result_df.loc[idx, f'{prefix}_c_pstd'] = fit_result['c_pstd']
                    result_df.loc[idx, f'{prefix}_sigma_r'] = fit_result.get('sigma_r', np.nan)
                    result_df.loc[idx, f'{prefix}_SE_sigma_r'] = fit_result.get('SE_sigma_r', np.nan)
                    result_df.loc[idx, f'{prefix}_pval_Ar'] = fit_result['pval_Ar']
                    result_df.loc[idx, f'{prefix}_hval_AD'] = fit_result.get('hval_AD', np.nan)
                    # Note: ch{N}_detected remains False (fitted, not detected)
                    n_fitted += 1
                else:
                    n_failed += 1
        
        if verbose:
            print(f"  Fitted: {n_fitted}, Failed: {n_failed}")
    
    if verbose:
        print("\n" + "=" * 70)
        print("INTENSITY FITTING COMPLETE")
        print("=" * 70 + "\n")
    
    return result_df

#### Part 6: Compress track dataframes and classify as apical, lateral, and basal ####
class Track:
    """
    A class to represent a track with multi-channel detection and fitting results.
    
    Each Track compresses to a single DataFrame row with Series for per-frame data.
    Channel-agnostic: uses ch1, ch2, ch3 naming convention.
    
    Attributes
    ----------
    track_id : int
        Unique identifier for the track
    track_start, track_end : int
        Starting and ending frames (track spots only, excludes buffers)
    track_length : int
        Number of frames in the track (excludes buffer frames)
    mean_z : float
        Mean z-coordinate of track (from tracking channel, track spots only)
    spot_type : pd.Series
        'track' or 'buffer' for each frame
    
    Per-channel attributes (for each enabled channel N):
    - ch{N}_x, ch{N}_y, ch{N}_z : pd.Series - coordinates
    - ch{N}_A, ch{N}_c : pd.Series - amplitude and background
    - ch{N}_max_A : float - maximum amplitude (track spots only; detected frames only for +ve secondary tracks)
    - ch{N}_detected : pd.Series - detection status (secondary channels only)
    - ch{N}_positive_track : bool - track-level classification (secondary channels only)
    """
    
    def __init__(self, track_id, track_data, tracking_channel, channel_config):
        """
        Initialize Track from a DataFrame subset for one track.
        
        Parameters
        ----------
        track_id : int
            Unique track identifier
        track_data : pd.DataFrame
            DataFrame rows for this track (already filtered by track_id)
        tracking_channel : int
            Channel number used for tracking (1-based)
        channel_config : dict
            Channel configuration with 'enabled' flags
        """
        # Sort by frame
        track_data = track_data.sort_values('frame').reset_index(drop=True)
        
        # Basic track attributes
        self.track_id = track_id
        
        # Spot type (track or buffer)
        if 'spot_type' in track_data.columns:
            self.spot_type = track_data['spot_type']
            # Track length/start/end excludes buffer frames
            track_only = track_data[track_data['spot_type'] == 'track']
            self.track_length = len(track_only)
            self.track_start = int(track_only['frame'].min())
            self.track_end = int(track_only['frame'].max())
        else:
            self.spot_type = pd.Series(['track'] * len(track_data))
            track_only = track_data
            self.track_length = len(track_data)
            self.track_start = int(track_data['frame'].min())
            self.track_end = int(track_data['frame'].max())
        
        # Store tracking channel reference
        self.tracking_channel = tracking_channel
        
        # Extract per-channel data
        for channel_num, config in channel_config.items():
            if not config.get('enabled', False):
                continue
            
            prefix = f'ch{channel_num}'
            
            # Core columns (coordinates, amplitude, background) - Series
            for col in ['x', 'y', 'z', 'A', 'c']:
                full_col = f'{prefix}_{col}'
                if full_col in track_data.columns:
                    setattr(self, f'{prefix}_{col}', track_data[full_col])
            
            # Secondary channel specific columns
            if channel_num != tracking_channel:
                # detected - Series
                detected_col = f'{prefix}_detected'
                if detected_col in track_data.columns:
                    setattr(self, f'{prefix}_detected', track_data[detected_col])
                
                # positive_track - scalar (track-level)
                positive_col = f'{prefix}_positive_track'
                if positive_col in track_data.columns:
                    setattr(self, f'{prefix}_positive_track', track_data[positive_col].iloc[0])
            
            # Compute max_A for this channel (track spots only)
            setattr(self, f'{prefix}_max_A', self._compute_max_A(channel_num, track_data))
        
        # Compute mean_z from tracking channel (track spots only)
        z_col = f'ch{tracking_channel}_z'
        if z_col in track_only.columns:
            self.mean_z = track_only[z_col].mean()
        else:
            self.mean_z = np.nan
    
    def _compute_max_A(self, channel_num, track_data):
        """
        Compute maximum amplitude for specified channel (track spots only).
        
        For tracking channel: max of all amplitudes.
        For secondary channels:
          - If positive_track: max among detected frames only
          - If negative_track: max among all fitted frames
        """
        prefix = f'ch{channel_num}'
        A_col = f'{prefix}_A'
        
        if A_col not in track_data.columns:
            return np.nan
        
        # Only consider track spots, not buffers
        if 'spot_type' in track_data.columns:
            track_only = track_data[track_data['spot_type'] == 'track']
        else:
            track_only = track_data
        
        if len(track_only) == 0:
            return np.nan
        
        # Tracking channel: all spots were detected
        if channel_num == self.tracking_channel:
            return track_only[A_col].max()
        
        # Secondary channel
        positive_col = f'{prefix}_positive_track'
        detected_col = f'{prefix}_detected'
        
        is_positive = track_only[positive_col].iloc[0] if positive_col in track_only.columns else False
        
        if is_positive and detected_col in track_only.columns:
            # Max among detected frames only
            detected_mask = track_only[detected_col] == True
            if detected_mask.any():
                return track_only.loc[detected_mask, A_col].max()
        
        # Negative track or no detected frames: max among all fits
        return track_only[A_col].max()


def create_tracks_from_dataframe(df, tracking_channel, channel_config, verbose=True):
    """
    Create Track objects from a DataFrame with multi-channel data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: frame, track_id, spot_type,
        and per-channel columns (ch{N}_x, ch{N}_A, etc.)
    tracking_channel : int
        Channel number used for tracking (1-based)
    channel_config : dict
        Channel configuration with 'enabled' flags
    verbose : bool
        If True, print summary. Default True.
        
    Returns
    -------
    list of Track
        List of Track objects, one per unique track_id
    """
    tracks = []
    track_ids = df['track_id'].unique()
    
    for track_id in track_ids:
        track_data = df[df['track_id'] == track_id]
        track = Track(track_id, track_data, tracking_channel, channel_config)
        tracks.append(track)
    
    if verbose:
        print(f"Created {len(tracks)} Track objects")
        for ch_num, config in channel_config.items():
            if config.get('enabled', False) and ch_num != tracking_channel:
                pos_attr = f'ch{ch_num}_positive_track'
                n_positive = sum(1 for t in tracks if getattr(t, pos_attr, False))
                print(f"  Channel {ch_num} positive: {n_positive}/{len(tracks)}")
    
    return tracks


def tracks_to_dataframe(tracks, tracking_channel, channel_config):
    """
    Convert list of Track objects to a summary DataFrame (one row per track).
    
    Parameters
    ----------
    tracks : list of Track
        List of Track objects
    tracking_channel : int
        Channel number used for tracking (1-based)
    channel_config : dict
        Channel configuration with 'enabled' flags
        
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per track containing:
        - track_id, track_start, track_end, track_length
        - mean_z (from tracking channel, track spots only)
        - spot_type (Series: 'track' or 'buffer')
        - Per-channel: ch{N}_x, ch{N}_y, ch{N}_z, ch{N}_A, ch{N}_c (Series)
        - Per-channel: ch{N}_max_A (scalar, track spots only)
        - Secondary channels: ch{N}_detected (Series), ch{N}_positive_track (bool)
    """
    rows = []
    
    for track in tracks:
        row = {
            'track_id': track.track_id,
            'track_start': track.track_start,
            'track_end': track.track_end,
            'track_length': track.track_length,
            'mean_z': track.mean_z,
            'spot_type': track.spot_type,
        }
        
        # Add per-channel data
        for ch_num, config in channel_config.items():
            if not config.get('enabled', False):
                continue
            
            prefix = f'ch{ch_num}'
            
            # Series: x, y, z, A, c
            for col in ['x', 'y', 'z', 'A', 'c']:
                attr = f'{prefix}_{col}'
                row[attr] = getattr(track, attr, None)
            
            # Scalar: max_A
            row[f'{prefix}_max_A'] = getattr(track, f'{prefix}_max_A', np.nan)
            
            # Secondary channel only
            if ch_num != tracking_channel:
                # Series: detected
                row[f'{prefix}_detected'] = getattr(track, f'{prefix}_detected', None)
                # Scalar: positive_track
                row[f'{prefix}_positive_track'] = getattr(track, f'{prefix}_positive_track', None)
        
        rows.append(row)
    
    return pd.DataFrame(rows)

#### Part 6: Plotting and visualization ####
def create_track_montage(track_id, tracks_df, zarr_data, channel_config, tracking_channel,
                         buffer_frames=1, frame_spacing=1,
                         save_plot=False, output_dir=None, dpi=600):
    """
    Create a montage visualization of a track showing all enabled channels.
    
    Parameters
    ----------
    track_id : int
        Track ID to visualize
    tracks_df : pd.DataFrame
        Compressed tracks DataFrame (one row per track) with Series columns
    zarr_data : zarr.Array
        Movie data with shape (T, C, Z, Y, X)
    channel_config : dict
        Channel configuration with 'enabled' flags and sigma values
    tracking_channel : int
        Channel number used for tracking (1-based)
    buffer_frames : int
        Number of frames to show before/after track (default 5)
    frame_spacing : int
        Pixels of white space between frames (default 2)
    save_plot : bool
        If True, save plot to file (default False)
    output_dir : str or None
        Directory to save plot. If None, uses current directory.
    dpi : int
        Resolution for saved figure (default 300)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    
    # Helper function to add spacing between frames
    def add_spacing(stack_list, spacing, is_rgb=False):
        """Add white spacing between frames. Assumes data is normalized to [0, 1]."""
        if spacing == 0 or len(stack_list) <= 1:
            return np.hstack(stack_list)
        
        result = [stack_list[0]]
        for frame in stack_list[1:]:
            if is_rgb:
                # White spacer for RGB (all channels = 1)
                spacer = np.ones((frame.shape[0], spacing, 3), dtype=np.float32)
            else:
                # White spacer for grayscale (value = 1)
                spacer = np.ones((frame.shape[0], spacing), dtype=np.float32)
            result.extend([spacer, frame])
        return np.hstack(result)
    
    # Helper function to normalize image to [0, 1]
    def normalize_image(img):
        """Normalize image to [0, 1] range."""
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(img, dtype=np.float32)
    
    # Get image dimensions
    n_frames, n_ch, nz, ny, nx = zarr_data.shape
    
    # Get track row
    track_row = tracks_df[tracks_df['track_id'] == track_id]
    if len(track_row) == 0:
        raise ValueError(f"Track ID {track_id} not found in tracks_df")
    track_row = track_row.iloc[0]
    
    # Get track info
    track_start = track_row['track_start']
    track_end = track_row['track_end']
    track_length = track_row['track_length']
    spot_type = track_row['spot_type']
    
    # Get enabled channels
    enabled_channels = sorted([ch for ch, cfg in channel_config.items() if cfg.get('enabled', False)])
    n_channels = len(enabled_channels)
    
    # Get coordinates for each channel
    channel_coords = {}
    for ch in enabled_channels:
        prefix = f'ch{ch}'
        channel_coords[ch] = {
            'x': track_row[f'{prefix}_x'],
            'y': track_row[f'{prefix}_y'],
            'z': track_row[f'{prefix}_z']
        }
    
    # Get crop sizes for each channel based on sigma
    channel_crop_xy = {}
    channel_crop_z = {}
    for ch in enabled_channels:
        sigma_xy = channel_config[ch].get('sigma_xy', 1.5)
        sigma_z = channel_config[ch].get('sigma_z', 2.0)
        channel_crop_xy[ch] = int(np.ceil(2 * sigma_xy))
        channel_crop_z[ch] = int(np.ceil(2 * sigma_z))
    
    # Build time and coordinate lists from spot_type (track frames only)
    time = []
    track_coords = {ch: {'x': [], 'y': [], 'z': []} for ch in enabled_channels}
    
    for i, st in enumerate(spot_type):
        if st == 'track':
            time.append(track_start + len(time))
            for ch in enabled_channels:
                track_coords[ch]['x'].append(channel_coords[ch]['x'].iloc[i])
                track_coords[ch]['y'].append(channel_coords[ch]['y'].iloc[i])
                track_coords[ch]['z'].append(channel_coords[ch]['z'].iloc[i])
    
    print(f'Track {track_id}: frames {min(time)}-{max(time)} ({len(time)} frames)')
    
    # Extend time range by buffer_frames before and after
    extended_time = list(range(
        max(0, min(time) - buffer_frames),
        min(n_frames, max(time) + buffer_frames + 1)
    ))
    
    # Remove frames that are in the gap (between min and max but not in time)
    extended_time = [f for f in extended_time if f < min(time) or f > max(time) or f in time]
    
    print(f'Extended time: {extended_time[0]}-{extended_time[-1]} ({len(extended_time)} frames)')
    
    # Extend coordinates for each channel
    min_frame, max_frame = min(time), max(time)
    extended_coords = {ch: {'x': [], 'y': [], 'z': []} for ch in enabled_channels}
    
    for frame in extended_time:
        for ch in enabled_channels:
            if frame < min_frame:
                extended_coords[ch]['x'].append(track_coords[ch]['x'][0])
                extended_coords[ch]['y'].append(track_coords[ch]['y'][0])
                extended_coords[ch]['z'].append(track_coords[ch]['z'][0])
            elif frame > max_frame:
                extended_coords[ch]['x'].append(track_coords[ch]['x'][-1])
                extended_coords[ch]['y'].append(track_coords[ch]['y'][-1])
                extended_coords[ch]['z'].append(track_coords[ch]['z'][-1])
            else:
                frame_idx = time.index(frame)
                extended_coords[ch]['x'].append(track_coords[ch]['x'][frame_idx])
                extended_coords[ch]['y'].append(track_coords[ch]['y'][frame_idx])
                extended_coords[ch]['z'].append(track_coords[ch]['z'][frame_idx])
    
    # Prepare stacks for each channel (store raw projections first)
    channel_stacks_raw = {ch: [] for ch in enabled_channels}
    merge_stack = []
    
    # Get max crop size for consistent patch dimensions in merge
    max_crop_xy = max(channel_crop_xy.values())
    
    for t_idx, t in enumerate(extended_time):
        channel_projs = {}
        
        for ch in enabled_channels:
            ch_idx = ch - 1
            crop_xy = channel_crop_xy[ch]
            crop_z = channel_crop_z[ch]
            
            x = int(np.round(np.clip(extended_coords[ch]['x'][t_idx], 0, nx - 1)))
            y = int(np.round(np.clip(extended_coords[ch]['y'][t_idx], 0, ny - 1)))
            z = int(np.round(np.clip(extended_coords[ch]['z'][t_idx], 0, nz - 1)))
            
            z_start = max(0, z - crop_z)
            z_end = min(nz, z + crop_z + 1)
            y_start = max(0, y - crop_xy)
            y_end = min(ny, y + crop_xy + 1)
            x_start = max(0, x - crop_xy)
            x_end = min(nx, x + crop_xy + 1)
            
            patch = zarr_data[t, ch_idx, z_start:z_end, y_start:y_end, x_start:x_end]
            proj = np.max(patch, axis=0)
            
            # Pad to consistent size if needed
            expected_size = 2 * crop_xy + 1
            if proj.shape[0] != expected_size or proj.shape[1] != expected_size:
                padded = np.zeros((expected_size, expected_size), dtype=proj.dtype)
                pad_y = max(0, crop_xy - y)
                pad_x = max(0, crop_xy - x)
                padded[pad_y:pad_y + proj.shape[0], pad_x:pad_x + proj.shape[1]] = proj
                proj = padded
            
            channel_stacks_raw[ch].append(proj)
            channel_projs[ch] = proj
        
        # Create RGB merge (normalize each channel for merge)
        if n_channels >= 1:
            merge_size = 2 * max_crop_xy + 1
            rgb = np.zeros((merge_size, merge_size, 3), dtype=np.float32)
            
            for i, ch in enumerate(enabled_channels[:3]):
                proj = channel_projs[ch].astype(np.float32)
                
                if proj.shape[0] != merge_size or proj.shape[1] != merge_size:
                    from skimage.transform import resize
                    proj = resize(proj, (merge_size, merge_size), preserve_range=True)
                
                # Normalize for merge
                proj_norm = normalize_image(proj)
                rgb[:, :, i] = proj_norm
            
            merge_stack.append(rgb)
    
    # Normalize each channel strip globally (across all frames in that channel)
    # This preserves relative intensity changes across time
    channel_stacks = {}
    for ch in enabled_channels:
        raw_stack = channel_stacks_raw[ch]
        # Find global min/max for this channel across all frames
        all_values = np.concatenate([frame.ravel() for frame in raw_stack])
        global_min, global_max = all_values.min(), all_values.max()
        
        # Normalize each frame using global min/max
        normalized_stack = []
        for frame in raw_stack:
            if global_max > global_min:
                frame_norm = (frame.astype(np.float32) - global_min) / (global_max - global_min)
            else:
                frame_norm = np.zeros_like(frame, dtype=np.float32)
            normalized_stack.append(frame_norm)
        
        channel_stacks[ch] = normalized_stack
    
    # Concatenate strips with spacing (now all data is [0, 1] so white = 1.0)
    channel_strips = {ch: add_spacing(channel_stacks[ch], frame_spacing, is_rgb=False) 
                      for ch in enabled_channels}
    merge_strip = add_spacing(merge_stack, frame_spacing, is_rgb=True) if merge_stack else None
    
    # Arrow positions for buffer frames (account for spacing)
    arrow_positions = {}
    for ch in enabled_channels:
        patch_width = 2 * channel_crop_xy[ch] + 1
        arrow_positions[ch] = []
        for i, frame in enumerate(extended_time):
            if frame not in time:
                pos = i * (patch_width + frame_spacing) + patch_width // 2
                arrow_positions[ch].append(pos)
    
    merge_patch_width = 2 * max_crop_xy + 1
    merge_arrow_positions = []
    for i, frame in enumerate(extended_time):
        if frame not in time:
            pos = i * (merge_patch_width + frame_spacing) + merge_patch_width // 2
            merge_arrow_positions.append(pos)
    
    # Create figure
    n_rows = n_channels + (1 if merge_strip is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(len(extended_time) * 0.8, n_rows * 1.2), dpi=dpi)
    
    if n_rows == 1:
        axes = [axes]
    
    # Plot channel strips (data is now [0, 1], so vmin=0, vmax=1)
    for i, ch in enumerate(enabled_channels):
        axes[i].imshow(channel_strips[ch], cmap='gray', vmin=0, vmax=1)
        for pos in arrow_positions[ch]:
            axes[i].annotate('', xy=(pos, -0.5), xytext=(pos, 3),
                           arrowprops=dict(arrowstyle='-|>', color='red', lw=1.5))
        sigma_xy = channel_config[ch].get('sigma_xy', 'N/A')
        sigma_z = channel_config[ch].get('sigma_z', 'N/A')
        axes[i].set_title(f'Channel {ch} (σ_xy={sigma_xy}, σ_z={sigma_z})', 
                         fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Plot merge (RGB data is already [0, 1])
    if merge_strip is not None:
        axes[-1].imshow(np.clip(merge_strip, 0, 1))
        for pos in merge_arrow_positions:
            axes[-1].annotate('', xy=(pos, -0.5), xytext=(pos, 3),
                            arrowprops=dict(arrowstyle='-|>', color='white', lw=1.5))
        axes[-1].set_title('Merge', fontsize=10, fontweight='bold')
        axes[-1].axis('off')
    
    plt.suptitle(f'Track {track_id} (frames {track_start}-{track_end}, length={track_length})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_plot:
        if output_dir is None:
            output_dir = '.'
        filename = os.path.join(output_dir, f'montage_track_{track_id}.png')
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'Saved: {filename}')
    
    return fig