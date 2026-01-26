"""
Secondary Channel Detection for LLSM Data
Fits DNM2 and ARPC3 channels at AP2 track positions using Aguet's two-step approach.
Designed to integrate with notebook 03__extracting_alt_channel_intensities_BD.ipynb
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Import from your existing llsm_buffer_analysis.py
from llsm_buffer_analysis import (
    fit_gaussian_3d_matlab,
    estimate_gaussian_amplitude_3d,
    SIGMA_VALUES,
    K_LEVEL,
    eng  # MATLAB engine
)


def fit_secondary_channel_at_position(x, y, z, frame, labels, sigma_secondary, 
                                      sigma_master_xy, channel_idx, 
                                      timing_dict=None, debug=False):
    """
    Fit secondary channel at a given AP2 position using Aguet's two-step approach.
    
    This implements the exact logic from detectMovies3D.m:
    1. Fit with fixed position ('Ac' mode) 
    2. Fit with position refinement ('xyAc' mode)
    3. Select best based on distance (<3σ_master) and amplitude improvement
    
    Parameters:
    -----------
    x, y, z : float
        AP2 track position (master channel coordinates)
    frame : 3D array
        Secondary channel frame data [z, y, x]
    labels : 3D array or None
        Label mask for excluding nearby objects
    sigma_secondary : float or list
        PSF sigma for secondary channel [sigma_xy, sigma_z]
    sigma_master_xy : float
        Master channel (AP2) sigma_xy for distance threshold
    channel_idx : int
        Channel index for reference
    timing_dict : dict, optional
        Dictionary to accumulate timing statistics
    debug : bool, optional
        If True, print debugging information
        
    Returns:
    --------
    dict with fitted parameters or None if fitting fails
    Contains: 'Ac_fit', 'xyAc_fit', 'best_fit', 'selection_reason', 
              'distance_from_master', 'amplitude_improvement'
    """
    t_start = time.time()
    
    nz, ny, nx = frame.shape
    
    # Ensure sigma is in list form [sigma_xy, sigma_z]
    if np.isscalar(sigma_secondary):
        sigma_list = [sigma_secondary, sigma_secondary]
    else:
        sigma_list = list(sigma_secondary)
    
    # Convert to integer coordinates
    xi = int(np.round(np.clip(x, 0, nx-1)))
    yi = int(np.round(np.clip(y, 0, ny-1)))
    zi = int(np.round(np.clip(z, 0, nz-1)))
    
    # Define window boundaries (±2σ)
    w2x = int(np.ceil(2 * sigma_list[0]))
    w2z = int(np.ceil(2 * sigma_list[1]))
    
    # Extract window
    xa = slice(max(0, xi-w2x), min(nx, xi+w2x+1))
    ya = slice(max(0, yi-w2x), min(ny, yi+w2x+1))
    za = slice(max(0, zi-w2z), min(nz, zi+w2z+1))
    
    window = frame[za, ya, xa].copy()
    
    if timing_dict is not None:
        timing_dict['extract_window'] = timing_dict.get('extract_window', 0) + (time.time() - t_start)
    
    # Mask out other objects if labels provided
    t0 = time.time()
    if labels is not None:
        mask_window = labels[za, ya, xa]
        center_z = min(zi-max(0, zi-w2z), mask_window.shape[0]-1)
        center_y = min(yi-max(0, yi-w2x), mask_window.shape[1]-1)
        center_x = min(xi-max(0, xi-w2x), mask_window.shape[2]-1)
        center_label = mask_window[center_z, center_y, center_x]
        window[np.logical_and(mask_window != 0, mask_window != center_label)] = np.nan
    
    if timing_dict is not None:
        timing_dict['apply_mask'] = timing_dict.get('apply_mask', 0) + (time.time() - t0)
    
    # Relative coordinates in window
    ox = xi - max(0, xi-w2x)
    oy = yi - max(0, yi-w2x)
    oz = zi - max(0, zi-w2z)
    
    # Estimate initial parameters
    t0 = time.time()
    A_est, c_est = estimate_gaussian_amplitude_3d(window, sigma_list[0], window_size=5)
    ai = A_est[oz, oy, ox] if not np.isnan(A_est[oz, oy, ox]) else np.nanmax(window)
    ci = c_est[oz, oy, ox] if not np.isnan(c_est[oz, oy, ox]) else np.nanmin(window)
    
    if timing_dict is not None:
        timing_dict['estimate_params'] = timing_dict.get('estimate_params', 0) + (time.time() - t0)
    
    # Initial parameters for fitting
    initial_params = [ox, oy, oz, ai, sigma_list[0], ci]
    
    # =======================================================================
    # STEP 1: Fixed position fit ('Ac' mode) - matches MATLAB line ~145-147
    # =======================================================================
    t0 = time.time()
    Ac_fit = fit_gaussian_3d_matlab(
        window, 
        initial_params, 
        sigma_list, 
        fit_mode='Ac', 
        debug=debug
    )
    
    if timing_dict is not None:
        timing_dict['fit_Ac'] = timing_dict.get('fit_Ac', 0) + (time.time() - t0)
    
    if Ac_fit is None:
        if debug:
            print(f"   ✗ Fixed position fit ('Ac') failed for channel {channel_idx}")
        return None
    
    # Convert to global coordinates (Ac keeps position fixed)
    Ac_fit['x'] = x
    Ac_fit['y'] = y
    Ac_fit['z'] = z
    
    # =======================================================================
    # STEP 2: Position refinement fit ('xyAc' mode) - matches MATLAB line ~150-152
    # =======================================================================
    # Use Ac_fit amplitude as initial guess for refinement
    initial_params_refined = [ox, oy, oz, Ac_fit['A'], sigma_list[0], Ac_fit['c']]
    
    t0 = time.time()
    xyAc_fit = fit_gaussian_3d_matlab(
        window,
        initial_params_refined,
        sigma_list,
        fit_mode='xyAc',
        debug=debug
    )
    
    if timing_dict is not None:
        timing_dict['fit_xyAc'] = timing_dict.get('fit_xyAc', 0) + (time.time() - t0)
    
    if xyAc_fit is None:
        if debug:
            print(f"   ✗ Position refinement fit ('xyAc') failed, using Ac result")
        return {
            'Ac_fit': Ac_fit,
            'xyAc_fit': None,
            'best_fit': Ac_fit,
            'selection_reason': 'xyAc_failed',
            'distance_from_master': 0.0,
            'amplitude_improvement': 0.0
        }
    
    # Convert refined position to global coordinates
    dx = xyAc_fit['x'] - ox
    dy = xyAc_fit['y'] - oy
    dz = xyAc_fit['z'] - oz
    xyAc_fit['x'] = xi + dx
    xyAc_fit['y'] = yi + dy
    xyAc_fit['z'] = zi + dz
    
    # =======================================================================
    # STEP 3: Select best fit - matches MATLAB line ~153-155
    # =======================================================================
    # Calculate distance from master position (2D in XY plane)
    distance = np.sqrt((xyAc_fit['x'] - x)**2 + (xyAc_fit['y'] - y)**2)
    
    # Aguet threshold: 3σ of MASTER channel
    threshold = 3 * sigma_master_xy
    
    # Amplitude improvement
    amplitude_improvement = xyAc_fit['A'] - Ac_fit['A']
    
    # Selection criteria: BOTH must be true to accept refinement
    # 1. Distance < 3σ_master
    # 2. Refined amplitude > fixed amplitude
    if distance < threshold and xyAc_fit['A'] > Ac_fit['A']:
        best_fit = xyAc_fit
        selection_reason = 'xyAc_selected'
        if debug:
            print(f"   ✓ Position refinement accepted: dist={distance:.2f} < {threshold:.2f}, "
                  f"ΔA={amplitude_improvement:.1f}")
    else:
        best_fit = Ac_fit
        if distance >= threshold:
            selection_reason = 'xyAc_too_far'
            if debug:
                print(f"   ✗ Position refinement rejected (too far): dist={distance:.2f} >= {threshold:.2f}")
        else:
            selection_reason = 'xyAc_worse_amplitude'
            if debug:
                print(f"   ✗ Position refinement rejected (worse amplitude): ΔA={amplitude_improvement:.1f} <= 0")
    
    return {
        'Ac_fit': Ac_fit,
        'xyAc_fit': xyAc_fit,
        'best_fit': best_fit,
        'selection_reason': selection_reason,
        'distance_from_master': distance,
        'amplitude_improvement': amplitude_improvement
    }


def add_xyac_fit_mode_support():
    """
    Check if fit_gaussian_3d_matlab supports 'xyAc' mode.
    If not, print instructions for adding support.
    
    Returns:
    --------
    bool : True if supported, False otherwise
    """
    # Test with dummy data
    test_window = np.random.randn(5, 5, 5) * 10 + 100
    test_params = [2, 2, 2, 100, 1.5, 50]
    test_sigma = [1.5, 2.0]
    
    try:
        result = fit_gaussian_3d_matlab(
            test_window,
            test_params,
            test_sigma,
            fit_mode='xyAc',
            debug=False
        )
        if result is not None:
            print("✓ 'xyAc' fit mode is supported")
            return True
        else:
            print("⚠ 'xyAc' fit mode may not be fully supported - test returned None")
            return False
    except Exception as e:
        print(f"✗ 'xyAc' fit mode not supported: {e}")
        print("\nTo add support, modify fit_gaussian_3d_matlab() in llsm_buffer_analysis.py:")
        print("""
        if fit_mode == 'xyAc':
            # Fit xy position, amplitude, background (z fixed)
            # MATLAB expects: [x, y, A, sigma_xy, sigma_z, c]
            init_matlab = matlab.double([
                float(initial_params[0]),  # x
                float(initial_params[1]),  # y
                float(initial_params[3]),  # A
                float(sigma_fixed[0]),     # sigma_xy
                float(sigma_fixed[1]),     # sigma_z
                float(initial_params[5])   # c
            ])
            
            # Returns: [x, y, A, sigma_xy, sigma_z, c] - 6 values
            x = prm[0]
            y = prm[1]
            z = initial_params[2]  # Keep z fixed
            A = prm[2]
            c = prm[5]
            A_pstd = prmStd[2]
            c_pstd = prmStd[3]
        """)
        return False


def detect_secondary_channels_frame_by_frame(track_df_cleaned, zarr_data,
                                              master_channel_idx=2,
                                              secondary_channel_indices=[0, 1],
                                              channel_names=['ARPC3', 'DNM2'],
                                              p_threshold=0.05,
                                              verbose=True):
    """
    Detect and fit secondary channels (DNM2, ARPC3) at AP2 positions.
    
    Uses FRAME-BY-FRAME processing for efficiency (matches notebook 3 pattern).
    Implements Aguet's two-step fitting: 'Ac' → 'xyAc' with selection criteria.
    
    Parameters:
    -----------
    track_df_cleaned : pandas DataFrame
        DataFrame with AP2 tracks containing 'mu_x', 'mu_y', 'mu_z', 'frame' columns
    zarr_data : zarr array
        Movie data with shape [t, c, z, y, x]
    master_channel_idx : int
        Master channel index (AP2, default: 2 for channel 3)
    secondary_channel_indices : list of int
        Secondary channel indices to fit (default: [0, 1] for channels 1, 2)
    channel_names : list of str
        Names for secondary channels (default: ['ARPC3', 'DNM2'])
    p_threshold : float
        Significance threshold for hypothesis testing (default: 0.05)
    verbose : bool
        If True, print progress and timing information
        
    Returns:
    --------
    pandas DataFrame
        track_df_cleaned with added columns for each secondary channel:
        - c{ch}_fitted_x, c{ch}_fitted_y, c{ch}_fitted_z : Fitted positions
        - c{ch}_A : Fitted amplitude
        - c{ch}_c : Fitted background
        - c{ch}_A_pstd : Amplitude uncertainty
        - c{ch}_c_pstd : Background uncertainty
        - c{ch}_pval_Ar : P-value for amplitude significance
        - c{ch}_significant : Per-frame significance (1 if p < 0.05, else 0)
        - c{ch}_fit_mode : Which fit was selected ('Ac' or 'xyAc')
        - c{ch}_distance : Distance from master position (pixels)
        
    Also adds track-level classification columns:
        - track_recruited_{name} : Boolean, True if ≥2 frames significant
        - track_n_significant_{name} : Number of significant frames
    """
    
    if verbose:
        print("="*70)
        print("SECONDARY CHANNEL DETECTION - Frame-by-Frame Processing")
        print("="*70)
        print(f"Master channel (AP2): Channel {master_channel_idx + 1}")
        print(f"Secondary channels: {[f'Channel {i+1} ({name})' for i, name in zip(secondary_channel_indices, channel_names)]}")
        print(f"Total frames in data: {len(track_df_cleaned['frame'].unique())}")
        print(f"Total spots to process: {len(track_df_cleaned)}")
        print("="*70 + "\n")
    
    # Get sigma values
    sigma_master = SIGMA_VALUES[f'channel_{master_channel_idx + 1}']
    if isinstance(sigma_master, list):
        sigma_master_xy = sigma_master[0]
    else:
        sigma_master_xy = sigma_master
    
    # Initialize new columns in dataframe
    for ch_idx, ch_name in zip(secondary_channel_indices, channel_names):
        ch_num = ch_idx + 1
        track_df_cleaned[f'c{ch_num}_fitted_x'] = np.nan
        track_df_cleaned[f'c{ch_num}_fitted_y'] = np.nan
        track_df_cleaned[f'c{ch_num}_fitted_z'] = np.nan
        track_df_cleaned[f'c{ch_num}_A'] = np.nan
        track_df_cleaned[f'c{ch_num}_c'] = np.nan
        track_df_cleaned[f'c{ch_num}_A_pstd'] = np.nan
        track_df_cleaned[f'c{ch_num}_c_pstd'] = np.nan
        track_df_cleaned[f'c{ch_num}_sigma_r'] = np.nan
        track_df_cleaned[f'c{ch_num}_pval_Ar'] = np.nan
        track_df_cleaned[f'c{ch_num}_significant'] = 0
        track_df_cleaned[f'c{ch_num}_fit_mode'] = ''
        track_df_cleaned[f'c{ch_num}_distance'] = np.nan
        track_df_cleaned[f'c{ch_num}_amplitude_improvement'] = np.nan
    
    # Build frame-to-rows mapping for efficient processing
    if verbose:
        print("Building frame-to-spots mapping...")
    
    frame_to_rows = {}
    for idx, row in track_df_cleaned.iterrows():
        frame_idx = int(row['frame'])
        if frame_idx not in frame_to_rows:
            frame_to_rows[frame_idx] = []
        frame_to_rows[frame_idx].append(idx)
    
    if verbose:
        print(f"  Mapped {len(frame_to_rows)} frames")
        print(f"  Average spots per frame: {len(track_df_cleaned) / len(frame_to_rows):.1f}\n")
    
    # Timing statistics
    timing_stats = {
        'load_frame': 0,
        'fit_secondary': 0,
        'store_results': 0
    }
    
    fit_timing = {}  # Detailed timing from fit function
    
    # Process frame by frame
    frames_to_process = sorted(frame_to_rows.keys())
    
    if verbose:
        print("Processing frames...")
    
    for frame_idx in tqdm(frames_to_process, desc="Fitting secondary channels", disable=not verbose):
        # Load frame data ONCE for all channels
        t0 = time.time()
        frame_data = zarr_data[frame_idx]  # Shape: [c, z, y, x]
        timing_stats['load_frame'] += time.time() - t0
        
        # Get all spots in this frame
        row_indices = frame_to_rows[frame_idx]
        
        # Process each spot in this frame
        for row_idx in row_indices:
            row = track_df_cleaned.loc[row_idx]
            
            # Get AP2 position (master channel coordinates)
            x_master = row['mu_x']
            y_master = row['mu_y']
            z_master = row['mu_z']
            
            # Fit each secondary channel at this position
            t0 = time.time()
            for ch_idx, ch_name in zip(secondary_channel_indices, channel_names):
                ch_num = ch_idx + 1
                
                # Get secondary channel frame
                secondary_frame = frame_data[ch_idx]
                
                # Get sigma for this channel
                sigma_secondary = SIGMA_VALUES[f'channel_{ch_num}']
                
                # Perform two-step fit (Ac → xyAc with selection)
                fit_result = fit_secondary_channel_at_position(
                    x_master, y_master, z_master,
                    secondary_frame,
                    labels=None,  # TODO: Add label masking support
                    sigma_secondary=sigma_secondary,
                    sigma_master_xy=sigma_master_xy,
                    channel_idx=ch_idx,
                    timing_dict=fit_timing,
                    debug=False
                )
                
                # Store results in dataframe
                if fit_result is not None:
                    best_fit = fit_result['best_fit']
                    
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_fitted_x'] = best_fit['x']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_fitted_y'] = best_fit['y']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_fitted_z'] = best_fit['z']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_A'] = best_fit['A']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_c'] = best_fit['c']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_A_pstd'] = best_fit['A_pstd']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_c_pstd'] = best_fit['c_pstd']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_sigma_r'] = best_fit['sigma_r']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_pval_Ar'] = best_fit['pval_Ar']
                    
                    # Significance: 1 if pval < threshold, else 0
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_significant'] = int(best_fit['pval_Ar'] < p_threshold)
                    
                    # Metadata
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_fit_mode'] = fit_result['selection_reason']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_distance'] = fit_result['distance_from_master']
                    track_df_cleaned.loc[row_idx, f'c{ch_num}_amplitude_improvement'] = fit_result['amplitude_improvement']
            
            timing_stats['fit_secondary'] += time.time() - t0
    
    # =======================================================================
    # Compute track-level recruitment classification
    # =======================================================================
    if verbose:
        print("\nComputing track-level recruitment statistics...")
    
    t0 = time.time()
    
    for ch_idx, ch_name in zip(secondary_channel_indices, channel_names):
        ch_num = ch_idx + 1
        
        # Group by track_id and count significant frames
        track_stats = track_df_cleaned.groupby('track_id').agg({
            f'c{ch_num}_significant': ['sum', 'count']
        })
        
        track_stats.columns = ['n_significant', 'n_total']
        
        # Aguet criterion: >1 significant frame (i.e., ≥2 frames)
        track_stats['recruited'] = track_stats['n_significant'] > 1
        
        # Merge back into main dataframe
        track_df_cleaned = track_df_cleaned.merge(
            track_stats[['recruited', 'n_significant']],
            left_on='track_id',
            right_index=True,
            how='left',
            suffixes=('', f'_{ch_name}')
        )
        
        # Rename columns
        track_df_cleaned.rename(columns={
            'recruited': f'track_recruited_{ch_name}',
            'n_significant': f'track_n_significant_{ch_name}'
        }, inplace=True)
    
    timing_stats['store_results'] += time.time() - t0
    
    # =======================================================================
    # Print summary statistics
    # =======================================================================
    if verbose:
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        for ch_idx, ch_name in zip(secondary_channel_indices, channel_names):
            ch_num = ch_idx + 1
            
            # Per-frame statistics
            n_spots = len(track_df_cleaned)
            n_significant = track_df_cleaned[f'c{ch_num}_significant'].sum()
            pct_significant = 100 * n_significant / n_spots
            
            # Fit mode selection statistics
            fit_mode_counts = track_df_cleaned[f'c{ch_num}_fit_mode'].value_counts()
            
            # Track-level statistics
            n_tracks = track_df_cleaned['track_id'].nunique()
            n_recruited = track_df_cleaned.groupby('track_id')[f'track_recruited_{ch_name}'].first().sum()
            pct_recruited = 100 * n_recruited / n_tracks
            
            print(f"\nChannel {ch_num} ({ch_name}):")
            print(f"  Per-frame significance:")
            print(f"    Significant frames: {n_significant}/{n_spots} ({pct_significant:.1f}%)")
            print(f"  Fit mode selection:")
            for mode, count in fit_mode_counts.items():
                print(f"    {mode}: {count} ({100*count/n_spots:.1f}%)")
            print(f"  Track-level recruitment:")
            print(f"    Recruited tracks: {n_recruited}/{n_tracks} ({pct_recruited:.1f}%)")
        
        print("\n" + "="*70)
        print("TIMING BREAKDOWN")
        print("="*70)
        
        total_time = sum(timing_stats.values())
        
        print("\nTop-level operations:")
        for operation, duration in sorted(timing_stats.items(), key=lambda x: x[1], reverse=True):
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"  {operation:25s}: {duration:8.2f}s ({pct:5.1f}%)")
        
        if fit_timing:
            print("\nDetailed fitting operations:")
            fit_total = sum(fit_timing.values())
            for operation, duration in sorted(fit_timing.items(), key=lambda x: x[1], reverse=True):
                pct = (duration / fit_total * 100) if fit_total > 0 else 0
                pct_total = (duration / total_time * 100) if total_time > 0 else 0
                print(f"  {operation:25s}: {duration:8.2f}s ({pct:5.1f}% of fit, {pct_total:5.1f}% of total)")
        
        print(f"\n  {'TOTAL TIME':25s}: {total_time:8.2f}s")
        print("="*70 + "\n")
    
    return track_df_cleaned


# Test the function
if __name__ == "__main__":
    print("Testing secondary channel detection setup...")
    add_xyac_fit_mode_support()
