"""
LLSM Buffer Frame Analysis Library
Core functions for track filtering based on buffer frame intensities
Uses MATLAB Engine for fitGaussian3D MEX function
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import anderson
from scipy.ndimage import label, median_filter
from skimage.filters import gaussian
from tqdm import tqdm
import warnings
from scipy.signal import fftconvolve
warnings.filterwarnings('ignore')

# MATLAB Engine setup
try:
    import matlab.engine
    print("Initializing MATLAB engine...")
    eng = matlab.engine.start_matlab()
    # Add path to fitGaussian3D MEX file
    # Adjust this path to your llsmtools location
    eng.addpath(r'D:\Akamatsu_Lab\llsmtools\psdetect3d', nargout=0)
    print("MATLAB engine initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize MATLAB engine: {e}")
    eng = None

# Configuration constants
BUFFER_FRAMES = 3
CONFIDENCE_LEVEL = 0.95
K_LEVEL = 1.96  # Z-score for 95% confidence
WINDOW_SIZE = 5

# Sigma values for PSF [sigma_xy, sigma_z] in pixels
SIGMA_VALUES = {
    'channel_1': [1.5, 2.0],
    'channel_2': [1.5, 2.0], 
    'channel_3': 2.25
}


def estimate_gaussian_amplitude_3d(frame, sigma, window_size=15):
    """
    Estimate initial amplitude and background for Gaussian fitting.
    
    Parameters:
    -----------
    frame : 3D numpy array
        Image frame
    sigma : list
        [sigma_xy, sigma_z]
    window_size : int
        Size of local window for background estimation
        
    Returns:
    --------
    A_est : 3D array
        Estimated amplitudes
    c_est : 3D array
        Estimated background
    """

    
    # Create 3D Gaussian kernel
    # Size = ±3σ captures 99.7% of Gaussian
    kernel_radius = int(np.ceil(2 * sigma))
    
    
    # Create coordinate grids
    z, y, x = np.ogrid[-kernel_radius:kernel_radius+1,
                        -kernel_radius:kernel_radius+1, 
                        -kernel_radius:kernel_radius+1]
    
    # Compute Gaussian (no normalization needed since we fit A and c)
    kernel = np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

    # Number of elements in the kernel (used in linear system)
    n = kernel.size

    # Pre-compute sums for linear system (massive speed gain)
    Σg = np.sum(kernel)        # Σg
    Σg2 = np.sum(kernel**2)    # Σg²

       
    # Compute convolutions (fastest way to get sums at each voxel)
    # These replace the explicit loops in the paper's equations

    Σf  = fftconvolve(frame, np.ones_like(kernel), mode="same") ### Keep in mind that rounding errors can occur with this method
    Σgf = fftconvolve(frame, kernel, mode="same")
    Σf2 = fftconvolve(frame**2, np.ones_like(kernel), mode="same")  # needed for variance


    # Solve linear system at each voxel (vectorized for speed)
    # From paper: [Σg²  Σg ] [A] = [Σgf]
    #            [Σg   n  ] [c]   [Σf ]
    denominator = n * Σg2 - Σg**2  # Determinant of 2x2 system
    
    # Avoid division by zero
    if abs(denominator) < 1e-10:
        print(f"  Warning: Singular system - determinant is near zero; returning NaN arrays")
        # Return NaN arrays matching the input frame shape so callers receive valid A,c shapes
        A_nan = np.full_like(frame, np.nan, dtype=float)
        c_nan = np.full_like(frame, np.nan, dtype=float)
        return A_nan, c_nan
    
    # Solve for amplitude A at each voxel (vectorized)
    A = (n * Σgf - Σg * Σf) / denominator


    # Solve for background c at each voxel
    c = (Σf - Σg * A) / n

    
    return A, c

# def fit_gaussian_3d_matlab(window, initial_params, sigma_fixed, fit_mode='xyzAc', matlab_engine=None, debug=False):
#     """
#     Wrapper for MATLAB fitGaussian3D MEX function.

#     Parameters:
#     -----------
#     window : 3D numpy array
#         Data window to fit
#     initial_params : list
#         [x0, y0, z0, amplitude, sigma, background]
#     sigma_fixed : list
#         [sigma_xy, sigma_z] - fixed PSF values
#     fit_mode : str
#         'xyzAc' - fit position, amplitude, background
#         'Ac' - fit only amplitude and background
#         'xyAc' - fit xy position, amplitude, background (z fixed)
#     matlab_engine : matlab.engine object, optional
#         MATLAB engine instance to use
#     debug : bool, optional
#         If True, print debugging information

#     Returns:
#     --------
#     dict with fitted parameters and statistics
#     """
#     from scipy.stats import t as t_dist
    
#     engine = matlab_engine if matlab_engine is not None else eng
#     if engine is None:
#         if debug:
#             print("⚠️  MATLAB engine not available, using Python fallback method")
#         return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)
    
#     try:
#         # Convert numpy array to MATLAB format
#         window_transposed = np.transpose(window, (1, 2, 0))
#         window_clean = np.ascontiguousarray(window_transposed, dtype=np.float64)
#         window_matlab = matlab.double(window_clean.tolist())

#         # Prepare initial parameters for MATLAB
#         # MATLAB expects: [x, y, z, A, sigma_xy, sigma_z, c]
#         init_matlab = matlab.double([
#             float(initial_params[0]), float(initial_params[1]), float(initial_params[2]),
#             float(initial_params[3]), float(sigma_fixed[0]), float(sigma_fixed[1]),
#             float(initial_params[5])
#         ])

#         # Call MATLAB fitGaussian3D
#         result = engine.fitGaussian3D(window_matlab, init_matlab, fit_mode, nargout=4)

#         prm = np.array(result[0]).flatten()
#         prmStd = np.array(result[1]).flatten()
#         res = result[3]

#         # Extract residual statistics from MATLAB dictionary
#         try:
#             sigma_r = float(res.get('std', np.nan))
#         except (TypeError, ValueError):
#             sigma_r = np.nan
        
#         try:
#             hval_AD_raw = res.get('hAD', np.nan)
#             hval_AD = float(hval_AD_raw) if isinstance(hval_AD_raw, bool) else float(hval_AD_raw)
#         except (TypeError, ValueError):
#             hval_AD = np.nan

#         # Calculate p-value for amplitude significance
#         npx = np.sum(~np.isnan(window.flatten()))
        
#         if np.isnan(sigma_r) or sigma_r == 0:
#             se_sigma_r = np.nan
#             se_r = np.nan
#         else:
#             se_sigma_r = sigma_r / np.sqrt(2 * (npx - 1))
#             se_r = se_sigma_r * K_LEVEL
        
#         if debug:
#             print(f"npx: {npx}, sigma_r: {sigma_r}, se_sigma_r: {se_sigma_r}")

#         # Extract parameters from correct indices based on fit_mode
#         if fit_mode == 'xyzAc':
#             # Full fit returns 7 parameters: [x, y, z, A, sigma_xy, sigma_z, c]
#             x = prm[0]
#             y = prm[1]
#             z = prm[2]
#             A = prm[3]
#             c = prm[6]  # Background is at index 6
            
#             # Uncertainties for position fit
#             A_pstd = prmStd[3]
#             c_pstd = prmStd[4]
            
#         elif fit_mode == 'Ac':
#             # Amplitude-only fit returns 2 parameters: [A, c]
#             A = prm[0]
#             c = prm[1]
            
#             # Use initial positions
#             x = initial_params[0]
#             y = initial_params[1]
#             z = initial_params[2]
            
#             # Uncertainties for amplitude-only fit
#             A_pstd = prmStd[0]
#             c_pstd = prmStd[1]
        
#         elif fit_mode == 'xyAc':
#             # XY position refinement returns 6 parameters: [x, y, A, sigma_xy, sigma_z, c]
#             x = prm[0]
#             y = prm[1]
#             z = initial_params[2]  # Z is FIXED
#             A = prm[2]
#             c = prm[5]  # Background is at index 5
            
#             # Uncertainties
#             A_pstd = prmStd[2]
#             c_pstd = prmStd[3]
        
#         else:
#             raise ValueError(f"Unknown fit_mode: {fit_mode}. Must be 'xyzAc', 'Ac', or 'xyAc'")

#         df2 = (npx - 1) * (A_pstd**2 + se_r**2)**2 / (A_pstd**4 + se_r**4)
#         scomb = np.sqrt((A_pstd**2 + se_r**2) / npx)
#         T = (A - sigma_r * K_LEVEL) / scomb
#         pval_Ar = t_dist.cdf(-T, df2)

#         if debug:
#             print(f"✓ MATLAB fitting ({fit_mode}) succeeded: A={A:.2f}, c={c:.2f}, sigma_r={sigma_r:.4f}, pval_Ar={pval_Ar:.4f}")

#         return {
#             'x': x,
#             'y': y,
#             'z': z,
#             'A': A,
#             'sigma_xy': sigma_fixed[0],
#             'sigma_z': sigma_fixed[1],
#             'c': c,
#             'A_pstd': A_pstd,
#             'c_pstd': c_pstd,
#             'sigma_r': sigma_r,
#             'SE_sigma_r': se_sigma_r,
#             'pval_Ar': pval_Ar,
#             'hval_AD': hval_AD,
#             'npx': npx
#         }

#     except Exception as e:
#         if debug:
#             print(f"✗ MATLAB fitGaussian3D failed with error: {e}")
#             print(f"   Attempting Python fallback...")
#         return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)



def fit_gaussian_3d_matlab(window, initial_params, sigma_fixed, fit_mode='xyzAc', matlab_engine=None, debug=False):
    """
    Wrapper for MATLAB fitGaussian3D MEX function.

    Parameters:
    -----------
    window : 3D numpy array
        Data window to fit
    initial_params : list
        [x0, y0, z0, amplitude, sigma, background]
    sigma_fixed : list
        [sigma_xy, sigma_z] - fixed PSF values
    fit_mode : str
        'xyzAc' - fit position, amplitude, background
        'Ac' - fit only amplitude and background
    matlab_engine : matlab.engine object, optional
        MATLAB engine instance to use
    debug : bool, optional
        If True, print debugging information

    Returns:
    --------
    dict with fitted parameters and statistics
    """
    engine = matlab_engine if matlab_engine is not None else eng
    if engine is None:
        if debug:
            print("⚠️  MATLAB engine not available, using Python fallback method")
        return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)
    
    try:
        # Convert numpy array to MATLAB format
        # Need to ensure window is contiguous and in the right order for MATLAB
        window_transposed = np.transpose(window, (1, 2, 0)) #### This is the most critical part, check indexing ####
        window_clean = np.ascontiguousarray(window_transposed, dtype=np.float64)
        window_matlab = matlab.double(window_clean.tolist())

        # Prepare initial parameters for MATLAB
        # MATLAB expects: [x, y, z, A, sigma_xy, sigma_z, c]
        # Ensure all values are float
        init_matlab = matlab.double([
            float(initial_params[0]), float(initial_params[1]), float(initial_params[2]),
            float(initial_params[3]), float(sigma_fixed[0]), float(sigma_fixed[1]),
            float(initial_params[5])
        ])

        # Call MATLAB fitGaussian3D
        # Returns: [prm, prmStd, C, res]
        result = engine.fitGaussian3D(window_matlab, init_matlab, fit_mode, nargout=4)

        prm = np.array(result[0]).flatten()
        prmStd = np.array(result[1]).flatten()
        res = result[3]

        # ===================================================================
        # FIX 1: Extract residual statistics from MATLAB dictionary
        # ===================================================================
        # res is a dict with keys: 'data', 'hAD', 'mean', 'std', 'RSS'
        try:
            sigma_r = float(res.get('std', np.nan))
        except (TypeError, ValueError):
            sigma_r = np.nan
        
        try:
            hval_AD_raw = res.get('hAD', np.nan)
            # hAD is boolean (True/False), convert to float
            hval_AD = float(hval_AD_raw) if isinstance(hval_AD_raw, bool) else float(hval_AD_raw)
        except (TypeError, ValueError):
            hval_AD = np.nan

        # Calculate p-value for amplitude significance
        npx = np.sum(~np.isnan(window.flatten())) #### Does this value match the Aguet MATLAB code? ####
        
        if np.isnan(sigma_r) or sigma_r == 0:
            # If sigma_r is invalid, we can't calculate statistics
            se_sigma_r = np.nan
            se_r = np.nan
        else:
            se_sigma_r = sigma_r / np.sqrt(2 * (npx - 1))
            se_r = se_sigma_r * K_LEVEL
        
        if debug:
            print(f"npx: {npx}, sigma_r: {sigma_r}, se_sigma_r: {se_sigma_r}")

        # ===================================================================
        # FIX 2: Extract parameters from correct indices
        # ===================================================================
        # MATLAB returns different formats depending on fit_mode:
        # - 'xyzAc': [x, y, z, A, sigma_xy, sigma_z, c] - 7 values
        # - 'Ac': [A, c] - 2 values
        
        if fit_mode == 'xyzAc':
            # Full fit returns 7 parameters: [x, y, z, A, sigma_xy, sigma_z, c]
            x = prm[0]
            y = prm[1]
            z = prm[2]
            A = prm[3]
            # prm[4] and prm[5] are sigma_xy and sigma_z (we already know these)
            c = prm[6]  # ✅ Background is at index 6, not 4!
            
            # Uncertainties for position fit
            A_pstd = prmStd[3]
            c_pstd = prmStd[4]
            
        else:  # fit_mode == 'Ac'
            # Amplitude-only fit returns 2 parameters: [A, c]
            A = prm[0]
            c = prm[1]
            
            # Use initial positions
            x = initial_params[0]
            y = initial_params[1]
            z = initial_params[2]
            
            # Uncertainties for amplitude-only fit
            A_pstd = prmStd[0]
            c_pstd = prmStd[1]

        df2 = (npx - 1) * (A_pstd**2 + se_r**2)**2 / (A_pstd**4 + se_r**4)
        scomb = np.sqrt((A_pstd**2 + se_r**2) / npx)
        T = (A - sigma_r * K_LEVEL) / scomb
        pval_Ar = t_dist.cdf(-T, df2)

        if debug:
            print(f"✓ MATLAB fitting succeeded: A={A:.2f}, c={c:.2f}, sigma_r={sigma_r:.4f}, pval_Ar={pval_Ar:.4f}")

        return {
            'x': x,
            'y': y,
            'z': z,
            'A': A,
            'sigma_xy': sigma_fixed[0],
            'sigma_z': sigma_fixed[1],
            'c': c,
            'A_pstd': A_pstd,
            'c_pstd': c_pstd,
            'sigma_r': sigma_r,
            'SE_sigma_r': se_sigma_r,
            'pval_Ar': pval_Ar,
            'hval_AD': hval_AD,
            'npx': npx
        }

    except Exception as e:
        if debug:
            print(f"✗ MATLAB fitGaussian3D failed with error: {e}")
            print(f"   Attempting Python fallback...")
        return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)

# def fit_gaussian_3d_matlab(window, initial_params, sigma_fixed, fit_mode='xyzAc', matlab_engine=None, debug=False):
#     """
#     Wrapper for MATLAB fitGaussian3D MEX function.

#     Parameters:
#     -----------
#     window : 3D numpy array
#         Data window to fit
#     initial_params : list
#         [x0, y0, z0, amplitude, sigma, background]
#     sigma_fixed : list
#         [sigma_xy, sigma_z] - fixed PSF values
#     fit_mode : str
#         'xyzAc' - fit position, amplitude, background
#         'Ac' - fit only amplitude and background
#     matlab_engine : matlab.engine object, optional
#         MATLAB engine instance to use
#     debug : bool, optional
#         If True, print debugging information

#     Returns:
#     --------
#     dict with fitted parameters and statistics
#     """
#     engine = matlab_engine if matlab_engine is not None else eng
#     if engine is None:
#         if debug:
#             print("⚠️  MATLAB engine not available, using Python fallback method")
#         return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)
    
#     try:
#         # Convert numpy array to MATLAB format
#         # Need to ensure window is contiguous and in the right order for MATLAB
#         window_transposed = np.transpose(window, (1, 2, 0)) #### This is the most critical part, check indexing ####
#         window_clean = np.ascontiguousarray(window_transposed, dtype=np.float64)
#         window_matlab = matlab.double(window_clean.tolist())

#         # Prepare initial parameters for MATLAB
#         # MATLAB expects: [x, y, z, A, sigma_xy, sigma_z, c]
#         # Ensure all values are float
#         init_matlab = matlab.double([
#             float(initial_params[0]), float(initial_params[1]), float(initial_params[2]),
#             float(initial_params[3]), float(sigma_fixed[0]), float(sigma_fixed[1]),
#             float(initial_params[5])
#         ])

#         # Call MATLAB fitGaussian3D
#         # Returns: [prm, prmStd, C, res]
#         result = engine.fitGaussian3D(window_matlab, init_matlab, fit_mode, nargout=4)

#         prm = np.array(result[0]).flatten()
#         prmStd = np.array(result[1]).flatten()
#         res = result[3]

#         # Extract residual statistics from MATLAB structure
#         sigma_r = float(res['std']) if hasattr(res, 'std') else np.nan
#         hval_AD = float(res['hAD']) if hasattr(res, 'hAD') else np.nan

#         # Calculate p-value for amplitude significance
#         npx = np.sum(~np.isnan(window.flatten())) #### Does this value match the Aguet MATLAB code? ####
#         se_sigma_r = sigma_r / np.sqrt(2 * (npx - 1))
#         print(f"npx: {npx}, sigma_r: {sigma_r}, se_sigma_r: {se_sigma_r}")
#         se_r = se_sigma_r * K_LEVEL

#         # Handle different return formats based on fit_mode
#         if fit_mode == 'xyzAc':
#             # Full fit: [x, y, z, A, c] - 5 parameters (sigma fixed)
#             A_pstd = prmStd[3]
#             c_pstd = prmStd[4]
#             x, y, z, A, c = prm[0], prm[1], prm[2], prm[3], prm[4]
#         else:  # fit_mode == 'Ac'
#             # Amplitude only: [A, c] - 2 parameters
#             A_pstd = prmStd[0]
#             c_pstd = prmStd[1]
#             # Use initial positions
#             x, y, z = initial_params[0], initial_params[1], initial_params[2]
#             A, c = prm[0], prm[1]

#         df2 = (npx - 1) * (A_pstd**2 + se_r**2)**2 / (A_pstd**4 + se_r**4)
#         scomb = np.sqrt((A_pstd**2 + se_r**2) / npx)
#         T = (A - sigma_r * K_LEVEL) / scomb
#         pval_Ar = t_dist.cdf(-T, df2)

#         if debug:
#             print(f"✓ MATLAB fitting succeeded: A={A:.2f}, c={c:.2f}, sigma_r={sigma_r:.4f}, pval_Ar={pval_Ar:.4f}")

#         return {
#             'x': x,
#             'y': y,
#             'z': z,
#             'A': A,
#             'sigma_xy': sigma_fixed[0],
#             'sigma_z': sigma_fixed[1],
#             'c': c,
#             'A_pstd': A_pstd,
#             'c_pstd': c_pstd,
#             'sigma_r': sigma_r,
#             'SE_sigma_r': se_sigma_r,
#             'pval_Ar': pval_Ar,
#             'hval_AD': hval_AD,
#             'npx': npx
#         }

#     except Exception as e:
#         if debug:
#             print(f"✗ MATLAB fitGaussian3D failed with error: {e}")
#             print(f"   Attempting Python fallback...")
#         return fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode, debug=debug)


def fit_gaussian_3d_python_fallback(window, initial_params, sigma_fixed, fit_mode='xyzAc', debug=False):
    """
    Python fallback for Gaussian fitting if MATLAB is unavailable.
    Uses scipy.optimize.curve_fit
    """
    from scipy.optimize import curve_fit

    if debug:
        print(f"   → Using Python fallback for fitting...")

    nz, ny, nx = window.shape
    
    # Create coordinate grids
    x = np.arange(nx)
    y = np.arange(ny) 
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = (X.ravel(), Y.ravel(), Z.ravel())
    
    # Flatten data and remove NaN values
    data_flat = window.ravel()
    valid_mask = ~np.isnan(data_flat)
    
    if np.sum(valid_mask) < 20:
        return None
        
    coords_valid = (coords[0][valid_mask], coords[1][valid_mask], coords[2][valid_mask])
    data_valid = data_flat[valid_mask]
    
    def gaussian_3d(coords, x0, y0, z0, amplitude, sigma_xy, sigma_z, background):
        X, Y, Z = coords
        exp_term = -((X - x0)**2 + (Y - y0)**2) / (2 * sigma_xy**2) - (Z - z0)**2 / (2 * sigma_z**2)
        return amplitude * np.exp(exp_term) + background
    
    try:
        if fit_mode == 'xyzAc':
            p0 = [initial_params[0], initial_params[1], initial_params[2],
                  initial_params[3], sigma_fixed[0], sigma_fixed[1], initial_params[5]]
            
            bounds = ([0, 0, 0, 0, sigma_fixed[0]*0.9, sigma_fixed[1]*0.9, 0],
                     [nx, ny, nz, np.inf, sigma_fixed[0]*1.1, sigma_fixed[1]*1.1, np.inf])
            
            popt, pcov = curve_fit(gaussian_3d, coords_valid, data_valid, 
                                 p0=p0, bounds=bounds, maxfev=5000)
            
        elif fit_mode == 'Ac':
            x0, y0, z0 = initial_params[0:3]
            p0 = [initial_params[3], initial_params[5]]
            
            def fixed_position_gaussian(coords, A, c):
                return gaussian_3d(coords, x0, y0, z0, A, sigma_fixed[0], sigma_fixed[1], c)
            
            popt, pcov = curve_fit(fixed_position_gaussian, coords_valid, data_valid,
                                 p0=p0, bounds=([0, 0], [np.inf, np.inf]), maxfev=5000)
            
            popt = np.array([x0, y0, z0, popt[0], sigma_fixed[0], sigma_fixed[1], popt[1]])
            
    except Exception as e:
        if debug:
            print(f"   ✗ Python fitting failed: {e}")
        return None

    # Calculate residuals and statistics
    fitted_data = gaussian_3d(coords_valid, popt[0], popt[1], popt[2],
                            popt[3], popt[4], popt[5], popt[6])
    residuals = data_valid - fitted_data
    sigma_r = np.std(residuals)
    se_sigma_r = sigma_r / np.sqrt(2 * (len(data_valid) - 1))

    # Calculate p-value
    se_r = se_sigma_r * K_LEVEL
    npx = len(data_valid)
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt)

    A_pstd = perr[3] if fit_mode == 'xyzAc' else perr[0]
    df2 = (npx - 1) * (A_pstd**2 + se_r**2)**2 / (A_pstd**4 + se_r**4)
    scomb = np.sqrt((A_pstd**2 + se_r**2) / npx)
    T = (popt[3] - sigma_r * K_LEVEL) / scomb
    pval_Ar = t_dist.cdf(-T, df2)

    # Anderson-Darling test
    if len(residuals) > 7:
        ad_result = anderson(residuals, dist='norm')
        hval_AD = ad_result.statistic
    else:
        hval_AD = np.nan

    if debug:
        print(f"   ✓ Python fallback succeeded: A={popt[3]:.2f}, c={popt[6]:.2f}, sigma_r={sigma_r:.4f}, pval_Ar={pval_Ar:.4f}")

    return {
        'x': popt[0], 'y': popt[1], 'z': popt[2],
        'A': popt[3], 'sigma_xy': popt[4], 'sigma_z': popt[5], 'c': popt[6],
        'A_pstd': A_pstd, 'c_pstd': perr[6] if fit_mode == 'xyzAc' else perr[1],
        'sigma_r': sigma_r, 'SE_sigma_r': se_sigma_r,
        'pval_Ar': pval_Ar, 'hval_AD': hval_AD, 'npx': npx
    }


def interpolate_track_buffer(x, y, z, frame, labels, sigma, channel_idx, timing_dict=None, debug=False):
    """
    Interpolate track position and intensity in buffer frame.

    Parameters:
    -----------
    x, y, z : float
        Track position
    frame : 3D array
        Image frame
    labels : 3D array
        Label mask for excluding nearby objects
    sigma : list
        PSF parameters [sigma_xy, sigma_z]
    channel_idx : int
        Channel index
    timing_dict : dict, optional
        Dictionary to accumulate timing statistics
    debug : bool, optional
        If True, print debugging information

    Returns:
    --------
    dict with fitted parameters or None if fitting fails
    """
    import time

    nz, ny, nx = frame.shape

    # Convert to integer coordinates
    xi = int(np.round(np.clip(x, 0, nx-1)))
    yi = int(np.round(np.clip(y, 0, ny-1)))
    zi = int(np.round(np.clip(z, 0, nz-1)))

    # Define window boundaries
    w1x = int(np.ceil(sigma))
    w2x = int(np.ceil(2 * sigma))
    w1z = int(np.ceil(sigma))
    w2z = int(np.ceil(2 * sigma))

    # Extract window
    t0 = time.time()
    xa = slice(max(0, xi-w2x), min(nx, xi+w2x+1))
    ya = slice(max(0, yi-w2x), min(ny, yi+w2x+1))
    za = slice(max(0, zi-w2z), min(nz, zi+w2z+1))

    window = frame[za, ya, xa].copy()
    if timing_dict is not None:
        timing_dict['extract_window'] = timing_dict.get('extract_window', 0) + (time.time() - t0)

    # Mask out other objects if labels provided
    t0 = time.time()
    if labels is not None:
        mask_window = labels[za, ya, xa]
        center_z = min(zi-max(0, zi-w2z), mask_window.shape[0]-1)
        center_y = min(yi-max(0, yi-w2x), mask_window.shape[1]-1)
        center_x = min(xi-max(0, xi-w2x), mask_window.shape[2]-1)
        center_label = mask_window[center_z, center_y, center_x]

        # Set pixels belonging to other objects to NaN
        window[np.logical_and(mask_window != 0, mask_window != center_label)] = np.nan
    if timing_dict is not None:
        timing_dict['apply_mask'] = timing_dict.get('apply_mask', 0) + (time.time() - t0)

    # Relative coordinates in window
    ox = xi - max(0, xi-w2x)
    oy = yi - max(0, yi-w2x)
    oz = zi - max(0, zi-w2z)
    # print(ox,oy,oz)

    # Estimate initial parameters
    t0 = time.time()

    # window_transposed = np.transpose(window, (1, 2, 0))  # [z,y,x] → [y,x,z] #### Indexing is super important, check this. ####
    # window_matlab = matlab.double(window_transposed.tolist())
    # window_matlab = matlab.double(window.tolist())

    # Ensure sigma is in list form for MATLAB (expecting [sigma_xy, sigma_z])
    # if np.isscalar(sigma):
    #     sigma_list = [float(sigma), float(sigma)]
    # else:
    #     # Try to coerce to two floats; if only one provided, duplicate it
    #     sigma_list = list(map(float, sigma))
    #     if len(sigma_list) == 1:
    #         sigma_list = [sigma_list[0], sigma_list[0]]

    # sigma_matlab = matlab.double(sigma_list)
    
    # # Call MATLAB's estGaussianAmplitude3D
    # # Syntax: [A_est, c_est] = estGaussianAmplitude3D(frame, sigma, 'WindowSize', windowSize)
    # A_est_matlab, c_est_matlab = eng.estGaussianAmplitude3D(window_matlab, sigma_matlab, 'WindowSize', float(WINDOW_SIZE), nargout = 2)

    # # Convert MATLAB arrays back to numpy
    # A_est = np.array(A_est_matlab)
    # c_est = np.array(c_est_matlab)

    A_est, c_est = estimate_gaussian_amplitude_3d(window, sigma, WINDOW_SIZE) #### Aguet llsmtools runs this function on the entire frame, and uses global coordinates.
    ai = A_est[oz, oy, ox] if not np.isnan(A_est[oz, oy, ox]) else np.nanmax(window)
    ci = c_est[oz, oy, ox] if not np.isnan(c_est[oz, oy, ox]) else np.nanmin(window)
    # ai = A_est[oy, ox, oz] if not np.isnan(A_est[oy, ox, oz]) else np.nanmax(window)
    # ci = c_est[oy, ox, oz] if not np.isnan(c_est[oy, ox, oz]) else np.nanmin(window)
    
    # print(A_est[5,5,5], c_est[5,5,5])
    # print(f"Channel {channel_idx}: Initial estimates - A: {ai}, c: {ci}")
    if timing_dict is not None:
        timing_dict['estimate_params'] = timing_dict.get('estimate_params', 0) + (time.time() - t0)

    # Initial parameters for fitting
    initial_params = [x-xi+ox, y-yi+oy, z-zi+oz, ai, sigma, ci]

    # Ensure sigma is a list [sigma_xy, sigma_z]
    sigma_list = [sigma, sigma] if np.isscalar(sigma) else sigma

    # Try position fitting first
    t0 = time.time()
    fit_result = fit_gaussian_3d_matlab(window, initial_params, sigma_list, fit_mode='xyzAc', debug=debug)
    if timing_dict is not None:
        timing_dict['gaussian_fit'] = timing_dict.get('gaussian_fit', 0) + (time.time() - t0)

    if fit_result is not None:
        # Check if fitted position is within reasonable bounds
        dx = fit_result['x'] - ox
        dy = fit_result['y'] - oy
        dz = fit_result['z'] - oz

        if abs(dx) > w1x or abs(dy) > w1x or abs(dz) > w1z:
            # Position fit failed, try amplitude-only fit
            if debug:
                print(f"   Position fit out of bounds (dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}), trying amplitude-only fit...")
            t0 = time.time()
            fit_result = fit_gaussian_3d_matlab(window, initial_params, sigma_list, fit_mode='Ac', debug=debug)
            if timing_dict is not None:
                timing_dict['gaussian_fit_retry'] = timing_dict.get('gaussian_fit_retry', 0) + (time.time() - t0)

            if fit_result is not None:
                fit_result['x'] = x
                fit_result['y'] = y
                fit_result['z'] = z
        else:
            # Convert back to global coordinates
            fit_result['x'] = xi + dx
            fit_result['y'] = yi + dy
            fit_result['z'] = zi + dz

    return fit_result


def process_track_buffers(track, movie_data, buffer_frames=BUFFER_FRAMES, pbar=None):
    """
    Process buffer frames for a single track.

    Parameters:
    -----------
    track : Track object
        Track to process
    movie_data : zarr array or numpy array
        Movie data [t, c, z, y, x]
    buffer_frames : int
        Number of buffer frames
    pbar : tqdm progress bar, optional
        Progress bar for channel processing

    Returns:
    --------
    dict with start_buffer and end_buffer results
    """
    n_frames, n_channels, n_z, n_y, n_x = movie_data.shape

    # Get track temporal information
    track_frames = track.frames.values if hasattr(track.frames, 'values') else track.frames
    track_start = np.min(track_frames)
    track_end = np.max(track_frames)

    # Initialize buffer storage
    start_buffer = {ch: [] for ch in range(n_channels)}
    end_buffer = {ch: [] for ch in range(n_channels)}

    # Process start buffer frames
    start_buffer_frames = range(max(0, track_start - buffer_frames), track_start)
    for buffer_idx, frame_idx in enumerate(start_buffer_frames):
        # Load frame data
        frame_data = movie_data[frame_idx]

        # TODO: Load actual detection masks if available
        labels = None

        for ch_idx in range(n_channels):
            channel_frame = frame_data[ch_idx]
            sigma = SIGMA_VALUES[f'channel_{ch_idx+1}']

            # Use first track position for buffer before
            if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
                x = track.x.iloc[0]
                y = track.y.iloc[0]
                z = track.z.iloc[0]
            else:
                # Handle different track data structures
                x = track.x[0] if isinstance(track.x, (list, np.ndarray)) else track.x
                y = track.y[0] if isinstance(track.y, (list, np.ndarray)) else track.y
                z = track.z[0] if isinstance(track.z, (list, np.ndarray)) else track.z


            result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, ch_idx)
            start_buffer[ch_idx].append(result)

            if pbar is not None:
                pbar.update(1)

    # Process end buffer frames
    end_buffer_frames = range(track_end + 1, min(n_frames, track_end + buffer_frames + 1))
    for buffer_idx, frame_idx in enumerate(end_buffer_frames):
        frame_data = movie_data[frame_idx]
        labels = None

        for ch_idx in range(n_channels):
            channel_frame = frame_data[ch_idx]
            sigma = SIGMA_VALUES[f'channel_{ch_idx+1}']

            # Use last track position for buffer after
            if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
                x = track.x.iloc[-1]
                y = track.y.iloc[-1]
                z = track.z.iloc[-1]
            else:
                x = track.x[-1] if isinstance(track.x, (list, np.ndarray)) else track.x
                y = track.y[-1] if isinstance(track.y, (list, np.ndarray)) else track.y
                z = track.z[-1] if isinstance(track.z, (list, np.ndarray)) else track.z

            result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, ch_idx)
            end_buffer[ch_idx].append(result)

            if pbar is not None:
                pbar.update(1)

    track_id = track.track_id.values[0] if hasattr(track.track_id, 'values') else track.track_id

    return {
        'start_buffer': start_buffer,
        'end_buffer': end_buffer,
        'track_id': track_id
    }

def get_binary_segment_lengths(binary_array):
    """
    Get lengths and values of consecutive segments in a binary array.
    
    Example:
    --------
    >>> binary_array = [True, True, False, False, False, True]
    >>> lengths, values = get_binary_segment_lengths(binary_array)
    >>> # Returns: lengths = [2, 3, 1], values = [1, 0, 1]
    
    Parameters:
    -----------
    binary_array : array-like
        Boolean or binary (0/1) array
        
    Returns:
    --------
    lengths : ndarray
        Length of each consecutive segment
    values : ndarray
        Value (0 or 1) of each segment
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


def classify_tracks(tracks, movie_data, buffer_frames=BUFFER_FRAMES, ap2_channel_idx=2):
    """
    Classify tracks based on Aguet's buffer frame analysis criteria.

    Aguet Criteria (all must pass):
    1. At least Tbuffer=2 consecutive non-significant frames in EACH buffer
    2. Frames bordering the signal (last start, first end) must be non-significant
    3. Maximum buffer intensity must be less than maximum track intensity

    Categories:
    - Complete: Full buffers calculated, passed all intensity criteria
    - Partial: Truncated at beginning or end of acquisition
    - Persistent: Present throughout entire acquisition
    - Invalid: Failed buffer intensity significance tests

    Parameters:
    -----------
    tracks : list
        List of Track objects
    movie_data : array
        Movie data [t, c, z, y, x]
    buffer_frames : int
        Number of buffer frames (default: 3)
    ap2_channel_idx : int
        Index of AP2 channel (0-based) - only this channel will be analyzed

    Returns:
    --------
    dict with categorized track lists
    """
    import time

    n_frames = movie_data.shape[0]
    
    # Aguet parameters
    Tbuffer = 2  # Minimum consecutive non-significant frames required
    p_threshold = 0.05  # Significance threshold

    complete_tracks = []
    partial_tracks = []
    persistent_tracks = []
    invalid_tracks = []

    # Timing statistics
    timing_stats = {
        'load_frame_data': 0,
        'interpolate_buffer_overhead': 0,
        'intensity_analysis': 0,
        'track_classification': 0
    }

    interpolate_timing = {}

    print(f"Processing {len(tracks)} tracks (analyzing AP2 channel only - index {ap2_channel_idx})...")
    print(f"Using Aguet criteria: Tbuffer={Tbuffer}, p_threshold={p_threshold}")

    for i, track in enumerate(tqdm(tracks, desc="Classifying tracks")):

        # Get track temporal extent
        track_frames = track.frames.values if hasattr(track.frames, 'values') else track.frames
        track_start = np.min(track_frames)
        track_end = np.max(track_frames)
        track_length = len(track_frames)

        # Skip very short tracks
        if track_length < 3:
            invalid_tracks.append(track)
            continue

        # Check if track is persistent (at movie boundaries)
        if track_start <= buffer_frames and track_end >= n_frames - buffer_frames:
            persistent_tracks.append(track)
            continue

        # Check if track is partial (insufficient buffer frames)
        at_start_boundary = (track_start <= buffer_frames)
        at_end_boundary = (track_end >= n_frames - buffer_frames)
        
        if at_start_boundary or at_end_boundary:
            partial_tracks.append(track)
            continue

        # === PROCESS BUFFER FRAMES (AP2 CHANNEL ONLY) ===
        start_buffer = []
        end_buffer = []

        sigma = SIGMA_VALUES[f'channel_{ap2_channel_idx+1}']

        # Process start buffer frames
        start_buffer_frames = range(max(0, track_start - buffer_frames), track_start)
        for buffer_idx, frame_idx in enumerate(start_buffer_frames):
            t0 = time.time()
            channel_frame = movie_data[frame_idx, ap2_channel_idx]
            labels = None
            timing_stats['load_frame_data'] += time.time() - t0

            # Use first track position
            if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
                x, y, z = track.x.iloc[0], track.y.iloc[0], track.z.iloc[0]
            else:
                x = track.x[0] if isinstance(track.x, (list, np.ndarray)) else track.x
                y = track.y[0] if isinstance(track.y, (list, np.ndarray)) else track.y
                z = track.z[0] if isinstance(track.z, (list, np.ndarray)) else track.z

            t0 = time.time()
            result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, 
                                             ap2_channel_idx, timing_dict=interpolate_timing)
            timing_stats['interpolate_buffer_overhead'] += time.time() - t0
            start_buffer.append(result)

        # Process end buffer frames
        end_buffer_frames = range(track_end + 1, min(n_frames, track_end + buffer_frames + 1))
        for buffer_idx, frame_idx in enumerate(end_buffer_frames):
            t0 = time.time()
            channel_frame = movie_data[frame_idx, ap2_channel_idx]
            labels = None
            timing_stats['load_frame_data'] += time.time() - t0

            # Use last track position
            if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
                x, y, z = track.x.iloc[-1], track.y.iloc[-1], track.z.iloc[-1]
            else:
                x = track.x[-1] if isinstance(track.x, (list, np.ndarray)) else track.x
                y = track.y[-1] if isinstance(track.y, (list, np.ndarray)) else track.y
                z = track.z[-1] if isinstance(track.z, (list, np.ndarray)) else track.z

            t0 = time.time()
            result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, 
                                             ap2_channel_idx, timing_dict=interpolate_timing)
            timing_stats['interpolate_buffer_overhead'] += time.time() - t0
            end_buffer.append(result)

        # Store buffer results
        track_id = track.track_id.values[0] if hasattr(track.track_id, 'values') else track.track_id
        buffer_results = {
            'start_buffer': start_buffer,
            'end_buffer': end_buffer,
            'track_id': track_id,
            'channel_idx': ap2_channel_idx
        }
        track.buffer_results = buffer_results

        # === APPLY AGUET CLASSIFICATION CRITERIA ===
        t0 = time.time()

        # Get track maximum intensity (A + c)
        if hasattr(track, 'A') and hasattr(track, 'c'):
            track_max_intensity = np.max(track.A[ap2_channel_idx] + track.c[ap2_channel_idx])
        elif hasattr(track, 'peak_intensities'):
            track_max_intensity = track.peak_intensities[ap2_channel_idx]
        elif hasattr(track, 'intensities'):
            track_max_intensity = np.max(track.intensities[ap2_channel_idx])
        else:
            # Conservative: assume track is valid if we can't determine intensity
            track_max_intensity = np.inf

        # === ANALYZE START BUFFER ===
        # Extract p-values (H0: A = background)
        start_pvals = np.array([b['pval_Ar'] if b is not None and 'pval_Ar' in b else 1.0 
                                for b in start_buffer])
        
        # Binary: True = significant (reject H0), False = non-significant (background)
        start_significant = start_pvals < p_threshold
        
        # Get consecutive segment lengths
        start_lengths, start_values = get_binary_segment_lengths(start_significant)
        
        # Check criteria
        has_start_buffer_segment = np.any((start_lengths >= Tbuffer) & (start_values == 0))
        start_border_valid = (start_significant[-1] == False) if len(start_significant) > 0 else False
        
        # Get maximum start buffer intensity
        start_max_intensity = -np.inf
        for b in start_buffer:
            if b is not None and 'A' in b and 'c' in b:
                intensity = b['A'] + b['c']
                start_max_intensity = max(start_max_intensity, intensity)

        # === ANALYZE END BUFFER ===
        end_pvals = np.array([b['pval_Ar'] if b is not None and 'pval_Ar' in b else 1.0 
                              for b in end_buffer])
        
        end_significant = end_pvals < p_threshold
        end_lengths, end_values = get_binary_segment_lengths(end_significant)
        
        has_end_buffer_segment = np.any((end_lengths >= Tbuffer) & (end_values == 0))
        end_border_valid = (end_significant[0] == False) if len(end_significant) > 0 else False
        
        # Get maximum end buffer intensity
        end_max_intensity = -np.inf
        for b in end_buffer:
            if b is not None and 'A' in b and 'c' in b:
                intensity = b['A'] + b['c']
                end_max_intensity = max(end_max_intensity, intensity)

        # === APPLY ALL AGUET CRITERIA ===
        buffer_max_intensity = max(start_max_intensity, end_max_intensity)
        
        # All criteria must pass for track to be valid
        valid = (has_start_buffer_segment and 
                 has_end_buffer_segment and
                 start_border_valid and 
                 end_border_valid and
                 buffer_max_intensity < track_max_intensity)

        timing_stats['intensity_analysis'] += time.time() - t0

        # === CLASSIFY TRACK ===
        t0 = time.time()
        if valid:
            complete_tracks.append(track)
        else:
            invalid_tracks.append(track)
        timing_stats['track_classification'] += time.time() - t0

    # === PRINT RESULTS ===
    print(f"\n{'='*60}")
    print(f"Classification complete (Aguet criteria):")
    print(f"  Complete tracks: {len(complete_tracks)}")
    print(f"  Partial tracks: {len(partial_tracks)}")
    print(f"  Persistent tracks: {len(persistent_tracks)}")
    print(f"  Invalid tracks: {len(invalid_tracks)}")

    print(f"\n{'='*60}")
    print(f"TIMING BREAKDOWN - TOP LEVEL OPERATIONS:")
    print(f"{'='*60}")

    all_timing = timing_stats.copy()
    interpolate_subtotal = sum(interpolate_timing.values())
    all_timing['interpolate_buffer_overhead'] -= interpolate_subtotal
    if all_timing['interpolate_buffer_overhead'] < 0:
        all_timing['interpolate_buffer_overhead'] = 0

    total_time = sum(all_timing.values()) + interpolate_subtotal

    for operation, duration in sorted(all_timing.items(), key=lambda x: x[1], reverse=True):
        pct = (duration / total_time * 100) if total_time > 0 else 0
        print(f"  {operation:30s}: {duration:8.2f}s ({pct:5.1f}%)")

    print(f"\n{'-'*60}")
    print(f"DETAILED BREAKDOWN - interpolate_track_buffer sub-steps:")
    print(f"{'-'*60}")

    interpolate_total = sum(interpolate_timing.values())
    for operation, duration in sorted(interpolate_timing.items(), key=lambda x: x[1], reverse=True):
        pct = (duration / interpolate_total * 100) if interpolate_total > 0 else 0
        pct_total = (duration / total_time * 100) if total_time > 0 else 0
        print(f"  {operation:30s}: {duration:8.2f}s ({pct:5.1f}% of interp, {pct_total:5.1f}% of total)")

    print(f"\n{'='*60}")
    print(f"  {'GRAND TOTAL':30s}: {total_time:8.2f}s")
    print(f"{'='*60}\n")

    return {
        'complete': complete_tracks,
        'partial': partial_tracks,
        'persistent': persistent_tracks,
        'invalid': invalid_tracks
    }

# def classify_tracks(tracks, movie_data, buffer_frames=BUFFER_FRAMES, ap2_channel_idx=2):
#     """
#     Classify tracks based on buffer frame analysis criteria.

#     Categories:
#     - Complete: Full buffers calculated, passed intensity criteria
#     - Partial: Truncated at beginning or end of acquisition
#     - Persistent: Present throughout entire acquisition
#     - Invalid: Failed buffer intensity significance tests

#     Parameters:
#     -----------
#     tracks : list
#         List of Track objects
#     movie_data : array
#         Movie data [t, c, z, y, x]
#     buffer_frames : int
#         Number of buffer frames
#     ap2_channel_idx : int
#         Index of AP2 channel (0-based) - only this channel will be analyzed

#     Returns:
#     --------
#     dict with categorized track lists
#     """
#     import time

#     n_frames = movie_data.shape[0]

#     complete_tracks = []
#     partial_tracks = []
#     persistent_tracks = []
#     invalid_tracks = []

#     # Timing statistics - high level
#     timing_stats = {
#         'load_frame_data': 0,
#         'interpolate_buffer_overhead': 0,
#         'intensity_analysis': 0,
#         'track_classification': 0
#     }

#     # Detailed timing for interpolate_track_buffer sub-operations
#     interpolate_timing = {}

#     print(f"Processing {len(tracks)} tracks (analyzing AP2 channel only - index {ap2_channel_idx})...")

#     for i, track in enumerate(tqdm(tracks, desc="Classifying tracks")):

#         # Get track temporal extent
#         track_frames = track.frames.values if hasattr(track.frames, 'values') else track.frames
#         track_start = np.min(track_frames)
#         track_end = np.max(track_frames)
#         track_length = len(track_frames)

#         # Skip very short tracks
#         if track_length < 3:
#             invalid_tracks.append(track)
#             continue

#         # Check if track is persistent
#         if track_start <= buffer_frames and track_end >= n_frames - buffer_frames:
#             persistent_tracks.append(track)
#             continue

#         # Process start and end buffers with timing - AP2 CHANNEL ONLY
#         start_buffer = []
#         end_buffer = []

#         sigma = SIGMA_VALUES[f'channel_{ap2_channel_idx+1}']

#         # === PROCESS START BUFFER FRAMES ===
#         start_buffer_frames = range(max(0, track_start - buffer_frames), track_start)
#         for buffer_idx, frame_idx in enumerate(start_buffer_frames):
#             # Time: Load frame data - ONLY AP2 CHANNEL
#             t0 = time.time()
#             channel_frame = movie_data[frame_idx, ap2_channel_idx]
#             # nz, ny, nx = channel_frame.shape
#             # A_est, c_est = estimate_gaussian_amplitude_3d(channel_frame, sigma, WINDOW_SIZE)
#             labels = None
#             timing_stats['load_frame_data'] += time.time() - t0

#             # Use first track position for buffer before
#             if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
#                 x = track.x.iloc[0]
#                 y = track.y.iloc[0]
#                 z = track.z.iloc[0]
#             else:
#                 x = track.x[0] if isinstance(track.x, (list, np.ndarray)) else track.x
#                 y = track.y[0] if isinstance(track.y, (list, np.ndarray)) else track.y
#                 z = track.z[0] if isinstance(track.z, (list, np.ndarray)) else track.z

#             # xi = int(np.round(np.clip(x, 0, nx-1)))
#             # yi = int(np.round(np.clip(y, 0, ny-1)))
#             # zi = int(np.round(np.clip(z, 0, nz-1)))
#             # ai = A_est[zi, yi, xi] if not np.isnan(A_est[zi, yi, xi]) else np.nanmax(window)
#             # ci = c_est[zi, yi, xi] if not np.isnan(c_est[zi, yi, xi]) else np.nanmin(window)
#             # Time: Interpolate buffer with detailed sub-timing
#             t0 = time.time()
#             result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, ap2_channel_idx, timing_dict=interpolate_timing)
#             timing_stats['interpolate_buffer_overhead'] += time.time() - t0

#             start_buffer.append(result)

#         # === PROCESS END BUFFER FRAMES ===
#         end_buffer_frames = range(track_end + 1, min(n_frames, track_end + buffer_frames + 1))
#         for buffer_idx, frame_idx in enumerate(end_buffer_frames):
#             # Time: Load frame data - ONLY AP2 CHANNEL
#             t0 = time.time()
#             channel_frame = movie_data[frame_idx, ap2_channel_idx]
#             labels = None
#             timing_stats['load_frame_data'] += time.time() - t0

#             # Use last track position for buffer after
#             if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
#                 x = track.x.iloc[-1]
#                 y = track.y.iloc[-1]
#                 z = track.z.iloc[-1]
#             else:
#                 x = track.x[-1] if isinstance(track.x, (list, np.ndarray)) else track.x
#                 y = track.y[-1] if isinstance(track.y, (list, np.ndarray)) else track.y
#                 z = track.z[-1] if isinstance(track.z, (list, np.ndarray)) else track.z

#             # Time: Interpolate buffer with detailed sub-timing
#             t0 = time.time()
#             result = interpolate_track_buffer(x, y, z, channel_frame, labels, sigma, ap2_channel_idx, timing_dict=interpolate_timing)
#             timing_stats['interpolate_buffer_overhead'] += time.time() - t0

#             end_buffer.append(result)

#         # Store buffer results
#         track_id = track.track_id.values[0] if hasattr(track.track_id, 'values') else track.track_id
#         buffer_results = {
#             'start_buffer': start_buffer,
#             'end_buffer': end_buffer,
#             'track_id': track_id,
#             'channel_idx': ap2_channel_idx
#         }
#         track.buffer_results = buffer_results

#         # === ANALYZE AP2 CHANNEL BUFFER INTENSITIES ===
#         t0 = time.time()
#         valid_start = True
#         valid_end = True

#         # Get track's peak intensity for comparison
#         if hasattr(track, 'peak_intensities'):
#             track_peak = track.peak_intensities[ap2_channel_idx]
#         else:
#             track_peak = np.max(track.intensities[ap2_channel_idx]) if hasattr(track, 'intensities') else 1000

#         threshold = track_peak / 2.5

#         # Check start buffer
#         if track_start > buffer_frames:
#             start_intensities = [b['A'] for b in start_buffer if b is not None]

#             if start_intensities:
#                 mean_start = np.mean(start_intensities)
#                 if mean_start > threshold:
#                     valid_start = False

#         # Check end buffer
#         if track_end < n_frames - buffer_frames:
#             end_intensities = [b['A'] for b in end_buffer if b is not None]

#             if end_intensities:
#                 mean_end = np.mean(end_intensities)
#                 if mean_end > threshold:
#                     valid_end = False

#         timing_stats['intensity_analysis'] += time.time() - t0

#         # === CLASSIFY TRACK ===
#         t0 = time.time()
#         if not valid_start or not valid_end:
#             invalid_tracks.append(track)
#         elif track_start <= buffer_frames or track_end >= n_frames - buffer_frames:
#             partial_tracks.append(track)
#         else:
#             complete_tracks.append(track)
#         timing_stats['track_classification'] += time.time() - t0

#     print(f"\n{'='*60}")
#     print(f"Classification complete:")
#     print(f"  Complete tracks: {len(complete_tracks)}")
#     print(f"  Partial tracks: {len(partial_tracks)}")
#     print(f"  Persistent tracks: {len(persistent_tracks)}")
#     print(f"  Invalid tracks: {len(invalid_tracks)}")

#     print(f"\n{'='*60}")
#     print(f"TIMING BREAKDOWN - TOP LEVEL OPERATIONS:")
#     print(f"{'='*60}")

#     # Combine high-level timing with detailed interpolate timing
#     all_timing = timing_stats.copy()

#     # Subtract interpolate sub-operations from overhead to get true overhead
#     interpolate_subtotal = sum(interpolate_timing.values())
#     all_timing['interpolate_buffer_overhead'] -= interpolate_subtotal
#     if all_timing['interpolate_buffer_overhead'] < 0:
#         all_timing['interpolate_buffer_overhead'] = 0

#     total_time = sum(all_timing.values()) + interpolate_subtotal

#     for operation, duration in sorted(all_timing.items(), key=lambda x: x[1], reverse=True):
#         pct = (duration / total_time * 100) if total_time > 0 else 0
#         print(f"  {operation:30s}: {duration:8.2f}s ({pct:5.1f}%)")

#     print(f"\n{'-'*60}")
#     print(f"DETAILED BREAKDOWN - interpolate_track_buffer sub-steps:")
#     print(f"{'-'*60}")

#     interpolate_total = sum(interpolate_timing.values())
#     for operation, duration in sorted(interpolate_timing.items(), key=lambda x: x[1], reverse=True):
#         pct = (duration / interpolate_total * 100) if interpolate_total > 0 else 0
#         pct_total = (duration / total_time * 100) if total_time > 0 else 0
#         print(f"  {operation:30s}: {duration:8.2f}s ({pct:5.1f}% of interp, {pct_total:5.1f}% of total)")

#     print(f"\n{'='*60}")
#     print(f"  {'GRAND TOTAL':30s}: {total_time:8.2f}s")
#     print(f"{'='*60}\n")

#     return {
#         'complete': complete_tracks,
#         'partial': partial_tracks,
#         'persistent': persistent_tracks,
#         'invalid': invalid_tracks
#     }


def determine_track_validity(track, ap2_channel_idx=2, Tbuffer=2, p_threshold=0.05, verbose=False):
    """
    Determine if track passes Aguet buffer frame criteria.
    
    This now matches the EXACT logic from classify_tracks().
    
    Aguet Criteria (all must pass):
    1. At least Tbuffer=2 consecutive non-significant frames in start buffer
    2. At least Tbuffer=2 consecutive non-significant frames in end buffer
    3. Last frame of start buffer must be non-significant
    4. First frame of end buffer must be non-significant
    5. Maximum buffer intensity < maximum track intensity
    
    Parameters:
    -----------
    track : Track object
        Track with buffer_results attribute
    ap2_channel_idx : int
        Channel index (default: 2)
    Tbuffer : int
        Minimum consecutive non-significant frames (default: 2)
    p_threshold : float
        Significance threshold (default: 0.05)
    verbose : bool
        If True, print why track failed
    
    Returns:
    --------
    bool
        True if valid, False if invalid
    """
    if not hasattr(track, 'buffer_results'):
        if verbose:
            print("    ❌ No buffer_results attribute")
        return False
    
    buffer_results = track.buffer_results
    start_buffer = buffer_results['start_buffer']
    end_buffer = buffer_results['end_buffer']
    
    # Get track maximum intensity
    if hasattr(track, 'A') and hasattr(track, 'c'):
        track_max_intensity = np.max(track.A[ap2_channel_idx] + track.c[ap2_channel_idx])
    elif hasattr(track, 'peak_intensities'):
        track_max_intensity = track.peak_intensities[ap2_channel_idx]
    elif hasattr(track, 'intensities'):
        track_max_intensity = np.max(track.intensities[ap2_channel_idx])
    else:
        if verbose:
            print("    ❌ Cannot determine track intensity")
        return False
    
    # === ANALYZE START BUFFER ===
    start_pvals = np.array([b['pval_Ar'] if b is not None and 'pval_Ar' in b else 1.0 
                            for b in start_buffer])
    start_significant = start_pvals < p_threshold
    start_lengths, start_values = get_binary_segment_lengths(start_significant)
    
    has_start_buffer_segment = np.any((start_lengths >= Tbuffer) & (start_values == 0))
    start_border_valid = (start_significant[-1] == False) if len(start_significant) > 0 else False
    
    start_max_intensity = -np.inf
    for b in start_buffer:
        if b is not None and 'A' in b and 'c' in b:
            intensity = b['A'] + b['c']
            start_max_intensity = max(start_max_intensity, intensity)
    
    # === ANALYZE END BUFFER ===
    end_pvals = np.array([b['pval_Ar'] if b is not None and 'pval_Ar' in b else 1.0 
                          for b in end_buffer])
    end_significant = end_pvals < p_threshold
    end_lengths, end_values = get_binary_segment_lengths(end_significant)
    
    has_end_buffer_segment = np.any((end_lengths >= Tbuffer) & (end_values == 0))
    end_border_valid = (end_significant[0] == False) if len(end_significant) > 0 else False
    
    end_max_intensity = -np.inf
    for b in end_buffer:
        if b is not None and 'A' in b and 'c' in b:
            intensity = b['A'] + b['c']
            end_max_intensity = max(end_max_intensity, intensity)
    
    # === CHECK ALL CRITERIA ===
    buffer_max_intensity = max(start_max_intensity, end_max_intensity)
    intensity_criterion = buffer_max_intensity < track_max_intensity
    
    valid = (has_start_buffer_segment and 
             has_end_buffer_segment and
             start_border_valid and 
             end_border_valid and
             intensity_criterion)
    
    # If verbose and failed, print why
    if verbose and not valid:
        print(f"    ❌ FAILED Aguet criteria:")
        if not has_start_buffer_segment:
            print(f"       • Start buffer: no ≥{Tbuffer} consecutive non-sig frames")
        if not has_end_buffer_segment:
            print(f"       • End buffer: no ≥{Tbuffer} consecutive non-sig frames")
        if not start_border_valid:
            print(f"       • Last start frame is significant (p={start_pvals[-1]:.4f})")
        if not end_border_valid:
            print(f"       • First end frame is significant (p={end_pvals[0]:.4f})")
        if not intensity_criterion:
            print(f"       • Buffer max ({buffer_max_intensity:.1f}) ≥ Track max ({track_max_intensity:.1f})")
        print(f"    Details: Track max={track_max_intensity:.1f}, Start buf max={start_max_intensity:.1f}, End buf max={end_max_intensity:.1f}")
    
    return valid


def fit_all_track_frames(tracks, movie_data, buffer_frames=3, channels_to_fit=None):
    """
    Perform Gaussian fitting on all frames of tracks (track frames + buffer frames).
    
    This function extends the classify_tracks approach but fits ALL frames and returns
    the full set of fit parameters and uncertainties instead of classifying.
    
    Parameters:
    -----------
    tracks : list
        List of Track objects to fit
    movie_data : zarr array or numpy array
        Movie data with shape [t, c, z, y, x]
    buffer_frames : int
        Number of buffer frames before/after track (default: 3)
    channels_to_fit : list of int, optional
        Channel indices to fit (0-based). If None, fits all channels.
        Example: [0, 1, 2] for channels 1, 2, 3
    
    Returns:
    --------
    list of dicts, one per track containing:
        - 'track_id': Track identifier
        - 'fit_results': dict with keys for each channel
            - Each channel contains list of fit results, one per frame
            - Frame order: [buffer_before..., track_frames..., buffer_after...]
        - 'frame_indices': array of frame indices corresponding to fit_results
        - 'frame_types': array indicating 'buffer_before', 'track', 'buffer_after'
    """
    from llsm_buffer_analysis import (
        interpolate_track_buffer, 
        SIGMA_VALUES
    )
    
    n_frames, n_channels, n_z, n_y, n_x = movie_data.shape
    
    # Default to all channels if not specified
    if channels_to_fit is None:
        channels_to_fit = list(range(n_channels))
    
    all_track_fits = []
    
    print(f"Fitting {len(tracks)} tracks across {len(channels_to_fit)} channels...")
    print(f"Buffer frames: {buffer_frames}")
    
    for track in tqdm(tracks, desc="Fitting tracks"):
        # Get track temporal extent
        track_frames = track.frames.values if hasattr(track.frames, 'values') else track.frames
        track_start = np.min(track_frames)
        track_end = np.max(track_frames)
        track_length = len(track_frames)
        
        # Get track ID
        track_id = track.track_id.values[0] if hasattr(track.track_id, 'values') else track.track_id
        
        # Define all frames to fit (buffers + track)
        start_buffer_frames = list(range(max(0, track_start - buffer_frames), track_start))
        track_frame_list = list(track_frames)
        end_buffer_frames = list(range(track_end + 1, min(n_frames, track_end + buffer_frames + 1)))
        
        all_frames_to_fit = start_buffer_frames + track_frame_list + end_buffer_frames
        
        # Create frame type labels
        frame_types = (['buffer_before'] * len(start_buffer_frames) + 
                      ['track'] * len(track_frame_list) + 
                      ['buffer_after'] * len(end_buffer_frames))
        
        # Initialize storage for this track
        fit_results = {ch_idx: [] for ch_idx in channels_to_fit}
        
        # Get track positions (for interpolation to buffer frames)
        if hasattr(track, 'x') and hasattr(track.x, 'iloc'):
            track_x = track.x.values
            track_y = track.y.values
            track_z = track.z.values
        else:
            track_x = track.x if isinstance(track.x, (list, np.ndarray)) else [track.x]
            track_y = track.y if isinstance(track.y, (list, np.ndarray)) else [track.y]
            track_z = track.z if isinstance(track.z, (list, np.ndarray)) else [track.z]
        
        # Fit each frame
        for frame_idx, frame_type in zip(all_frames_to_fit, frame_types):
            # Determine which track position to use for this frame
            if frame_type == 'buffer_before':
                # Use first track position
                x, y, z = track_x[0], track_y[0], track_z[0]
            elif frame_type == 'buffer_after':
                # Use last track position
                x, y, z = track_x[-1], track_y[-1], track_z[-1]
            else:  # frame_type == 'track'
                # Use actual position at this frame
                frame_in_track = np.where(track_frames == frame_idx)[0][0]
                x, y, z = track_x[frame_in_track], track_y[frame_in_track], track_z[frame_in_track]
            
            # Load frame data (all channels at once for efficiency)
            frame_data = movie_data[frame_idx]
            
            # TODO: Load actual detection masks if available
            labels = None
            
            # Fit each channel
            for ch_idx in channels_to_fit:
                channel_frame = frame_data[ch_idx]
                sigma = SIGMA_VALUES[f'channel_{ch_idx+1}']
                
                # Perform fitting using existing function
                result = interpolate_track_buffer(
                    x, y, z, 
                    channel_frame, 
                    labels, 
                    sigma, 
                    ch_idx
                )
                
                # Add frame metadata to result
                if result is not None:
                    result['frame_idx'] = frame_idx
                    result['frame_type'] = frame_type
                
                fit_results[ch_idx].append(result)
        
        # Store results for this track
        track_fit_data = {
            'track_id': track_id,
            'fit_results': fit_results,
            'frame_indices': np.array(all_frames_to_fit),
            'frame_types': np.array(frame_types),
            'track_start': track_start,
            'track_end': track_end,
            'n_buffer_before': len(start_buffer_frames),
            'n_track_frames': len(track_frame_list),
            'n_buffer_after': len(end_buffer_frames)
        }
        
        all_track_fits.append(track_fit_data)
    
    print(f"\nFitting complete!")
    print(f"  Total tracks fitted: {len(all_track_fits)}")
    print(f"  Channels fitted: {channels_to_fit}")
    
    return all_track_fits


def extract_fit_parameters_to_dataframe(track_fits, channel_idx=None):
    """
    Convert fit results to a pandas DataFrame for easier analysis.
    
    Parameters:
    -----------
    track_fits : list
        Output from fit_all_track_frames()
    channel_idx : int, optional
        If specified, only extract data for this channel.
        If None, creates a multi-level DataFrame with all channels.
    
    Returns:
    --------
    pandas DataFrame with columns:
        - track_id: Track identifier
        - frame_idx: Frame number
        - frame_type: 'buffer_before', 'track', or 'buffer_after'
        - channel: Channel index (if channel_idx=None)
        - x, y, z: Fitted positions
        - A: Amplitude
        - c: Background
        - A_pstd: Amplitude standard deviation
        - c_pstd: Background standard deviation
        - sigma_r: Residual standard deviation
        - SE_sigma_r: Standard error of sigma_r
        - pval_Ar: P-value for amplitude significance
        - hval_AD: Anderson-Darling test statistic
        - npx: Number of pixels used in fit
    """
    records = []
    
    for track_data in track_fits:
        track_id = track_data['track_id']
        frame_indices = track_data['frame_indices']
        frame_types = track_data['frame_types']
        
        channels = [channel_idx] if channel_idx is not None else track_data['fit_results'].keys()
        
        for ch in channels:
            fit_list = track_data['fit_results'][ch]
            
            for i, (frame_idx, frame_type, fit_result) in enumerate(
                zip(frame_indices, frame_types, fit_list)
            ):
                if fit_result is None:
                    # Failed fit
                    record = {
                        'track_id': track_id,
                        'frame_idx': frame_idx,
                        'frame_type': frame_type,
                        'channel': ch,
                        'fit_success': False
                    }
                else:
                    # Successful fit
                    record = {
                        'track_id': track_id,
                        'frame_idx': frame_idx,
                        'frame_type': frame_type,
                        'channel': ch,
                        'fit_success': True,
                        'x': fit_result.get('x', np.nan),
                        'y': fit_result.get('y', np.nan),
                        'z': fit_result.get('z', np.nan),
                        'A': fit_result.get('A', np.nan),
                        'c': fit_result.get('c', np.nan),
                        'A_pstd': fit_result.get('A_pstd', np.nan),
                        'c_pstd': fit_result.get('c_pstd', np.nan),
                        'sigma_xy': fit_result.get('sigma_xy', np.nan),
                        'sigma_z': fit_result.get('sigma_z', np.nan),
                        'sigma_r': fit_result.get('sigma_r', np.nan),
                        'SE_sigma_r': fit_result.get('SE_sigma_r', np.nan),
                        'pval_Ar': fit_result.get('pval_Ar', np.nan),
                        'hval_AD': fit_result.get('hval_AD', np.nan),
                        'npx': fit_result.get('npx', np.nan)
                    }
                
                records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by track and frame
    if not df.empty:
        df = df.sort_values(['track_id', 'channel', 'frame_idx']).reset_index(drop=True)
    
    return df


def get_track_intensity_profiles(track_fits, channel_idx):
    """
    Extract intensity profiles (A + c) for all tracks in a given channel.
    
    Useful for visualizing intensity over time including buffer frames.
    
    Parameters:
    -----------
    track_fits : list
        Output from fit_all_track_frames()
    channel_idx : int
        Channel to extract (0-based)
    
    Returns:
    --------
    dict with keys = track_id, values = dict containing:
        - 'frame_indices': array of frame numbers
        - 'frame_types': array of frame type labels
        - 'intensities': array of A + c values
        - 'amplitudes': array of A values
        - 'backgrounds': array of c values
        - 'amplitude_errors': array of A_pstd values
    """
    profiles = {}
    
    for track_data in track_fits:
        track_id = track_data['track_id']
        frame_indices = track_data['frame_indices']
        frame_types = track_data['frame_types']
        fit_list = track_data['fit_results'][channel_idx]
        
        intensities = []
        amplitudes = []
        backgrounds = []
        amplitude_errors = []
        
        for fit_result in fit_list:
            if fit_result is not None:
                A = fit_result.get('A', np.nan)
                c = fit_result.get('c', np.nan)
                A_pstd = fit_result.get('A_pstd', np.nan)
                
                intensities.append(A + c)
                amplitudes.append(A)
                backgrounds.append(c)
                amplitude_errors.append(A_pstd)
            else:
                intensities.append(np.nan)
                amplitudes.append(np.nan)
                backgrounds.append(np.nan)
                amplitude_errors.append(np.nan)
        
        profiles[track_id] = {
            'frame_indices': frame_indices,
            'frame_types': frame_types,
            'intensities': np.array(intensities),
            'amplitudes': np.array(amplitudes),
            'backgrounds': np.array(backgrounds),
            'amplitude_errors': np.array(amplitude_errors)
        }
    
    return profiles


"""
Multi-scale Gaussian Fitting for AP2 and DNM2 Channels
Performs Ac-mode fitting at multiple sigma values and selects best fit based on significance.
Designed to work between notebooks 2 and 3 in the LLSM analysis pipeline.
"""

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import from existing codebase
from llsm_buffer_analysis import (
    fit_gaussian_3d_matlab,
    estimate_gaussian_amplitude_3d,
    K_LEVEL,
    eng  # MATLAB engine
)


def fit_spot_multiscale(frame, x, y, z, sigma_values, fit_mode='Ac', 
                        p_threshold=0.05, labels=None, debug=False):
    """
    Fit a 3D Gaussian at multiple sigma scales and select the best fit.
    
    Selection criteria (in order):
    1. Fits with p-value < p_threshold (significant)
    2. Among significant fits, choose lowest p-value
    3. If multiple sigmas have same p-value, average A and c
    
    Parameters:
    -----------
    frame : 3D numpy array
        Image data [z, y, x]
    x, y, z : float
        Center position for fitting
    sigma_values : list of float or list of [sigma_xy, sigma_z]
        Sigma values to test (e.g., [1.25, 1.75, 2.25])
    fit_mode : str, default='Ac'
        Fitting mode ('Ac' for amplitude/background only)
    p_threshold : float, default=0.05
        Significance threshold for fit selection
    labels : 3D array, optional
        Label mask to exclude nearby objects
    debug : bool, default=False
        If True, print debugging information
        
    Returns:
    --------
    dict or None
        Best fit results with keys: 'x', 'y', 'z', 'A', 'c', 'A_pstd', 'c_pstd',
        'sigma_xy', 'sigma_z', 'pval_Ar', 'sigma_r', 'selected_sigma', 'n_averaged'
        Returns None if all fits fail
    """
    
    nz, ny, nx = frame.shape
    
    # Convert to integer coordinates for windowing
    xi = int(np.round(np.clip(x, 0, nx-1)))
    yi = int(np.round(np.clip(y, 0, ny-1)))
    zi = int(np.round(np.clip(z, 0, nz-1)))
    
    # Storage for all fits
    all_fits = []
    
    # Try each sigma value
    for sigma in sigma_values:
        # Convert sigma to [sigma_xy, sigma_z] format if scalar
        if np.isscalar(sigma):
            sigma_list = [sigma, sigma * 1.5]  # Assume z is 1.5x wider
        else:
            sigma_list = list(sigma)
        
        # Define window boundaries (±2σ)
        w2x = int(np.ceil(2 * sigma_list[0]))
        w2z = int(np.ceil(2 * sigma_list[1]))
        
        # Extract window with boundary checking
        xa = slice(max(0, xi-w2x), min(nx, xi+w2x+1))
        ya = slice(max(0, yi-w2x), min(ny, yi+w2x+1))
        za = slice(max(0, zi-w2z), min(nz, zi+w2z+1))
        
        window = frame[za, ya, xa].copy()
        
        # Apply label mask if provided
        if labels is not None:
            mask_window = labels[za, ya, xa]
            center_z = min(zi-max(0, zi-w2z), mask_window.shape[0]-1)
            center_y = min(yi-max(0, yi-w2x), mask_window.shape[1]-1)
            center_x = min(xi-max(0, xi-w2x), mask_window.shape[2]-1)
            center_label = mask_window[center_z, center_y, center_x]
            window[np.logical_and(mask_window != 0, mask_window != center_label)] = np.nan
        
        # Relative coordinates in window
        ox = xi - max(0, xi-w2x)
        oy = yi - max(0, yi-w2x)
        oz = zi - max(0, zi-w2z)
        
        # Estimate initial parameters
        A_est, c_est = estimate_gaussian_amplitude_3d(window, sigma_list[0], window_size=5)
        ai = A_est[oz, oy, ox] if not np.isnan(A_est[oz, oy, ox]) else np.nanmax(window)
        ci = c_est[oz, oy, ox] if not np.isnan(c_est[oz, oy, ox]) else np.nanmin(window)
        
        # Initial parameters: [x, y, z, A, sigma, c]
        initial_params = [ox, oy, oz, ai, sigma_list[0], ci]
        
        # Perform Gaussian fit
        fit_result = fit_gaussian_3d_matlab(
            window,
            initial_params,
            sigma_list,
            fit_mode=fit_mode,
            debug=debug
        )
        
        if fit_result is not None:
            # Convert to global coordinates
            fit_result['x'] = x  # Keep original center for Ac mode
            fit_result['y'] = y
            fit_result['z'] = z
            fit_result['sigma_xy'] = sigma_list[0]
            fit_result['sigma_z'] = sigma_list[1]
            fit_result['selected_sigma'] = sigma
            
            all_fits.append(fit_result)
    
    # No successful fits
    if len(all_fits) == 0:
        return None
    
    # Select best fit based on significance and p-value
    significant_fits = [f for f in all_fits if f['pval_Ar'] < p_threshold]
    
    if len(significant_fits) == 0:
        # No significant fits - choose lowest p-value among all
        best_fit = min(all_fits, key=lambda f: f['pval_Ar'])
        best_fit['n_averaged'] = 1
        return best_fit
    
    # Find minimum p-value among significant fits
    min_pval = min(f['pval_Ar'] for f in significant_fits)
    
    # Get all fits with minimum p-value (handles ties)
    best_fits = [f for f in significant_fits if abs(f['pval_Ar'] - min_pval) < 1e-10]
    
    if len(best_fits) == 1:
        best_fits[0]['n_averaged'] = 1
        return best_fits[0]
    
    # Multiple fits with same p-value - average A and c
    avg_fit = best_fits[0].copy()
    avg_fit['A'] = np.mean([f['A'] for f in best_fits])
    avg_fit['c'] = np.mean([f['c'] for f in best_fits])
    avg_fit['A_pstd'] = np.mean([f['A_pstd'] for f in best_fits])
    avg_fit['c_pstd'] = np.mean([f['c_pstd'] for f in best_fits])
    avg_fit['sigma_r'] = np.mean([f['sigma_r'] for f in best_fits])
    avg_fit['n_averaged'] = len(best_fits)
    avg_fit['selected_sigma'] = 'averaged'
    
    return avg_fit


def fit_ap2_and_dnm2_multiscale(df_with_dnm2, zarr_data,
                                ap2_channel_idx=2,
                                dnm2_channel_idx=1,
                                sigma_values=[1.25, 1.75, 2.25],
                                p_threshold=0.05,
                                verbose=True):
    """
    Perform multi-scale Gaussian fitting for AP2 and DNM2 channels.
    
    Fitting strategy:
    - AP2 channel: Fit at AP2 spot centers
    - DNM2 channel (DNM2+ spots): Fit at DNM2 centers
    - DNM2 channel (DNM2- spots): Fit at AP2 centers
    
    For each spot, fits are performed at multiple sigma values and the best
    fit is selected based on significance (p < 0.05) and lowest p-value.
    
    Parameters:
    -----------
    df_with_dnm2 : pandas DataFrame
        Output from find_dnm2_at_ap2_positions() with columns:
        ['mu_x', 'mu_y', 'mu_z', 'frame', 'track_id', 'dnm2_positive', 
         'dnm2_mu_x', 'dnm2_mu_y', 'dnm2_mu_z', ...]
    zarr_data : zarr array
        Movie data with shape [t, c, z, y, x]
    ap2_channel_idx : int, default=2
        AP2 channel index (0-based, channel 3 = index 2)
    dnm2_channel_idx : int, default=1
        DNM2 channel index (0-based, channel 2 = index 1)
    sigma_values : list, default=[1.25, 1.75, 2.25]
        Sigma values to test for each fit (detection scales)
    p_threshold : float, default=0.05
        Significance threshold for fit selection
    verbose : bool, default=True
        If True, show progress bars and print summary
        
    Returns:
    --------
    pandas DataFrame
        df_with_dnm2 with added columns for each channel:
        
        AP2 channel (c3_*):
        - c3_A, c3_c: Amplitude and background
        - c3_A_pstd, c3_c_pstd: Uncertainties
        - c3_sigma_r: Residual std
        - c3_pval_Ar: P-value for significance
        - c3_sigma_xy, c3_sigma_z: Selected PSF sigma
        - c3_selected_sigma: Which sigma was selected
        - c3_n_averaged: Number of sigmas averaged (if tied)
        
        DNM2 channel (c2_*):
        - c2_A, c2_c: Amplitude and background
        - c2_A_pstd, c2_c_pstd: Uncertainties
        - c2_sigma_r: Residual std
        - c2_pval_Ar: P-value for significance
        - c2_sigma_xy, c2_sigma_z: Selected PSF sigma
        - c2_selected_sigma: Which sigma was selected
        - c2_n_averaged: Number of sigmas averaged (if tied)
        - c2_fit_center: 'dnm2' or 'ap2' (which center was used)
    """
    
    if verbose:
        print("="*70)
        print("MULTI-SCALE GAUSSIAN FITTING (Ac MODE)")
        print("="*70)
        print(f"AP2 channel: Channel {ap2_channel_idx + 1}")
        print(f"DNM2 channel: Channel {dnm2_channel_idx + 1}")
        print(f"Sigma values to test: {sigma_values}")
        print(f"Significance threshold: p < {p_threshold}")
        print(f"Total spots to process: {len(df_with_dnm2)}")
        print("="*70 + "\n")
    
    # Create copy to avoid modifying original
    result_df = df_with_dnm2.copy()
    
    # Initialize AP2 channel columns
    ap2_ch = ap2_channel_idx + 1
    result_df[f'c{ap2_ch}_A'] = np.nan
    result_df[f'c{ap2_ch}_c'] = np.nan
    result_df[f'c{ap2_ch}_A_pstd'] = np.nan
    result_df[f'c{ap2_ch}_c_pstd'] = np.nan
    result_df[f'c{ap2_ch}_sigma_r'] = np.nan
    result_df[f'c{ap2_ch}_pval_Ar'] = np.nan
    result_df[f'c{ap2_ch}_sigma_xy'] = np.nan
    result_df[f'c{ap2_ch}_sigma_z'] = np.nan
    result_df[f'c{ap2_ch}_selected_sigma'] = pd.Series(dtype='object')
    result_df[f'c{ap2_ch}_n_averaged'] = 0
    
    # Initialize DNM2 channel columns
    dnm2_ch = dnm2_channel_idx + 1
    result_df[f'c{dnm2_ch}_A'] = np.nan
    result_df[f'c{dnm2_ch}_c'] = np.nan
    result_df[f'c{dnm2_ch}_A_pstd'] = np.nan
    result_df[f'c{dnm2_ch}_c_pstd'] = np.nan
    result_df[f'c{dnm2_ch}_sigma_r'] = np.nan
    result_df[f'c{dnm2_ch}_pval_Ar'] = np.nan
    result_df[f'c{dnm2_ch}_sigma_xy'] = np.nan
    result_df[f'c{dnm2_ch}_sigma_z'] = np.nan
    result_df[f'c{dnm2_ch}_selected_sigma'] = pd.Series(dtype='object')
    result_df[f'c{dnm2_ch}_n_averaged'] = 0
    result_df[f'c{dnm2_ch}_fit_center'] = ''  # 'dnm2' or 'ap2'
    
    # Build frame-to-rows mapping
    if verbose:
        print("Building frame-to-spots mapping...")
    
    frame_to_rows = {}
    for idx, row in result_df.iterrows():
        frame_idx = int(row['frame'])
        if frame_idx not in frame_to_rows:
            frame_to_rows[frame_idx] = []
        frame_to_rows[frame_idx].append(idx)
    
    if verbose:
        print(f"  Mapped {len(frame_to_rows)} frames\n")
    
    # Counters for statistics
    n_ap2_success = 0
    n_ap2_fail = 0
    n_dnm2_success = 0
    n_dnm2_fail = 0
    
    # Process frame by frame
    frames_to_process = sorted(frame_to_rows.keys())
    
    for frame_idx in tqdm(frames_to_process, desc="Fitting gaussians", disable=not verbose):
        # Load frame data ONCE for all channels
        frame_data = zarr_data[frame_idx]  # Shape: [c, z, y, x]
        
        ap2_frame = frame_data[ap2_channel_idx]
        dnm2_frame = frame_data[dnm2_channel_idx]
        
        # Get all spots in this frame
        row_indices = frame_to_rows[frame_idx]
        
        # Process each spot
        for row_idx in row_indices:
            row = result_df.loc[row_idx]
            
            # Get AP2 position
            x_ap2 = row['mu_x']
            y_ap2 = row['mu_y']
            z_ap2 = row['mu_z']
            
            # ================================================================
            # FIT AP2 CHANNEL at AP2 center
            # ================================================================
            ap2_fit = fit_spot_multiscale(
                ap2_frame,
                x_ap2, y_ap2, z_ap2,
                sigma_values=sigma_values,
                fit_mode='Ac',
                p_threshold=p_threshold,
                labels=None,
                debug=False
            )
            
            if ap2_fit is not None:
                n_ap2_success += 1
                result_df.loc[row_idx, f'c{ap2_ch}_A'] = ap2_fit['A']
                result_df.loc[row_idx, f'c{ap2_ch}_c'] = ap2_fit['c']
                result_df.loc[row_idx, f'c{ap2_ch}_A_pstd'] = ap2_fit['A_pstd']
                result_df.loc[row_idx, f'c{ap2_ch}_c_pstd'] = ap2_fit['c_pstd']
                result_df.loc[row_idx, f'c{ap2_ch}_sigma_r'] = ap2_fit['sigma_r']
                result_df.loc[row_idx, f'c{ap2_ch}_pval_Ar'] = ap2_fit['pval_Ar']
                result_df.loc[row_idx, f'c{ap2_ch}_sigma_xy'] = ap2_fit['sigma_xy']
                result_df.loc[row_idx, f'c{ap2_ch}_sigma_z'] = ap2_fit['sigma_z']
                result_df.loc[row_idx, f'c{ap2_ch}_selected_sigma'] = ap2_fit['selected_sigma']
                result_df.loc[row_idx, f'c{ap2_ch}_n_averaged'] = ap2_fit['n_averaged']
            else:
                n_ap2_fail += 1
            
            # ================================================================
            # FIT DNM2 CHANNEL
            # ================================================================
            # Determine fit center based on DNM2 detection status
            is_dnm2_positive = row['dnm2_positive']
            
            if is_dnm2_positive:
                # Check if DNM2 coordinates are valid (not 'two_spots')
                dnm2_x = row['dnm2_mu_x']
                dnm2_y = row['dnm2_mu_y']
                dnm2_z = row['dnm2_mu_z']
                
                if dnm2_x == 'two_spots':
                    # Equidistant case - use AP2 center
                    x_fit = x_ap2
                    y_fit = y_ap2
                    z_fit = z_ap2
                    fit_center = 'ap2_equidistant'
                else:
                    # Use DNM2 center
                    x_fit = dnm2_x
                    y_fit = dnm2_y
                    z_fit = dnm2_z
                    fit_center = 'dnm2'
            else:
                # DNM2-negative - use AP2 center
                x_fit = x_ap2
                y_fit = y_ap2
                z_fit = z_ap2
                fit_center = 'ap2'
            
            # Perform DNM2 channel fit
            dnm2_fit = fit_spot_multiscale(
                dnm2_frame,
                x_fit, y_fit, z_fit,
                sigma_values=sigma_values,
                fit_mode='Ac',
                p_threshold=p_threshold,
                labels=None,
                debug=False
            )
            
            if dnm2_fit is not None:
                n_dnm2_success += 1
                result_df.loc[row_idx, f'c{dnm2_ch}_A'] = dnm2_fit['A']
                result_df.loc[row_idx, f'c{dnm2_ch}_c'] = dnm2_fit['c']
                result_df.loc[row_idx, f'c{dnm2_ch}_A_pstd'] = dnm2_fit['A_pstd']
                result_df.loc[row_idx, f'c{dnm2_ch}_c_pstd'] = dnm2_fit['c_pstd']
                result_df.loc[row_idx, f'c{dnm2_ch}_sigma_r'] = dnm2_fit['sigma_r']
                result_df.loc[row_idx, f'c{dnm2_ch}_pval_Ar'] = dnm2_fit['pval_Ar']
                result_df.loc[row_idx, f'c{dnm2_ch}_sigma_xy'] = dnm2_fit['sigma_xy']
                result_df.loc[row_idx, f'c{dnm2_ch}_sigma_z'] = dnm2_fit['sigma_z']
                result_df.loc[row_idx, f'c{dnm2_ch}_selected_sigma'] = dnm2_fit['selected_sigma']
                result_df.loc[row_idx, f'c{dnm2_ch}_n_averaged'] = dnm2_fit['n_averaged']
                result_df.loc[row_idx, f'c{dnm2_ch}_fit_center'] = fit_center
            else:
                n_dnm2_fail += 1
                result_df.loc[row_idx, f'c{dnm2_ch}_fit_center'] = fit_center
    
    # Print summary statistics
    if verbose:
        total_spots = len(result_df)
        print("\n" + "="*70)
        print("FITTING SUMMARY")
        print("="*70)
        print(f"\nAP2 Channel (c{ap2_ch}):")
        print(f"  Successful fits: {n_ap2_success}/{total_spots} ({100*n_ap2_success/total_spots:.1f}%)")
        print(f"  Failed fits: {n_ap2_fail}/{total_spots} ({100*n_ap2_fail/total_spots:.1f}%)")
        
        print(f"\nDNM2 Channel (c{dnm2_ch}):")
        print(f"  Successful fits: {n_dnm2_success}/{total_spots} ({100*n_dnm2_success/total_spots:.1f}%)")
        print(f"  Failed fits: {n_dnm2_fail}/{total_spots} ({100*n_dnm2_fail/total_spots:.1f}%)")
        
        # Sigma selection statistics for successful fits
        if n_ap2_success > 0:
            sigma_counts = result_df[result_df[f'c{ap2_ch}_A'].notna()][f'c{ap2_ch}_selected_sigma'].value_counts()
            print(f"\nAP2 Sigma selection:")
            for sigma, count in sigma_counts.items():
                print(f"  σ = {sigma}: {count} ({100*count/n_ap2_success:.1f}%)")
        
        if n_dnm2_success > 0:
            sigma_counts = result_df[result_df[f'c{dnm2_ch}_A'].notna()][f'c{dnm2_ch}_selected_sigma'].value_counts()
            print(f"\nDNM2 Sigma selection:")
            for sigma, count in sigma_counts.items():
                print(f"  σ = {sigma}: {count} ({100*count/n_dnm2_success:.1f}%)")
            
            # Fit center statistics
            center_counts = result_df[result_df[f'c{dnm2_ch}_A'].notna()][f'c{dnm2_ch}_fit_center'].value_counts()
            print(f"\nDNM2 Fit center used:")
            for center, count in center_counts.items():
                print(f"  {center}: {count} ({100*count/n_dnm2_success:.1f}%)")
        
        print("="*70 + "\n")
    
    return result_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use fit_all_track_frames in your pipeline.
    
    This assumes you have:
    1. Loaded your tracks (e.g., from pickle files)
    2. Loaded your movie data (e.g., from zarr)
    3. Have llsm_buffer_analysis.py available for import
    """
    
    # Example setup (adjust to your actual data loading)
    # import pickle
    # import zarr
    # 
    # # Load tracks
    # with open('tracks.pkl', 'rb') as f:
    #     tracks = pickle.load(f)
    # 
    # # Load movie
    # z2 = zarr.open('movie.zarr', mode='r')
    
    # Fit all frames for selected tracks
    # Option 1: Fit all channels
    # track_fits = fit_all_track_frames(
    #     tracks[:10],  # First 10 tracks
    #     z2,
    #     buffer_frames=3
    # )
    
    # Option 2: Fit only AP2 channel (channel 3 = index 2)
    # track_fits = fit_all_track_frames(
    #     tracks[:10],
    #     z2,
    #     buffer_frames=3,
    #     channels_to_fit=[2]  # Only AP2
    # )
    
    # Convert to DataFrame for analysis
    # df = extract_fit_parameters_to_dataframe(track_fits, channel_idx=2)
    # print(df.head(20))
    
    # Get intensity profiles
    # profiles = get_track_intensity_profiles(track_fits, channel_idx=2)
    # for track_id, profile in list(profiles.items())[:3]:
    #     print(f"\nTrack {track_id}:")
    #     print(f"  Frames: {profile['frame_indices']}")
    #     print(f"  Types: {profile['frame_types']}")
    #     print(f"  Intensities: {profile['intensities']}")

# Cleanup function for MATLAB engine
def cleanup_matlab_engine():
    """Close MATLAB engine when done."""
    global eng
    if eng is not None:
        eng.quit()
        print("MATLAB engine closed")