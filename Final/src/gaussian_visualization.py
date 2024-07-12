import numpy as np

def visualize_3D_gaussians(zarr_obj, gaussians_df):
    
    """
    Visualizes 3D Gaussians based on the parameters extracted from a DataFrame and overlays them onto a 3D array.

    Parameters
    ----------
    zarr_obj : zarr.core.Array
        The raw 3D image data from which the Gaussians have been segmented and fitted.
    gaussians_df : pd.DataFrame
        A DataFrame containing the Gaussian parameters with columns 'amplitude', 'mu_x', 'mu_y', 'mu_z', 'sigma_x', 'sigma_y', 'sigma_z'.

    Returns
    -------
    np.ndarray
        A 3D array with the visualized Gaussians.
    """

    
    image_gaussians = np.zeros((zarr_obj.shape[2],zarr_obj.shape[3],zarr_obj.shape[4]))

    # Replace zero sigma values with 1
    # gaussians_df['sigma_x'] = gaussians_df['sigma_x'].replace(0, 1)
    # gaussians_df['sigma_y'] = gaussians_df['sigma_y'].replace(0, 1)
    # gaussians_df['sigma_z'] = gaussians_df['sigma_z'].replace(0, 1)

    gaussians_df.loc[gaussians_df['sigma_x'] == 0, 'sigma_x'] = 1
    gaussians_df.loc[gaussians_df['sigma_y'] == 0, 'sigma_y'] = 1
    gaussians_df.loc[gaussians_df['sigma_z'] == 0, 'sigma_z'] = 1
    
    # Extract Gaussian parameters from the DataFrame
    amplitudes = gaussians_df['amplitude'].values * 100
    mu_xs = gaussians_df['mu_x'].values.astype(int)
    mu_ys = gaussians_df['mu_y'].values.astype(int)
    mu_zs = gaussians_df['mu_z'].values.astype(int)
    sigma_xs = gaussians_df['sigma_x'].values
    sigma_ys = gaussians_df['sigma_y'].values
    sigma_zs = gaussians_df['sigma_z'].values

    for amplitude, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z in zip(amplitudes, mu_xs, mu_ys, mu_zs, sigma_xs, sigma_ys, sigma_zs):
        n_neighbors_x = int(3 * sigma_x) + 1
        n_neighbors_y = int(3 * sigma_y) + 1
        n_neighbors_z = int(3 * sigma_z) + 1

        z_range = np.arange(max(0, mu_z - n_neighbors_z), min(image_gaussians.shape[0], mu_z + n_neighbors_z + 1))
        y_range = np.arange(max(0, mu_y - n_neighbors_y), min(image_gaussians.shape[1], mu_y + n_neighbors_y + 1))
        x_range = np.arange(max(0, mu_x - n_neighbors_x), min(image_gaussians.shape[2], mu_x + n_neighbors_x + 1))

        zz, yy, xx = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        distances = (
            ((zz - mu_z) ** 2) / (2 * sigma_z ** 2) +
            ((yy - mu_y) ** 2) / (2 * sigma_y ** 2) +
            ((xx - mu_x) ** 2) / (2 * sigma_x ** 2)
        )
        gaussian_values = amplitude * np.exp(-distances)
        np.add.at(image_gaussians, (zz, yy, xx), gaussian_values)
    
    return image_gaussians
