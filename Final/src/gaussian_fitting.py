from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

from scipy.optimize import curve_fit
import numpy as np

def gaussian_1d_output(x,a,x0,sigma):
    """
    Computes the 1D Gaussian function.

    Parameters
    ----------
    x : np.ndarray
        The input array.
    a : float
        The amplitude of the Gaussian.
    x0 : float
        The mean (center) of the Gaussian.
    sigma : float
        The standard deviation (spread) of the Gaussian.

    Returns
    -------
    np.ndarray
        The computed Gaussian function values.
    """
    temp = a*np.exp(-(x-x0)**2/(2*sigma**2))

    return temp

def fit_gaussian(image, center, sigmas, width_parameters):
    """
    Fits a 1D Gaussian to the x, y, and z dimensions of a given 3D image around a specified center.

    Parameters
    ----------
    image : np.ndarray
        The 3D image array.
    center : tuple
        The center coordinates for the Gaussian fitting.
    sigmas : list
        The list of sigma values for the Gaussian in z, y, and x dimensions.
    width_parameters : float
        The width parameter for the Gaussian fitting bounds.

    Returns
    -------
    tuple
        A tuple containing the optimal parameters for the Gaussian and the parameters for each dimension.
    """
    try:
        x_center = center[1]
        y_center = center[2]
        z_center = center[0]

        peak =image[z_center,x_center,y_center]

        x_sigma = sigmas[1]
        y_sigma = sigmas[2]
        z_sigma = sigmas[0]

        from_x = max(0,x_center-x_sigma)
        to_x = min(image.shape[1],x_center+x_sigma)

        from_y = max(0,y_center-y_sigma)
        to_y = min(image.shape[2],y_center+y_sigma)

        from_z = max(0,z_center-z_sigma)
        to_z = min(image.shape[0],z_center+z_sigma)

        x_range = np.arange(from_x,to_x)
        y_range = np.arange(from_y,to_y)
        z_range = np.arange(from_z,to_z)

        x_data = np.array(image[z_center,x_range,y_center])
        y_data = np.array(image[z_center,x_center,y_range])
        z_data = np.array(image[z_range,x_center,y_center])

        x_popt, x_pcov = curve_fit(gaussian_1d_output, x_range, x_data, bounds = ([peak-width_parameters,x_center-width_parameters,-np.inf],[peak+width_parameters,x_center+width_parameters,np.inf]))
        y_popt, y_pcov = curve_fit(gaussian_1d_output, y_range, y_data, bounds = ([peak-width_parameters,y_center-width_parameters,-np.inf],[peak+width_parameters,y_center+width_parameters,np.inf]))
        z_popt, z_pcov = curve_fit(gaussian_1d_output, z_range, z_data, bounds = ([peak-width_parameters,z_center-width_parameters,-np.inf],[peak+width_parameters,z_center+width_parameters,np.inf]))

        optimal_parameters_each_dimension = []
        optimal_parameters_each_dimension.append(z_popt)
        optimal_parameters_each_dimension.append(x_popt)
        optimal_parameters_each_dimension.append(y_popt)

        optimal_parameter_one_gaussian = []
        mean_amplitude = np.mean([x_popt[0],y_popt[0],z_popt[0]])
        center = [z_popt[1],x_popt[1],y_popt[1]]
        sigmas = [z_popt[2],x_popt[2],y_popt[2]]
        optimal_parameter_one_gaussian.append(mean_amplitude)
        optimal_parameter_one_gaussian.append(center)
        optimal_parameter_one_gaussian.append(sigmas)

        return optimal_parameter_one_gaussian, optimal_parameters_each_dimension

    except:
        return -1, -1


def fit_multiple_gaussians(image,centers,sigmas,width_parameters):
    """
    Fits multiple 1D Gaussians to the x, y, and z dimensions of a given 3D image at specified centers.

    Parameters
    ----------
    image : np.ndarray
        The 3D image array.
    centers : list
        The list of center coordinates for the Gaussian fittings.
    sigmas : list
        The list of sigma values for the Gaussians.
    width_parameters : float
        The width parameter for the Gaussian fitting bounds.

    Returns
    -------
    tuple
        A tuple containing the net Gaussian parameters and the individual Gaussian parameters for each dimension.
    """
    net_gaussians = []
    individual_gaussians = []
    i = 0

    roundedPercent = 0
    for i in range(0,len(centers)):
        if roundedPercent != int(10*i/len(centers)):
            roundedPercent = int(10*i/len(centers))
            print("{}%({} of {})".format(10*roundedPercent,i,len(centers)))

        one_gaussian, each_dimension_gaussians = fit_gaussian(image, centers[i], sigmas[i], width_parameters)

        net_gaussians.append(one_gaussian)

        individual_gaussians.append(each_dimension_gaussians)
        i += 1
        
    print("{}%({} of {})".format(100,len(centers),len(centers)))
    return net_gaussians, individual_gaussians



def check_fitting_error(image,maximas,net_gaussians,sigmas_guesses):
    """
    Checks the fitting error for the Gaussian fittings.

    Parameters
    ----------
    image : np.ndarray
        The 3D image array.
    maximas : list
        The list of maxima coordinates.
    net_gaussians : list
        The list of net Gaussian parameters.
    sigmas_guesses : list
        The list of sigma values for the Gaussians.

    Returns
    -------
    tuple
        A tuple containing the list of absolute errors and the indices of maximas where the fitting did not succeed.
    """
    absolute_errors = []
    counter_fit = 0
    counter_not_fit = 0
    index_of_maximas = []

    for i in range(len(maximas)):
        temp_gaussian = net_gaussians[i]
        if temp_gaussian != -1:

            temp_absolute_error = []


            temp_maxima = maximas[i]
            temp_sigmas = sigmas_guesses[i]
            mean_absolute_error_amplitude = np.abs(temp_gaussian[0] - image[temp_maxima[0],temp_maxima[1],temp_maxima[2]])
            mean_absolute_error_mean = [np.abs(temp_maxima[0]-temp_gaussian[1][0]),np.abs(temp_maxima[1]-temp_gaussian[1][1]),np.abs(temp_maxima[2]-temp_gaussian[1][2])]
            mean_absolute_error_sigmas = [np.abs(temp_sigmas[0]-temp_gaussian[2][0]),np.abs(temp_sigmas[1]-temp_gaussian[2][1]),np.abs(temp_sigmas[2]-temp_gaussian[2][2])]

            temp_absolute_error.append(mean_absolute_error_amplitude)
            temp_absolute_error.append(mean_absolute_error_mean)
            temp_absolute_error.append(mean_absolute_error_sigmas)

            absolute_errors.append(temp_absolute_error)
            counter_fit += 1
        else:
            print('the gaussian did not fit')
            counter_not_fit += 1 
            index_of_maximas.append(i)

    print(f'the number of times the gaussian fitting worked was {counter_fit} and the number of times the gaussian did not fit was {counter_not_fit}')
    return absolute_errors, index_of_maximas
