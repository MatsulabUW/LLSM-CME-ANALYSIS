# now package it all into a function

import numpy as np
from skimage.feature import peak_local_max

def peak_local_max_3d(image,min_distance,threshold=0):

    """Find peaks in an 3D image as intensity array.
    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).
    If there are multiple local maxima with identical pixel intensities
    inside the region defined with `min_distance`,
    the coordinates of all such pixels are returned.

    ----------
    image : ndarray
        Input image.
    min_distance : int
        Minimum number of pixels separating peaks in a region of `2 *
        min_distance + 1` (i.e. peaks are separated by at least
        `min_distance`).
        To find the maximum number of peaks, use `min_distance=1`.

    Returns
    -------
    output : ndarray
        * If `indices = True`  : [[z1,x1,y1],[z2,x2,y2],...] coordinates of peaks.

    Notes
    -----
    The function relies on applying scikit image's 2D peak_local_max function
    to generate a candidate list of 3D maxima which then get elliminated in a
    subsequent step to fulfill the min_distance criterion.


    """

    ######### setup
    # make an array of zeros
    accumulator = np.zeros(image.shape)

    # accumulator for the coordinates and the coordinate intensities
    coordinateAccumulator = []

    ######### 2D
    # find all maxima in every 2D slice of the image
    for iz in range(0,image.shape[0]):

        #finds the local max in each z. (e.g z=0, z=1 and so on)
        #coordinates contains x,y for that z 
        coordinates=peak_local_max(image[iz],min_distance=min_distance)

        #write the max values into the accumulator at the right positions
        for coord in coordinates:
            ##coordValue contains the intensity(that can be seen in fiji as well) for that z-slice and x,y pair
            coordValue = image[iz][coord[0],coord[1]]
            ##accumulator contains (z,x,y) as the index and the intensities for it as value
            accumulator[iz,coord[0],coord[1]] = coordValue
            ##coordinate accumulator contains (z,x,y) and the intensity for that pair
            coordinateAccumulator.append([np.array([iz,coord[0],coord[1]]),coordValue])




    ######### 3D
    # Elliminate all that are too close together


    for maxCandidate in coordinateAccumulator:
        maxCandidate_z = maxCandidate[0][0]
        maxCandidate_x = maxCandidate[0][1]
        maxCandidate_y = maxCandidate[0][2]

        maxCandidate_value = maxCandidate[1]
        windowSizeHalf = int(min_distance/2)
        #print(windowSizeHalf)

#        print(maxCandidate_x-windowSizeHalf)
#        prnt(windowSizeHalf)
        from_x = max(0,maxCandidate_x-windowSizeHalf)
        # to_x = min(image.shape[1],maxCandidate_x+windowSizeHalf) ##Issue here 
        to_x = min(image.shape[1],maxCandidate_x+windowSizeHalf+1)
        #print(f'the image.shape[1] is {image.shape[1]} from_x is {from_x} and to_x is {to_x}')

        from_y = max(0,maxCandidate_y-windowSizeHalf)
        # to_y = min(image.shape[2],maxCandidate_y+windowSizeHalf) #Issue here
        to_y = min(image.shape[2],maxCandidate_y+windowSizeHalf+1)
        #print(f'the image.shape[2] is {image.shape[2]} from_y is {from_y} and to_y is {to_y}')

        from_z = max(0,maxCandidate_z-windowSizeHalf)
        # to_z = min(image.shape[0],maxCandidate_z+windowSizeHalf) #Issue here
        to_z = min(image.shape[0],maxCandidate_z+windowSizeHalf+1)
        #print(f'the image.shape[0] is {image.shape[0]} from_z is {from_z} and to_z is {to_z}')

        try:
            if(maxCandidate_value < threshold):
                accumulator[maxCandidate_z,maxCandidate_x,maxCandidate_y] = 0

            if(maxCandidate_value < np.amax(accumulator[from_z:to_z,from_x:to_x,from_y:to_y])):
                ##np.amax finds the max value returned from the accumulator. accumulator returns all local 
                ##maximas found within the range defined using from and to. The from and to range is found by 
                ##using the half window size in positive and negative direction
                #print("test")
                accumulator[maxCandidate_z,maxCandidate_x,maxCandidate_y] = 0
        except ValueError:  #raised if `y` is empty.
            pass
    ########## output
    #
    result = np.transpose(np.nonzero(accumulator))
    ##result is a numpy array which contains only non zero values and their indexes
    return result
##the return format of result is (z,x,y)
