import numpy as np
import pandas as pd 
import napari
import os
import matplotlib.pyplot as plt 
import seaborn as sns 


def combine_dataframes(number_of_files: int ,channel: int, directory_path: str):
    '''
    The function takes in all pickle files of the dataframes returned from detection for each time frame 
    and combines them into a single large dataframe. The function is provided with a specific folder path where it
    goes and extracts all the pickle files. 
    
    Note: 
    The naming convention of the files is fixed to a specific naming convention. For example for channel 2 frame 0
    the name of the file should be df_c2_t0. This must be followed for this function to work properly. 
    
    Inputs: 
    1. number_of_files: type(int), This is equal to the number of time frames/files in the specified folder 
    2. file_path: type(int), the path of the folder where all the files for a specific channel is stored
    3. channel: type(str), The channel number (e.g 1,2,3)
    
    Returns: 
    1. combined_df: type(Dataframe), this is a combined dataframe for all detections for a single channel 
    '''
    
    # Create an empty list to store the individual DataFrames
    dataframes = []

    # Load pickle files, add "frame" column, and store DataFrames
    for i in range(number_of_files):

        file_name = f"df_c{channel}_t{i}.pkl"
        file_path = os.path.join(directory_path, file_name)

        # Load the DataFrame from the pickle file
        df = pd.read_pickle(file_path)

        # Extract the frame number from the file name
        frame = int(file_name.split("_t")[1].split(".")[0])

        # Add the "frame" column to the DataFrame
        df["frame"] = frame

        # Append the DataFrame to the list
        dataframes.append(df)

    # Concatenate the list of DataFrames into a single larger DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df



def box_whisker_plot(dataframe: pd.DataFrame, column_name: str):
    '''
    This function takes in a column from a dataframe and plots box and whisker plot for that column. 
    It also labels the descriptive stats and returns a dict of descriptive stats 

    Inputs: 
    1. dataframe: dataframe from which a column is going to be picked for box plot
    2. column_name: type(string) the column name for which we want to build the box plot for 

    Output: 
    1. prints box plot 
    2. descriptive_stats: type(dict) returns all the relevant descriptive stats
    '''
    
    #to store the upper and lower whisker edge
    limits = []

    # Plot the box and whisker plot
    plt.figure(figsize=(10, 6))
    boxplot = plt.boxplot(dataframe[column_name])

    # Add annotations for edges (minimum, 25th percentile, median, 75th percentile, maximum)
    edges = ['min', '25%', 'median', '75%', 'max']
    values = [item.get_ydata()[0] for item in boxplot['whiskers']]

    # Add annotations for caps
    for value in [item.get_ydata()[0] for item in boxplot['caps']]:
        plt.text(0.5, value,
                 f'Cap: {value:.2f}',
                 horizontalalignment='right',
                 verticalalignment='center',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")
                 )
        limits.append(value)

    # Add annotations for 25th percentile, mean, and 75th percentile
    for value, label in zip([dataframe[column_name].quantile(0.25),
                             dataframe[column_name].mean(),
                             dataframe[column_name].quantile(0.75)],
                            ['25th Percentile', 'Mean', '75th Percentile']):
        plt.text(1.5, value, f'{label}: {value:.2f}',horizontalalignment='left',
                       verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"))


    plt.xlabel('Spots')
    plt.ylabel(column_name)
    plt.title(f'Box and Whisker Plot of {column_name} of spots')

    plt.show()
    
    descriptive_stats = {'lower_whisker': limits[0], 'upper_whisker': limits[1], 'mean':dataframe[column_name].mean(), 
                         'lower_quantile':dataframe[column_name].quantile(0.25),
                         'upper_quantile':dataframe[column_name].quantile(0.75)}
    
    return descriptive_stats
    

def hist_plot(dataframe: pd.DataFrame, column_name: str, bin_size: int = 1, custom_xaxis: bool = False, 
             upper_xlimit: int = None, lower_xlimit: int = None):
    
    '''
    This function makes a histogram for frequency of each range. It also plots the mean value and the kde curve on the 
    same graph 

    Inputs: 
    1. dataframe: dataframe from which a column is going to be picked for box plot
    2. column_name: type(string) the column name for which we want to build the box plot for
    3. bin_size: type (int) the bin size for the histograms 
    4. custom_xaxis: type(bool) in case we want to restrict the x axis range this can be set to True. default False 
    BELOW TWO INPUTS ARE ONLY RELEVANT WHEN CUSTOM_XAXIS IS TRUE
    5. upper_xlimit: type(int) upper limit for x axis 
    6. upper_ylimit: type(int) upper limit for y axis 

    Outputs: 
    1. Outputs the graph
    2. bin_vals: type(dict), returns the max value, min value and bin size of the histogram 
    '''
    # Set the figure size
    plt.figure(figsize=(10, 6))
    
    max_val = int(dataframe[column_name].max())
    min_val = int(dataframe[column_name].min())
    bin_vals = {'max_val':max_val, 'min_val':min_val, 'bin_size': bin_size}
    # Create the histogram with seaborn
    ax = sns.histplot(dataframe[column_name], bins=range(min_val, max_val, bin_size), kde=True, color = 'blue',
                 kde_kws={'bw_method': 1}, line_kws = dict(color='green'), edgecolor='black', alpha=0.7)
    ax.get_lines()[0].set_color('green') 
    
    if custom_xaxis == True:
        ax.set_xlim(lower_xlimit, upper_xlimit)
        
    # Add a mean line
    mean_value = dataframe[column_name].mean()
    plt.axvline(x=mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')

    # Set labels and title
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title(f'{column_name} Histogram with Mean Line and KDE')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    return bin_vals
    
def plot_histogram_cutoffs(spots_df,amplitude_cutoff,sigmax_cutoff,sigmay_cutoff,sigmaz_cutoff,mu_upper_cutoff,sigma_upper_cutoff):

    # Assuming cleaned_spots_df is your DataFrame with columns 'Amplitude', 'sigma_x', 'sigma_y', 'sigma_z'

    # Set up subplots as a 2x2 grid
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Histogram for Amplitude with bins of size 50 starting from 180
    axes[0].hist(spots_df['amplitude'].dropna(), bins=100)
    axes[0].axvline(amplitude_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[0].set_title('Amplitude Histogram')

    # Histogram for sigma_x with bins of size 1 starting from 0
    axes[1].hist(spots_df['sigma_x'].dropna(), bins=range(0, int(spots_df['sigma_x'].max()) + 1, 1))
    axes[1].axvline(sigmax_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[1].set_xlim(0, 20)
    axes[1].set_title('Sigma_x Histogram')

    # Histogram for sigma_y with bins of size 1 starting from 0
    axes[2].hist(spots_df['sigma_y'].dropna(), bins=range(0, int(spots_df['sigma_y'].max()) + 1, 1))
    axes[2].axvline(sigmay_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[2].set_xlim(0, 20)
    axes[2].set_title('Sigma_y Histogram')

    # Histogram for sigma_z with bins of size 1 starting from 0
    axes[3].hist(spots_df['sigma_z'].dropna(), bins=range(0, int(spots_df['sigma_z'].max()) + 1, 1))
    axes[3].axvline(sigmaz_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[3].set_xlim(0, 20)
    axes[3].set_title('Sigma_z Histogram')

    # Histogram for mean_errors_mu with bins of size 0.01 starting from 0
    axes[4].hist(spots_df['mean_errors_mu'].dropna(), 100)
    axes[4].axvline(mu_upper_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[4].set_title('Mean Errors Mu Histogram')

    # Histogram for mean_errors_sigma with bins of size 0.01 starting from 0
    axes[5].hist(spots_df['mean_errors_sigma'].dropna(), bins=range(0, int(spots_df['mean_errors_sigma'].max()) + 1, 1))
    axes[5].axvline(sigma_upper_cutoff, color='r', linestyle='dashed', linewidth=1)
    axes[5].set_xlim(0, 20)
    axes[5].set_title('Mean Errors Sigma Histogram')


    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


    

