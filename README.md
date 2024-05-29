# 3-D Time-Series Microscopy Image Analysis
## Pipeline for time-series microscopy data on Clathrin-Mediated Endocytosis

![main gif](https://github.com/Mdanishnadeem/Image-Analysis-Tracking/blob/main/main_image.gif)

## Overview: 
This project aims to analyze the dynamics of clathrin-mediated endocytosis, with a specific focus on the roles played by dynamin and actin across various membrane domains (apical, basal, and lateral). The dataset comprises 3D time-series data acquired using lattice light sheet microscopy coupled with fluorescence techniques.

Furthermore, this endeavor seeks to elucidate the distinct roles of proteins such as dynamin and actin in orchestrating endocytic events across the diverse membrane landscapes.

Lastly, an interactive dashboard is also developed to assist with viewing raw tracks (2-D projections) and features of each track for all channels. This would assist in manually identifying valid tracks. 


## Installation:

Ensure conda is installed before running the following steps. Details for installation can be found [here](https://docs.anaconda.com/free/miniconda/miniconda-install/)

```bash
git clone git@github.com:Mdanishnadeem/Image-Analysis-Tracking.git
conda create --name cme_pipeline python==3.10
conda activate cme_pipeline 
cd Image-Analysis-Tracking
pip install -r requirements.txt
```

## Dashboard:

Dashboard allows the user to interact with tracks using different parameters

The user can do the following: 
1. Select types of track (e.g. Channel 1 and Channel 3 positive)
2. Select the membrane region of the track (e.g. Apical)
3. Select the track number available from the above filtering criterion

    a. Select track number from dropdown 

    b. Go back and forth from the prev and next buttons 

4. Select the type of 2-D projection for the 3-D image 

    a. Maximum Intensity Projection 

    b. Max Z Slice (Z slice with max pixel sum)

    c. Total Z Sum (Sum across Z slices)

5. View intensity over time plot from the following options 

    a. Gaussian Peaks 

    b. Adjusted Voxel Sum (Adjusted for local background)

    c. Voxel Sum 

    d. Peak Pixel Value 

6. Select track as good or bad and also provide details for why a track is bad 
7. On the second page detailed track stats and 3-D graph for track movement can be seen 

![dashboard_home](https://github.com/Mdanishnadeem/Image-Analysis-Tracking/blob/main/misc/home_page.png)
![dashbord_second](https://github.com/Mdanishnadeem/Image-Analysis-Tracking/blob/main/misc/second_page.png)

## Main Functions: 
The [src](https://github.com/Mdanishnadeem/Image-Analysis-Tracking/tree/main/Final/src) folder contains all the source code for this project. You can explore this folder to gain a more detailed understanding of the project's implementation.

Feel free to browse through the files and directories to learn more about how the project works.

## How to Run: 
A detailed guide for the entire project and steps involved can be found in [manual](https://github.com/Mdanishnadeem/Image-Analysis-Tracking/blob/main/Image%20Analysis%20Pipeline%20Explained.docx)


## Acknowledgements:

This project takes assistance from the following:

1. [Pylattice](https://github.com/pylattice)
2. [LapTrack](https://github.com/yfukai/laptrack)
3. [Napari](https://napari.org/stable/)
4. [Dash & Plotly](https://dash.plotly.com/)