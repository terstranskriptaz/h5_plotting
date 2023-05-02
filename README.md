# h5_plotting

This code reads data from an HDF5 file that contains predictions of an object tracking algorithm. The data includes the locations of animals in each frame of a video. The code extracts the location data for the head, thorax, and abdomen of each animal and plots the X and Y coordinates of the thorax for each animal.

Here is a step-by-step breakdown of the code:

The necessary packages are imported: h5py, numpy, scipy.interpolate, seaborn, matplotlib.
The name of the HDF5 file is stored in the filename variable.
The contents of the HDF5 file are read and stored in the following variables:
dset_names: a list of the names of all datasets in the file
locations: an array of shape (frame_count, node_count, 2, instance_count) that stores the X and Y coordinates of each animal's body parts in each frame
node_names: a list of the names of all nodes in the HDF5 file
The shapes of the locations array are printed to the console.
A function called fill_missing is defined. This function takes a 2D array Y and fills in any missing values in each column of the array by interpolating between the surrounding non-missing values. The function returns the filled-in array.
The fill_missing function is called on the locations array to fill in any missing values.
Constants are defined to represent the indices of the head, thorax, and abdomen in the locations array.
The X and Y coordinates of the thorax for each animal are extracted from the locations array and stored in thorax_loc.
A plot is set up using seaborn and matplotlib.
The X and Y coordinates of the thorax for each animal are plotted in a separate subplot.
The trajectories of each animal are plotted in another subplot.
