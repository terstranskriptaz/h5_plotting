# -*- coding: utf-8 -*-
"""
Created on Tue May 02 10:40:31 2023

@author: okilic
"""
import h5py
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


filename= "predictions.analysis.h5"

#*******************************************************************************************************************
#describe the dataset of .h5 file has in lists and define track data as locations of the animals
#prints all the top-level objects as nodes

with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape / track data ===")
print(locations.shape)
print()

print("===top-level objects===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()


#printing out location data shape which has four index;
#   number of frames,
#   the number of nodes in the skeleton,
#   for the x and y coordinates
#   the number of distinct animal identities which were found

frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

#decribing track output as locations with filling missing tracks informations definition
#Fills missing values independently along each dimension after the first

def fill_missing(Y, kind="linear"):
    #y is a 2D array and  input and
    # fills in missing values in each column of the array by interpolating between the surrounding non-missing values

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        # interp1d build a linear or cubic spline interpolant for each column, depending on the value of the kind parameter

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        # the function fills in any leading or trailing NaNs in each column
        # with the nearest non-NaN values, using the np.interp function

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

locations = fill_missing(locations)

#print (locations)


# Define constants for the body part indices
HEAD_INDEX = 0
THORAX_INDEX = 1
ABDO_INDEX = 2

# Extract body part locations for all frames
head_loc = locations[:, HEAD_INDEX, :, :]
thorax_loc = locations[:, THORAX_INDEX, :, :]
abdo_loc = locations[:, ABDO_INDEX, :, :]


# Set up the plot
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

# Plot the X and Y coordinates of the thorax for each animal
num_animals = thorax_loc.shape[1]
colors = sns.color_palette(n_colors=num_animals)
print("Number of animals: ", num_animals)
'''
# Print the X and Y coordinates for each animal
for i in range(num_animals):
    print(f"Animal {i}:")
    x = thorax_loc[:, i, 0]
    y = -1 * thorax_loc[:, i, 1]
    for j in range(len(x)):
        if np.isnan(x[j]) or np.isnan(y[j]):
            print(f"  Frame {j}: missing values")
        else:
            print(f"  Frame {j}: x={x[j]}, y={y[j]}")
'''
# Print the X and Y coordinates for each animal
print("X and Y coordinates for each animal")
for i in range(num_animals):
    x = thorax_loc[:, i, 0]
    y = -1 * thorax_loc[:, i, 1]
    print(f"Animal {i} X: {x}")
    print(f"Animal {i} Y: {y}")

plt.figure()
for i in range(num_animals):
    x = thorax_loc[:, i, 0]
    y = -1 * thorax_loc[:, i, 1]
    plt.plot(x, y, color=colors[i], label='Animal {}'.format(i))

plt.legend(loc='center right')
plt.title('Thorax locations')

# Plot the trajectories of each animal
plt.figure(figsize=(7,7))
for i in range(num_animals):
    x = thorax_loc[:, i, 0]
    y = -1 * thorax_loc[:, i, 1]
    plt.plot(x, y, color=colors[i], label='Animal {}'.format(i))

plt.legend()
plt.xlim(0, 1024)
plt.xticks([])

plt.ylim(0, 1024)
plt.yticks([])
plt.title('Thorax tracks')

plt.show()
