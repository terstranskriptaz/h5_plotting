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



filename= "first_export.h5"

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
#insect_num = locations.shape[3]
#print("The number of termites: ", insect_num)
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

HEAD_INDEX = 0
THORAX_INDEX = 1
ABDO_INDEX = 2

head_loc = locations[:, HEAD_INDEX, :, :]
thorax_loc = locations[:, THORAX_INDEX, :, :]
abdo_loc = locations[:, ABDO_INDEX, :, :]

# Set up the plot
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]

# Plot the X and Y coordinates of the thorax for each animal
#num_animals = thorax_loc.shape[1]
num_animals = locations.shape[3]
colors = sns.color_palette(n_colors=num_animals)
print("Number of animals: ", num_animals)

# Set up the plot
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,15]

# Plot the X and Y coordinates of the thorax for each animal
num_animals = locations.shape[3]
colors = sns.color_palette(n_colors=num_animals)
print("Number of animals: ", num_animals)

fig, ax = plt.subplots()

for i in range(num_animals):
    x = locations[:, THORAX_INDEX, 0, i]
    y = locations[:, THORAX_INDEX, 1, i]
    ax.plot(x, y, color=colors[i], linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title("Thorax tracks")

plt.show()

# Set up the plot
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [20,20]

# Plot the X and Y coordinates of the thorax for each animal
num_animals = locations.shape[3]
colors = sns.color_palette(n_colors=num_animals)

# Plot every animal's locations over time
fig, axs = plt.subplots(num_animals, 1, sharex=True, sharey=True)
fig.suptitle('Animal Locations over Time')
for i in range(num_animals):
    axs[i].plot(locations[:, THORAX_INDEX, 0, i], locations[:, THORAX_INDEX, 1, i], color=colors[i])
    axs[i].set_ylabel(f'Animal {i+1}')
axs[-1].set_xlabel('Time (frames)')

plt.show()

# Set up the plot
sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [30, 20]

# Plot every animal's locations over time
num_animals = locations.shape[3]
colors = sns.color_palette(n_colors=num_animals)

fig, axs = plt.subplots(3, num_animals, sharex=True, sharey=True)
fig.suptitle('Animal Locations over Time')

for i in range(num_animals):
    axs[0, i].plot(head_loc[:, 0, i], head_loc[:, 1, i], color=colors[i])
    axs[0, i].set_ylabel(f'Animal {i+1}')
    axs[0, i].set_title('H')

    axs[1, i].plot(thorax_loc[:, 0, i], thorax_loc[:, 1, i], color=colors[i])
    #axs[1, i].set_ylabel(f'Animal {i+1}')
    axs[1, i].set_title('T')

    axs[2, i].plot(abdo_loc[:, 0, i], abdo_loc[:, 1, i], color=colors[i])
    #axs[2, i].set_ylabel(f'Animal {i+1}')
    axs[2, i].set_title('A')

axs[-1, 0].set_xlabel('X')
axs[-1, 0].set_ylabel('Y')

plt.show()

#distance line plot

# Generate some sample data
#n_ter = num_animals
n_steps = 100
data = np.random.rand(num_animals, n_steps)

# Create a line plot with a different color for each termite
colors = plt.cm.jet(np.linspace(0, 1, num_animals))
for i in range(num_animals):
    plt.plot(data[i], color=colors[i], linewidth=1)

# Add axis labels and a title
plt.xlabel('Time (steps)')
plt.ylabel('Distance traveled')
plt.title('Termite movement over time')

# Show the plot
plt.show()



