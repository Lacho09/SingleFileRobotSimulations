"""
MIT License

Copyright (c) 2024 Laciel Alonso Llanes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Contact Information:
# Author: Laciel Alonso Llanes
# Contact: laciel.alonso@gmail.com
# GitHub: https://github.com/LacielAlonsoLlanes

import numpy as np
from skimage import measure
from skimage import color
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from scipy.ndimage import label
from scipy.io import savemat
import logging

class Bot:
	def __init__(self):
		self.position = 0
		self.positions_seq = []
		self.speed = 0
		self.bin_speed_seq = []
		self.bin_detection_seq = []
		self.detection_status = False
		self.distance = 0
		self.total_steps_stopped = 0

class Computations:
	def __init__(self, simulation):
		self.bots = simulation.bots
		self.bot_number = simulation.num_cars
		self.sim_steps = simulation.num_steps
		self.road_length = simulation.road_length

		self.op = 0
		self.distribution_stopped_times = np.empty(0)
		self.distribution_running_times = np.empty(0)
		self.distribution_jammed_sizes = np.empty(0)
		self.distribution_unjammed_sizes = np.empty(0)

		self.dt = np.dtype([('mean', np.float64), ('std', np.float64)])
		self.speed = np.empty(0,dtype=self.dt)
		# self.flux = np.empty(0,dtype=self.dt)

		self.bin_detection_sim_seq = []

	def progress_bar(self, progress, total, bar_id):

		# print(bar_id + '\n')
		try:
			percent = 100 * progress / total

			bar = '█' * int(percent) + '-' * (100 - int(percent))

			print(f'\r|{bar}| {percent:.2f}%', end='\r')
		except:
			print('Problems to show progress bar')

	def cars_stopped(self):
		'''
		Counts the number of stopped bots (with speed equal to 0). 
		Intended to act on step values.
		'''

		counter = 0
		for bot in self.bots:
			if bot.speed == 0:
				counter += 1

		return counter
	
	def speed_binarization(self):
		'''
		Binarizes the velocity of the bots (0 if speed is 0, and 1 if speed bigger than 0).
		Appends the values in a sequence.
		Intended to act on step values
		'''

		for bot in self.bots:
			if bot.speed == 0:
				bot.bin_speed_seq.append(0)
			else:
				bot.bin_speed_seq.append(1)

	def bot_detection_binarization(self):
		'''
		Binarizes the detection of the bots (0 if the detection_status is FALSE, and 1 if it is TRUE).
		Saves the detection status in bot object
		'''
		
		for bot in self.bots:
			if bot.detection_status == False:
				bot.bin_detection_seq.append(0)
			else:
				bot.bin_detection_seq.append(1)

	def step_detection_binarization(self):
		'''
		Binarizes the detection of the bots (0 if the detection_status is FALSE, and 1 if it is TRUE).
		Appends the values in a sequence for each bot.
		Intended to act on step values
		'''
		bin_detection_step_seq = []
		for bot in self.bots:
			if bot.detection_status == False:
				bin_detection_step_seq.append(0)
			else:
				bin_detection_step_seq.append(1)

		self.bin_detection_sim_seq.append(bin_detection_step_seq)

	def export_mat_file(self):

		bin_detection_sim_array = np.array(self.bin_detection_sim_seq)
		# Save the NumPy array to a MATLAB .mat file
		savemat('jams.mat', {'numpy_array': bin_detection_sim_array})

	def dt_stopped(self):
		'''
		From the binarized sequence of speeds computes the number of continuous steps the bots are stopped or running 
		to analize the distribution.
		Intended to act on simulation values.
		'''

		for bot in self.bots:

			# Converts the list to a numpy array
			array = np.array(bot.bin_speed_seq)

			# Searches the indexes where the array changes value
			changes = np.where(array[:-1] != array[1:])[0] + 1
			# print(changes)

			changes = np.insert(changes, 0, 0)
			changes = np.append(changes, len(array))

			# Calculates the length of sequences of zeros
			longitudes = np.diff(changes)

			# As the zeros and ones alternate, we select only the lengths of the sequences of zeros.
			longitudes_ceros = longitudes[::2] if array[0] == 0 else longitudes[1::2]
			longitudes_unos = longitudes[::2] if array[0] == 1 else longitudes[1::2]
			# print(longitudes_ceros)

			# print(array)

			self.distribution_stopped_times = np.append(self.distribution_stopped_times, longitudes_ceros)
			self.distribution_running_times = np.append(self.distribution_running_times, longitudes_unos)
		
		# print(self.distribution_stopped_times, self.distribution_running_times)

	def jam_sizes(self):
		'''
		From the binarized sequence of sensor status it computes the number of bots involved in a jam (all bots 
		detecting the one in front itself) to analize the distribution.
		Intended to act on simulation values.
		'''

		for bot in self.bots:

			# Converts the list to a numpy array
			array = np.array(bot.bin_detection_seq)

			# Searches the indexes where the array changes value
			changes = np.where(array[:-1] != array[1:])[0] + 1
			# print(changes)

			changes = np.insert(changes, 0, 0)
			changes = np.append(changes, len(array))

			# Calculates the length of sequences of zeros
			longitudes = np.diff(changes)

			# As the zeros and ones alternate, we select only the lengths of the sequences of zeros.
			longitudes_ceros = longitudes[::2] if array[0] == 0 else longitudes[1::2]
			longitudes_unos = longitudes[::2] if array[0] == 1 else longitudes[1::2]
			# print(longitudes_ceros)

			# print(array)

			self.distribution_unjammed_sizes = np.append(self.distribution_unjammed_sizes, longitudes_ceros)
			self.distribution_jammed_sizes = np.append(self.distribution_jammed_sizes, longitudes_unos)
		
		# print(self.distribution_stopped_times, self.distribution_running_times)

	def order_parameter(self):
		'''
		Computes the order parameter, which the ratio of the steps the bots are stopped over the total steps of the simulation.
		To act on simulation values
		'''

		total_steps_stopped = 0
		for bot in self.bots:
			total_steps_stopped += bot.total_steps_stopped

		self.op = total_steps_stopped / (self.bot_number * self.sim_steps)

		logging.info(f'Order parameter: {self.op}')

		return self.op
	
	def mean_and_std(self, x):

		mean = np.mean(x)
		std = np.std(x)

		return mean, std

	def vel(self):
		'''
		Takes the speeds of all bots in a step and computes its mean and std. Those values are then saved into an array. 
		The array has a data type that allows it to be call speed['mean'] or speed['std'].
		To act on step values'''

		velocities = np.empty(0)

		for bot in self.bots:
			velocities = np.append(velocities, bot.speed)
		
		mean_vel, std_vel = self.mean_and_std(velocities)
		new_element = mean_vel, std_vel

		new_array = np.array([new_element], dtype=self.dt)

		self.speed = np.append(self.speed, new_array)
		# print(np.mean(self.speed['mean']))

		# return mean_vel, std_vel

	def flux(self):
		'''
		Computes the flux in bots/timestep
		'''

		mean_speed = np.mean(self.speed['mean'])
		mean_flux = mean_speed * self.bot_number / self.road_length

		logging.info(f'Mean flux: {mean_flux}')

		return mean_flux

	def jam_duration_and_size(self, labeled_array):
		'''
		Computes the duration and size of jams. Both measurements are made from a labeled array.
		The duration of the jam is considered as the number of rows involved in each cluster. The vertical direction 
		of the array corresponds to the time. The size is computed as the number of bots involved in the cluster, 
		considering each bot only once, i.e., if a bot that was already in the jam goes out and gets in while the jam 
		still exist, the bot is counted only once. This means that the maximum number of bots in a jam cannot be 
		bigger than the number of bots in the simulation.
		'''

		# Find unique cluster labels excluding 0
		unique_labels = np.unique(labeled_array[labeled_array != 0])

		# Create a zeroed array to store the duration of jam for each cluster
		cluster_durations = np.zeros(len(unique_labels), dtype=int)

		# Create a zeroed array to store the size of jam for each cluster
		cluster_sizes = np.zeros(len(unique_labels), dtype=int)

		# Iterate over each unique label
		for i, label in enumerate(unique_labels):

			# Find the row indices where the label occurs
			label_indices = np.where(labeled_array == label)
			
			# Calculate the duration of jam for the current label
			duration = np.max(label_indices[0]) - np.min(label_indices[0])

			# Store the duration in the array
			cluster_durations[i] = duration
			
			# Get the unique column IDs involved in the labeled cluster
			unique_columns = np.unique(label_indices[1])

			# Store the duration in the array
			cluster_sizes[i] = len(unique_columns)
		
		# jam_sizes_matlab = np.array(cluster_sizes)
		# savemat('jams_sizes.mat', {'numpy_array': jam_sizes_matlab})

		# jam_duration_matlab = np.array(cluster_durations)
		# savemat('jam_duration.mat', {'numpy_array': jam_duration_matlab})
		
		# Print the result
		# for i, duration in enumerate(cluster_durations):
		# 	print(f"Cluster {unique_labels[i]}: Duration of jam = {duration} rows")

		return cluster_durations, cluster_sizes

	def find_connected_cluster(self, py, px, labels, connected_steps):
		for conta in range(1, connected_steps + 1):
			mask = (py - conta >= 0) & (px + 1 < labels.shape[1])
			if np.any(mask):
				pix_exp = labels[py - conta, px + 1]
				if pix_exp > 0:
					return pix_exp
		return 0

	def merge_clusters_numpy(self, labels, connected_steps, bot_number):
		#TODO: Comentar mejor, y organizar

		"""
		Merge clusters based on specified vertical connections and relabel them accordingly using NumPy array operations.

		Parameters:
			labels (numpy.ndarray): Array with clusters identified from a horizontal concatenation of the original binary image.
			connected_steps (int): Steps to connect clusters vertically. Vertical direction is associated with time.
			bot_number (int): A parameter used in the condition calculations.

		Returns:
			numpy.ndarray: Updated array with merged and relabeled clusters.

		Note:
			The merging process involves connecting clusters that are connected in the vertical direction, associating them
			based on specified conditions. The method efficiently uses NumPy array operations and boolean masks for processing.

		Implementation Details:
			- Iterates over each cluster_id from 1 to the maximum cluster number.
			- Creates a binary mask for the current cluster_id in the 'labels' array.
			- Checks conditions for pixels above and to the right and finds connected clusters.
			- Updates the labels for the entire cluster based on connected clusters.
			- Provides progress updates using a progress bar.

		Logging:
			- Logs cluster analysis information using 'logging.info'.
			- Logs warnings for exceptions during the process using 'logging.warning'.

		Example:
			# Example usage of the method
			updated_labels = merge_clusters_numpy(labels_array, 3, 5)

		"""
		cluster_number = np.max(labels)
		
		logging.info(f'cluster analysis - {cluster_number} clusters')
		
		self.progress_bar(0, cluster_number, 'clusters')

		# Loop through each cluster_id from 1 to cluster_number
		for cluster_id in range(1, cluster_number + 1):
			# Create a binary mask for the current cluster_id in the 'labels' array
			mask = labels == cluster_id
			# Get the y, x coordinates of the pixels with cluster_id
			y, x = np.where(mask)
			# # Calculate the number of pixels in the current cluster
			# NumPix = len(x)

			# Check conditions for pixels above and to the right
			condi_up_right = (y >= connected_steps) & (x < 2 * bot_number - 1)
			# condi_up_right is a boolean mask that is True for pixels that satisfy both conditions (above or on the same 
			# row as connected_steps and to the right of 2 * bot_number - 1 column)
			
			# Get the indices of pixels that meet the conditions
			indices_up_right = np.where(condi_up_right)[0]

			# Loop through the pixels that meet the conditions
			for counter in indices_up_right:
				# Get the y, x coordinates of the current pixel
				py, px = y[counter], x[counter]
				# Find the connected cluster_id for the current pixel
				cluscon = self.find_connected_cluster(py, px, labels, connected_steps)

				# If a connected cluster_id is found (not 0), update the labels for the entire cluster
				if cluscon != 0:
					labels[mask] = cluscon

			self.progress_bar(cluster_id, cluster_number, 'clusters')
			
		try:
			# Check conditions for arriba
			condi_up = (y >= connected_steps) & (x <= 2 * bot_number - 1)
			indices_up = np.where(condi_up)[0]

			logging.info(f'indices up analysis - {len(indices_up)} indices')
			self.progress_bar(0, len(indices_up), 'indices up')

			for counter in indices_up:
				py, px = y[counter], x[counter]
				cluscon = self.find_connected_cluster(py, px, labels, connected_steps)
				
				if cluscon != 0:
					labels[mask] = cluscon

				self.progress_bar(counter+1, len(indices_up), 'indices up')
		except:
			logging.warning('condi_up exception ')

		return labels

	def label_connected_components(self):

		matrix = self.bin_detection_sim_seq
		
		rows, cols = len(matrix), len(matrix[0])
		
		labels = np.zeros_like(matrix)
		current_label = 0
		
		self.progress_bar(0, rows-1, 'rows')

		# First pass: Assign labels row by row
		for i in range(rows):

			self.progress_bar(i, rows-1, 'rows')

			for j in range(cols):

				if matrix[i][j] == 1:
					neighbors = []

					# look back
					if j > 0 and matrix[i][j - 1] == 1:
						neighbors.append(labels[i][j - 1])

					# look up right
					if j < len(matrix[0])-1 and i > 0 and matrix[i - 1][j + 1] == 1:
						neighbors.append(labels[i - 1][j + 1])

					# look up up right
					if j < len(matrix[0])-1 and i > 1 and matrix[i - 2][j + 1] == 1:
						neighbors.append(labels[i - 2][j + 1])

					# look up
					if i > 0 and matrix[i - 1][j] == 1:
						neighbors.append(labels[i - 1][j])

					# periodic conditions
					if j == len(matrix[0]) - 1 and i > 0 and matrix[i-1][0] == 1:
						neighbors.append(labels[i-1][0])

					if j == len(matrix[0]) - 1 and i > 1 and matrix[i-2][0] == 1:
						neighbors.append(labels[i-2][0])

					# print(neighbors)

					if not neighbors:
						current_label += 1
						labels[i][j] = current_label
					else:
						labels[i][j] = min(neighbors)
						for neigh in neighbors:
							labels[labels==neigh] = min(neighbors)

		return labels

	def jam_id(self):
		'''
		Concatenates the array and searches for clusters
		'''
		
		custom_connectivity = np.array([
			[False, True, True],
			[True, True, True],
			[True, True, False]
		], dtype=bool)

		# Concatenate the array twice, horizontally
		result_array = np.concatenate((self.bin_detection_sim_seq, self.bin_detection_sim_seq), axis=1)

		# Find clusters with the custom connectivity and label them
		labels, num_clusters = label(result_array, structure=custom_connectivity)

		# This connectivy matrix is used to connect clusters already identified by label()
		#  Connectivity matrix:
		#  [ 0 0 1
		#    0 1 1
		#    1 x 1
		#    0 1 0
		#    0 0 0 ]

		# Number of timesteps to consider searching for connections among clusters
		connected_steps = 2 
		
		# Merge clusters based on the connectivity matrix and relabel them accordingly
		merged_labels = self.merge_clusters_numpy(labels, connected_steps, self.bot_number)

		# Get properties of labeled regions, including coordinates
		merged_regions = measure.regionprops(merged_labels)

		# the number of cols in the arrray is the number of bots in the simulation
		cols = self.bot_number

		# Filter regions: clusters connected through the border of the concatenation
		filtered_regions = [region for region in merged_regions if any(coord[1] == cols-1 for coord in region.coords) 
							and any(coord[1] == cols for coord in region.coords)]

		# Cut array of labels by the middle (as it was before concatenation)
		crop_labels = merged_labels[:,:cols].copy()
		
		# show some info of the simulation status
		logging.info(f'iteration over filtered regions - {len(filtered_regions)} regions')

		# progress bar
		self.progress_bar(0, len(filtered_regions), 'regions')

		counter_regions = 0
		# Iterate over filtered regions
		for region in filtered_regions:
			counter_regions += 1

			# print(region.coords[0][0], region.coords[0][1] % cols)

			# Get the value of the cluster being analyzed
			original_cluster_value = crop_labels[region.coords[0][0], region.coords[0][1] % cols]
			# print('original cluster value', original_cluster_value)

			# Substitutes all values equal to original_cluster_value in crop_labels by the cluster_id from merged_labels
			crop_labels[crop_labels == original_cluster_value] = merged_labels[region.coords[0][0], region.coords[0][1]]

			self.progress_bar(counter_regions, len(filtered_regions), 'regions')

		return crop_labels


