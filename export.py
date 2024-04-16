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
import matplotlib.pyplot as plt
import os
import scipy
from datetime import datetime

class Export:
	def __init__(self,):
		pass


	def datafile(self, road_length, detection_dist, speed_limit, num_steps, p_slowdown, car_numb, density, cp_means, cp_stds, vel_means, vel_stds, flux_means, flux_stds, norm_vels, norm_stds):
		
		# Concatenate the arrays horizontally
		combined_array = np.hstack((car_numb.reshape(-1, 1), density.reshape(-1, 1),  cp_means.reshape(-1, 1), cp_stds.reshape(-1, 1), vel_means.reshape(-1, 1), vel_stds.reshape(-1, 1), flux_means.reshape(-1, 1), flux_stds.reshape(-1, 1), norm_vels.reshape(-1, 1), norm_stds.reshape(-1, 1)))
		
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

		filename = str(int(car_numb[0])) + ',' + str(detection_dist) + ',' + str(0) + ','+ str(road_length) + ',' + str(num_steps) + ',' + str(speed_limit) + ',' + str(p_slowdown) + f'_{timestamp}' + '.txt'

		headers = "bots, density, op mean, op std, speed mean, speed std, flux mean, flux std, speed norm mean, speed norm std"

		# Save the combined array as a text file
		np.savetxt(filename, combined_array, delimiter=',', header=headers)

	def plot(self, save_filename):
		# Save the plot in the same directory as the script
		if save_filename:
			script_directory = os.path.dirname(os.path.abspath(__file__))
			save_path = os.path.join(script_directory, save_filename)
			plt.savefig(save_path, bbox_inches='tight')

	def plot_survival_curves(all_bin_edges, all_survival_probabilities, labels):
		if len(all_bin_edges) > 1:
			for bin_edges, survival_probabilities, label in zip(all_bin_edges, all_survival_probabilities, labels):
				plt.loglog(bin_edges, survival_probabilities, marker='o', linestyle='-', label=label)
			plt.show()

	def save_data_to_csv(all_bin_edges, all_survival_probabilities, filename):
		all_bin_edges_array = np.array(all_bin_edges)
		all_survival_probabilities_array = np.array(all_survival_probabilities)
		data_to_save = np.vstack((all_bin_edges_array, all_survival_probabilities_array)).T
		np.savetxt(filename, data_to_save, delimiter=",", header="Value,Cumulative Probability", comments="")

	def save_lists_to_mat(self, *args, **kwargs):
		"""
		Save multiple lists to a MATLAB (.mat) file with a customized filename.

		Parameters:
		*args: Lists
			Variable number of lists to be saved in the MATLAB file.

		**kwargs: key-value pairs
			Additional parameters to customize the filename. These parameters will be
			included in the filename for easy identification of the simulation.

		Returns:
		None

		Example:
		list1 = [1, 2, 3, 4, 5]
		list2 = ['a', 'b', 'c', 'd', 'e']
		list3 = [0.1, 0.2, 0.3, 0.4, 0.5]

		save_lists_to_mat(list1, list2, list3, param1='value1', param2='value2')

		This will save the lists to a .mat file with a filename like
		"output_data_param1_value1_param2_value2.mat", where param1, param2 are the
		provided keyword arguments.

		The .mat file can be loaded in MATLAB using the load function.

		Note: Make sure to have the SciPy library installed (pip install scipy).
		"""

		# Get the directory of the current script
		script_directory = os.path.dirname(os.path.abspath(__file__))

		# Construct the full path for the .mat file with parameters in the name
		parameters_str = '_'.join([f"{key}_{value}" for key, value in kwargs.items()])
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		mat_file_path = os.path.join(script_directory, f'output_data_{parameters_str}_{timestamp}.mat')

		# Create a dictionary to store the lists with appropriate keys
		data = {}
		for i, lst in enumerate(args, start=1):
			key = f'list{i}'
			data[key] = lst

		# Save the data to the .mat file
		scipy.io.savemat(mat_file_path, data)
