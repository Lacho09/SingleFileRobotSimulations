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
import matplotlib.cm as cm
import settings
import os
import logging

from computations import Computations
from ns_model import Simulation
from plot import Plot
from export import Export


logging.basicConfig(
	filename='log.txt',
	level=logging.INFO, 
	format='%(asctime)s - %(levelname)s - %(message)s'
)

def progress_bar(progress, total):
	
	percent = 100 * progress / total
	bar = '█' * int(percent) + '-' * (100 - int(percent))
	print(f'\r|{bar}| {percent:.2f}%', end='\r')

def ns_modified(road_diameter, detection_dist, speed, n, num_steps):
	"""
	Runs a modified simulation using the NS model.

	Parameters:
	- road_diameter: float
		Diameter of the road.
	- detection_dist: float
		Detection distance of each bot.
	- speed: float
		Speed of the bots.
	- n: int
		Number of bots.
	- num_steps: int
		Number of simulation steps.

	Returns:
	Tuple containing simulation results and computed distributions:
	- ordparam: float
		Order parameter of the simulation.
	- mean_speed: float
		Mean speed of the bots.
	- std_speed: float
		Standard deviation of the speed of the bots.
	- distribution_running_times: NumPy array
		Distribution of running times of the bots.
	- distribution_stopped_times: NumPy array
		Distribution of stopped times of the bots.
	- distribution_jammed_sizes: NumPy array
		Distribution of jammed sizes.
	- distribution_unjammed_sizes: NumPy array
		Distribution of unjammed sizes.
	- dist_jam_duration: NumPy array
		Duration of jams.
	- dist_jam_size: NumPy array
		Size of jams.
	"""
	
	# sim object from Simulation class
	sim = Simulation(num_steps, road_diameter, n, detection_dist, speed, 0.2)
	
	cal = Computations(sim)
	exp_sim = Export()
	
	# Initializes the simulation
	sim.initialize()
	# sim.show_positions()

	for step in range(num_steps):

		### Simulation rules -------------------------
		sim.check_superposition()
		sim.calculating_separation_distances()
		sim.acceleration()
		sim.braking()
		sim.random_slowdown()
		sim.update_positions()
		#--------------------------------------------
		
		### Computations in-step ----------------------------
		
		# Counts the number of stopped bots (with speed equal to 0).
		sb = cal.cars_stopped()
		
		# Binarizes the velocity of the bots (0 if speed is 0, and 1 if speed bigger than 0).
		cal.speed_binarization()
		
		# Binarizes the detection of the bots (0 if the detection_status is FALSE, and 1 if it is TRUE).
		cal.bot_detection_binarization()
		cal.step_detection_binarization()

		# Takes the speeds of all bots in a step and computes its mean and std
		cal.vel()
		#---------------------------------------------

		# sim.show_positions()
		# sim.show_speeds()

	### Computations in-simulation --------------------------------------------
	
	# From the binarized sequence of speeds computes the number of continuous steps the bots are stopped or running
	cal.dt_stopped()
	
	# From the binarized sequence of sensor status it finds the number of bots involved in a jam (all bots 
	# detecting the one in front of itself)
	cal.jam_sizes()

	# labels jams according to a connectivity matrix
	# labeled_jams = cal.label_connected_components()
	labeled_jams = cal.jam_id()

	# Computes the duration and size of jams
	dist_jam_duration, dist_jam_size = cal.jam_duration_and_size(labeled_jams)
	
	# Computes the flux in bots/timestep
	flux = cal.flux()

	# Average velocity and std
	mean_speed = np.mean(cal.speed['mean'])
	std_speed = np.std(cal.speed['mean'])

	# Computes the order parameter
	ordparam = cal.order_parameter()

	# ------------------------------------------------------

	# Plots -----------------------------------------------------
	# plot = Plot(exp_sim)
	# plot.histogram(plot.distribution_running_times)
	# plot.
	# plot.histogram_log_log(plot.distribution_stopped_times, 'hist_loglog_'f'{n,road_diameter,num_steps,detection_dist,speed}')
	# # plot.cfd()
	# plot.survival_semilog(plot.distribution_running_times, 'surv_semilog_'f'{n,road_diameter,num_steps,detection_dist,speed}')
	# plot.survival_loglog(plot.distribution_stopped_times, 'surv_loglog_'f'{n,road_diameter,num_steps,detection_dist,speed}')
	# -------------------------------------------------------

	return ordparam, mean_speed, std_speed, cal.distribution_running_times, cal.distribution_stopped_times, cal.distribution_jammed_sizes, cal.distribution_unjammed_sizes, dist_jam_duration, dist_jam_size


def collect_data(road_diameter, detection_dist, speed_limit, steps, p_slowdown, reps, exp, plotter):
	"""
	Collects data for simulation based on specified parameters.

	Parameters:
	- road_diameter: float
	- detection_dist: float
	- speed_limit: float
	- steps: int
	- p_slowdown: float
	- reps: int
	- exp: Exporting object
	- plotter: Plotting object

	Returns:
	Tuple of NumPy arrays representing collected data.
	"""
	num_simulations = len(settings.list_bots)
	
	ordparam_means = np.empty(num_simulations)
	ordparam_stds = np.zeros(num_simulations)
	velocity_means = np.empty(num_simulations)
	velocity_stds = np.empty(num_simulations)
	flux_means = np.empty(num_simulations)
	flux_stds = np.empty(num_simulations)
	bot_numbers = np.empty(num_simulations)
	densities = np.empty(num_simulations)

	all_jamdur_bin_edges = []
	all_jamdur_survival_probabilities = []

	for i, bot_number in enumerate(settings.list_bots):

		logging.info(f'Simulation parameters -> {bot_number, detection_dist, road_diameter, steps, speed_limit, p_slowdown, reps}')

		ordparam, mean_speed, std_speed, dist_run, dist_stop, dist_jamsizes, dist_unjamsizes, dist_jamdur, dist_clusterjam_size = ns_modified(road_diameter, detection_dist, speed_limit, bot_number, steps)

		# Exports parameters in a .mat file in the current directory
		exp.save_lists_to_mat(dist_jamdur, dist_clusterjam_size, c=str(bot_number), road_diameter=str(road_diameter), 
								detection_dist=str(detection_dist), speed_limit=str(speed_limit), steps=str(steps), 
								p_slowdown=str(p_slowdown))

		bot_numbers[i] = bot_number
		densities[i] = (detection_dist * bot_number) / road_diameter

		ordparam_means[i] = ordparam

		velocity_means[i] = mean_speed
		velocity_stds[i] = std_speed

		flux_means[i] = mean_speed * bot_number / road_diameter
		flux_stds[i] = std_speed * bot_number / road_diameter

		norm_velocities = velocity_means / speed_limit
		norm_stds = velocity_stds / speed_limit

		# jamdur_bins, jamdur_cfd = plotter.survival_loglog_interactive(dist_jamdur, color[i], f'{bot_number}')

		# all_jamdur_bin_edges.append(jamdur_bins)
		# all_jamdur_survival_probabilities.append(jamdur_cfd)

	return bot_numbers, densities, ordparam_means, ordparam_stds, velocity_means, velocity_stds, flux_means, flux_stds, norm_velocities, norm_stds, all_jamdur_bin_edges, all_jamdur_survival_probabilities

def run(road_diameter, detection_dist, speed_limit, steps, p_slowdown, reps):

	# Set your other parameters and initialize objects as needed
	exp = Export()
	plotter = Plot(exp)
	# color = ['blue', 'red', 'green', 'yellow', 'navy', 'cyan', 'fuchsia']

	N = round(road_diameter/detection_dist) ### amount of car to reach density equal 1
	logging.info(f'Maximum bots capacity -> {N}')

	car_numb, density, ordparam_means, ordparam_stds, vel_means, vel_stds, flux_means, flux_stds, norm_vels, norm_stds, all_jamdur_bin_edges, all_jamdur_survival_probabilities = collect_data(road_diameter, detection_dist, speed_limit, steps, p_slowdown, reps, exp, plotter)

	#plot_survival_curves(all_jamdur_bin_edges, all_jamdur_survival_probabilities, settings.labels)

	# Optionally, save data to CSV
	#save_data_to_csv(all_jamdur_bin_edges, all_jamdur_survival_probabilities, 'survival_loglog_jamdur.csv')

	exp.datafile(road_diameter, detection_dist, speed_limit, steps, p_slowdown, car_numb, density, ordparam_means, ordparam_stds, vel_means, vel_stds, flux_means, flux_stds, norm_vels, norm_stds)

def main():
	for road_diameter in settings.road_diameters:
		for detection_dist in settings.detection_dists:
			for speed in settings.speeds:
				for p_slowdown in settings.ps:
					run(road_diameter, detection_dist, speed, settings.steps, p_slowdown, settings.reps)

main()