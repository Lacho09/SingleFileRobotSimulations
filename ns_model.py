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
from computations import Bot
import logging

class Simulation:
	def __init__(self, num_steps, road_length, num_cars, detection_dist, speed_limit, p_slowdown):
		self.num_steps = num_steps
		self.road_length = road_length
		self.num_cars = num_cars
		self.speed_limit = speed_limit
		self.p_slowdown = p_slowdown
		self.detection_dist = detection_dist

		self.step = 0
		self.detection_dists = np.zeros(self.num_cars)
		
		self.bin_detection_step_seq = []

		# Bot is a class defined in bot.py. It has all the parameters that every bot need to have
		self.bots = [Bot() for _ in range(num_cars)]

		# self.reaction_time = reaction_time
		# self.steps_stopped = np.empty(0)

	def initialize(self):
		'''
		Defines the initial positions of the bots. Take into account that it uses self.num_cars+1 instead of 
		self.num_cars because without it, and the periodic conditions, the bots are not evenly 
		spaced in the road. if you print the initial positions you will realize.
		'''

		initial_positions = np.linspace(0,self.road_length,self.num_cars+1, dtype=int)

		for i in range(self.num_cars):
			self.bots[i].position = initial_positions[i]

	def check_superposition(self):

		_, counts = np.unique([bot.position for bot in self.bots], return_counts=True)
		# print('counts',counts)
		all_different = np.all(counts == 1)
		# print('alldiffe',all_different)

		if not all_different:
			logging.error('Superposition... Stopping simulation')
			print(self.num_cars, self.step, [bot.position for bot in self.bots])
			raise Exception("Simulation stopped due to superposition")
		
	def random_slowdown(self):
		'''
		Stochaticity of the simulation, with a probability p_slowdown a bot can reduce its velocity in one point
		'''
		for bot in self.bots:
			if np.random.rand() < self.p_slowdown:
				bot.speed = np.maximum(0, bot.speed - 1)
	
	def acceleration(self):
		'''
		Increases in one point the velocity of the bots whenever it is smaller than the maximum speed and the detection sensor is OFF (False).
		'''
		# Apply acceleration rule - 1
		for bot in self.bots:
			gap = bot.distance - 1

			if bot.detection_status == False:
				bot.speed = np.minimum(np.minimum(self.speed_limit, bot.speed + 1), gap)
			else:
				bot.speed = 0
	
	def calculating_separation_distances(self):
		'''
		Computes the distances to the bots in front. This separation distance is an attribute of the class Bots, so 
		every bot has the info of the distance to the bot in front of it. 
		'''
		if len(self.bots) == 1:
			self.detection_dists[0] = self.road_length
			self.bots[0].distance = self.detection_dists[0]
		
		else:

			for i in range(0, len(self.bots)):
				try:
					self.detection_dists[i] = self.bots[i+1].position - self.bots[i].position
				except:
					self.detection_dists[i] = self.bots[0].position - self.bots[i].position
				
				if self.detection_dists[i] < 0:
					try:
						self.detection_dists[i] = self.bots[i+1].position + (self.road_length - self.bots[i].position)
					except:
						self.detection_dists[i] = self.bots[0].position + (self.road_length - self.bots[i].position)
				
				if self.detection_dists[i] == 0:
					raise Exception("Simulation stopped due to superposition")
				
				self.bots[i].distance = self.detection_dists[i]

	def braking(self):
		'''Makes zero the bots speeds if the distance to the bot in front is smaller than a threshold (defined as the 
		detection distance).
		It also sets the detector sensor in ON (True) or OFF (False).
		'''

		# Apply braking due interaction rule - 2
		for bot in self.bots:
			if bot.distance <= self.detection_dist:
				bot.speed = 0
				bot.detection_status = True
			else:
				bot.detection_status = False

	def update_positions(self):
		'''
		Once all the simulation process has taken place it updates the new positions by summing the speed to the current positions.
		Also appends the new position to the positions sequence.
		'''
		
		for bot in self.bots:
			bot.position = (bot.position + bot.speed) % self.road_length

			# Update the sequence of positions
			bot.positions_seq.append(bot.position)

			# print(bot.position)

			if bot.speed == 0:
				bot.total_steps_stopped += 1
		
			# 	# print(bot.seq_steps_stopped)
			# elif bot.speed >= 1:
			# 	np.append(self.steps_stopped, bot.seq_steps_stopped)
			# 	bot.seq_steps_stopped = 0
		
			# print(self.steps_stopped)

	def show_positions(self):
		positions = []
		for bot in self.bots:
			positions.append(bot.position)
		
		print(positions)

	def show_speeds(self):
		speeds = []
		for bot in self.bots:
			speeds.append(bot.speed)
		
		print(speeds)
