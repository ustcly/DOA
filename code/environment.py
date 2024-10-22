from __future__ import print_function

import copy
import time
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref

try:
	import queue
except ImportError:
	import Queue as queue

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')
try:
	import numpy as np
except ImportError:
	raise RuntimeError(
		'cannot import numpy, make sure numpy package is installed')
import carla
from carla import ColorConverter as cc
# from agents.navigation.roaming_agent import RoamingAgent
# from agents.navigation.basic_agent import BasicAgent
# from carla_tools import HUD
from carla_record.manual_control_steeringwheel import World, HUD
from carla_record.experiments_parameters import choose_parameter

import argparse
from collections import deque
import pandas as pd


def draw_waypoints(world, route):
	x0 = route[0, 0]
	y0 = route[0, 1]
	for k in range(1, route.shape[0]):
		r = route[k, :]
		x1 = r[0]
		y1 = r[1]
		dx = x1 - x0
		dy = y1 - y0
		if dx * dx + dy * dy > 1:  # original 2.5 after sqrt
			begin = carla.Location(x=x0, y=y0, z=0.01)
			end = carla.Location(x=x1, y=y1, z=0.01)
			# angle = np.radians(r[2])
			# end = begin + carla.Location(x=6 * np.cos(angle), y=6 * np.sin(angle))
			world.debug.draw_arrow(begin, end, thickness=0.05, arrow_size=0.08, life_time=2.0,
			                       color=carla.Color(78, 8, 77))
			# will cause carla crash
			x0 = x1
			y0 = y1


class environment():
	def __init__(self, BSrange, args, brakeSize=4, steerSize=9, traj_num=0, collectFlag=False, model='dqn',
	             vehicleNum=1,testFlag = False):
		step_T_bound = BSrange[0]  # Boundary of brake values
		step_S_bound = BSrange[1]  # Boundary of the steering angle values#-----------0.8
		self.args = args
		self.testFlag = testFlag
		log_level = logging.INFO

		logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

		logging.info('listening to server 127.0.0.1:2000')

		# self.vehicleNum = vehicleNum

		print('CollectFlag:', collectFlag, ', Carla_Sync:', self.args.sync_mode)
		# if not collectFlag:
		# 	start_location = carla.Location(x = self.route[0,0], y = self.route[0,1], z = 0.1)
		# 	start_rotation = carla.Rotation(pitch = 0, yaw = self.route[0,2], roll = 0)
		# else:
		# 	start_location = carla.Location()
		# 	start_rotation = carla.Rotation()

		# self.start_point = carla.Transform(location = start_location, rotation = start_rotation)  # type : Transform (location, rotation)

		self.client = carla.Client('127.0.0.1', 2000)
		self.client.set_timeout(4.0)
		# self.display = pygame.display.set_mode((600, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)
		self.display = pygame.display.set_mode((600, 480))  # flags=0, color depth=8
		self.hud = HUD(600, 480, 0)
		# ==============================================================================
		# -- scene parameters ----------------------------------------------------------
		# ==============================================================================
		self.param = choose_parameter(traj_num)  # -- 99  for test
		self.LR = self.param['LR']# drift to right as 1, left is -1
		# ==============================================================================
		self.clock = pygame.time.Clock()
		self.fixed_seconds = 0.01
		self.world = World(self.client, self.client.load_world(self.param['map']), self.hud, self.param['ego'],
		                   param_ID=traj_num,
		                   fixed_seconds=self.fixed_seconds, sync_mode=self.args.sync_mode,testFlag = testFlag)

		print('World created with map: ' + self.param['map'])

		self.mp = self.world.mp
		self.refreshRoute(traj_num, vehicleNum)  # a series of caral.transform
		self.minDis = 0
		self.collectFlag = collectFlag
		# self.traj_drawn_list = []

		self.destinationFlag = False
		self.away = [False, False]
		self.collisionFlag = False
		self.endFlag = False
		self.waypoints_ahead = []
		# self.waypoints_neighbor = []
		self.steer_history = deque(maxlen=20)
		self.brake_history = deque(maxlen=20)
		self.velocity_local = []
		self.model = model


		# if model == 'dqn':
		# 	self.step_T_pool = [step_T_bound[0]]
		# 	self.step_S_pool = [step_S_bound[0]]
		# 	t_step_rate = (step_T_bound[1] - step_T_bound[0]) / brakeSize
		# 	s_step_rate = (step_S_bound[1] - step_S_bound[0]) / steerSize
		# 	for i in range(brakeSize):
		# 		self.step_T_pool.append(self.step_T_pool[-1] + t_step_rate)
		# 	for i in range(steerSize):
		# 		self.step_S_pool.append(self.step_S_pool[-1] + s_step_rate)
		# 	# print(self.step_T_pool)
		# 	# print(self.step_S_pool)
		# 	self.tStateNum = len(self.step_T_pool)
		# 	self.sStateNum = len(self.step_S_pool)

		self.e_heading = 0.0
		self.e_d_heading = 0.0
		self.e_dis = None
		self.e_d_dis = 0.0
		self.e_slip = 0.0
		self.e_d_slip = 0.0
		self.e_vx = 0.0
		self.e_d_vx = 0.0
		self.e_vy = 0.0
		self.e_d_vy = 0.0

		self.tg = 0.0
		self.clock_history = 0.0  # pop the current location into self.waypoints_history every 0.2s

		self.k_heading = 0.1

		self.waypoints_ahead_local = []
		self.waypoints_history = deque(maxlen=5)
		self.waypoints_history_local = []

		self.last_steer = 0.0
		self.last_brake = 0.0

		self.tire_friction_array = np.arange(3, 4.2, 0.2)  # [3,4], 11D
		self.mass_array = np.arange(1400.0, 1900, 50)  # array([1700, 1750, 1800, 1850, 1900])

		self.ori_physics_control = self.world.player.get_physics_control()

	# self.world.world.set_weather(carla.WeatherParameters.ClearNoon)
	# self.world.world.tick()

	def refreshRoute(self, traj_num, vehicleNum):
		if vehicleNum == 1:
			_file = glob.glob('ref_trajectory/*' + str(traj_num) + '*.csv')
			traj = pd.read_csv(_file[0])
			print('Reading referance trajtory: ', _file[0])
		else:
			traj = pd.read_csv('ref_trajectory/traj_different_vehicles/' + str(vehicleNum) + '.csv')
			print('Reading referance trajtory: ref_trajectory/traj_different_vehicles/' + str(vehicleNum) + '.csv')
		self.route = traj.values
		self.route_len = self.route.shape[0]
		# ==========Append============
		# self.mp = self.world.world.get_map()
		location = carla.Location(x=self.route[-1, 0], y=self.route[-1, 1], z=self.route[-1, 11])
		map_wp = self.mp.get_waypoint(location)

		for i in range(20):
			wi = map_wp.next(0.4+0.4*i)[0]
			wx = wi.transform.location.x
			wy = wi.transform.location.y
			wz = wi.transform.location.z
			w_heading = wi.transform.rotation.yaw
			# world_x,world_y,world_heading,world_vx,world_vy,slip_angle,yaw_rate,steer,throttle,hand_brake,brake,world_z,world_vz,pitch
			_ref = np.array([wx,wy,w_heading,0,0,0,0,0,0,False,0,wz,0,wi.transform.rotation.pitch],ndmin = 2)
			self.route = np.concatenate((self.route,_ref), axis =0)
			# print(i,'after',self.route.shape)

		self.route_x = self.route[:, 0]
		self.route_y = self.route[:, 1]
		for row in self.route:
			if row[2] < 0:
				row[2] = row[2] + 360
			elif row[2] > 360:
				row[2] = row[2] - 360

	def step(self, actionfrom=1, steer=0.0, brake=0.0, manual_control=False):
		# apply the computed control commands, update endFlag and return state/reward
		if not manual_control:
			# ly: get action
			control = self.getAction(steer=steer, brake=brake)

			if self.model == 'sac':
				control.steer = 0.4 * control.steer + 0.6 * self.last_steer
				control.brake = 0.1 * control.brake + 0.9 * self.last_brake
			# if self.model == 'ddpg':
			# 	control.steer = 0.8 * control.steer + 0.2 * self.last_steer
			# 	control.brake = 0.9 * control.brake + 0.1 * self.last_brake

			self.last_steer = control.steer
			self.last_brake = control.brake

			# ly: apply control
			# print('Apply_control: steer=%.3f, brake=%.3f'%(control.steer,control.brake))
			control.steer = self.LR * control.steer
			# print(control.steer,control.brake)
			self.world.player.apply_control(control)
			control.steer = self.LR * control.steer
			if self.args.sync_mode:
				# self.clock.tick()
				self.world.world.tick()
			else:
				time.sleep(self.fixed_seconds)
			self.steer_history.append(control.steer)
			self.brake_history.append(control.brake)

		# elif not self.collectFlag:
		#     control = self.world.player.get_control()
		#     self.steer_history.append(control.steer)
		#     self.brake_history.append(control.brake)
		#     #time.sleep(0.02)

		# ly: get state

		newState = self.getState(control)

		if not self.collectFlag:
			self.collisionFlag = self.collisionDetect()
			if self.args.load:
				reward = self.getReward(newState, self.steer_history, self.brake_history)
			else:
				reward = self.getReward(newState, self.steer_history, self.brake_history)
			# ly: get reward

			# print("len_state:",len(newState),"reward:%.2f"%reward)
			n_s = self.state_tr(newState)
			# print("state:",n_s[10],n_s[14])

			return n_s, reward, self.collisionFlag, self.destinationFlag, self.away, control

		# else:
		# 	control = self.world.player.get_control()
		# 	return newState, control

	def reset(self, traj_num=0, testFlag=False, friction=4, mass=1410.0, testvehicle=False,vehicle ='audi.a2'):

		# random change the tire friction and vehicle mass:
		# if not testFlag:
		# 	index_friction = np.random.randint(0, self.tire_friction_array.shape[0])
		# 	index_mass = np.random.randint(0, self.mass_array.shape[0])
		# 	# friction = self.param['friction']
		# 	friction = self.tire_friction_array[index_friction]
		# 	mass = self.mass_array[index_mass]
		self.testFlag=testFlag
		if testvehicle:
			self.world.restart(testvehicle = True, friction = friction, vehicle = vehicle)
		else:
			self.world.restart(testvehicle = False, friction = friction, mass = mass)

		# if self.args.sync_mode:
		# 	self.world.world.tick()
		# 	self.world.world.tick()
		# else:
		# 	time.sleep(0.05)


		# detect:
		# if differentFriction or testFlag:
		# 	physics = self.world.player.get_physics_control()
		# 	print('firction: {}, mass: {}'.format(physics.wheels[0].tire_friction, physics.mass))
		# print('center of mass: ', physics.center_of_mass.x, physics.center_of_mass.y, physics.center_of_mass.z)

		# self.world.player.apply_control(carla.VehicleControl())

		self.world.collision_sensor.history = []
		self.away = [False, False]
		self.endFlag = False
		self.destinationFlag = False
		self.collisionFlag = False
		self.steer_history.clear()
		self.brake_history.clear()
		# self.waypoints_neighbor = []
		self.waypoints_ahead = []

		self.waypoints_ahead_local = []  # carla.location 10pts
		self.waypoints_history.clear()  # carla.location  5pts
		self.waypoints_history_local = []

		self.last_steer = 0.0
		self.last_brake = 0.0

		# self.drived_distance = 0

		# print('\nRESET!\n')

		return 0

	def getState(self, control):
		# location = self.world.player.get_location()
		# angular_velocity = self.world.player.get_angular_velocity()
		transform = self.world.player.get_transform()
		ego_yaw = transform.rotation.yaw  # degree
		ego_pitch = transform.rotation.pitch
		location = transform.location
		state = []
		if ego_yaw < 0:
			ego_yaw = ego_yaw + 360
		elif ego_yaw > 360:
			ego_yaw = ego_yaw - 360
		# ego_yaw = ego_yaw / 180.0 * 3.141592653


		self.getNearby(location)  # will update self.minDis. distance squared


		# self.getLocalHistoryWay(location, ego_yaw)
		self.getLocalFutureWay(location, ego_yaw)

		# self.velocity_local = self.velocity_world2local(ego_yaw)
		self.velocity_local = self.velocity_world2(ego_yaw)  # velocity_local is velocity_world
		# ego_yaw = ego_yaw / 3.141592653 * 180
		# if ego_yaw > 180:
		# 	ego_yaw = -(360 - ego_yaw)

		if not self.collectFlag:
			if self.args.sync_mode:
				dt = self.fixed_seconds
			else:
				dt = time.time() - self.tg
			if self.e_dis is not None:
				self.e_d_dis = (self.minDis - self.e_dis) / dt
			else:
				self.e_d_dis = 0
			self.e_dis = self.minDis
			vabs = abs(self.velocity_local[0]) + abs(self.velocity_local[1])
			# print('v:',self.velocity_local[0],self.velocity_local[1])

			if self.e_dis > 6 or vabs < 10:  # distance
				# away to be finished
				# print('edis,v:',self.e_dis,vabs)
				self.away[0] = True
			elif self.traj_index > self.route_len and self.e_dis > 3:
				# print('index,route_len:',self.traj_index,self.route_len)
				self.away[0] = True
				self.away[1] = True
			if self.e_dis > 3.1:
				# away to be punished
				self.away[1] = True

			# error of heading:
			# 1. calculate the abs
			# way_yaw = self.waypoints_ahead[0, 2]

			# 2. update the way_yaw based on vector guidance field:
			# vgf_left = self.vgf_direction(location)
			# # 3. if the vehicle is on the left of the nearst waypoint, according to the heading of the waypoint
			# if vgf_left:
			# 	way_yaw = np.arctan(self.k_heading * self.e_dis) / 3.141592653 * 180 + way_yaw
			# else:
			# 	way_yaw = -np.arctan(self.k_heading * self.e_dis) / 3.141592653 * 180 + way_yaw
			# if way_yaw > 180:
			# 	way_yaw = -(360 - way_yaw)
			# if way_yaw < -180:
			# 	way_yaw = way_yaw + 360
			#
			# if ego_yaw * way_yaw > 0:
			# 	e_heading = abs(ego_yaw - way_yaw)
			# else:
			# 	e_heading = abs(ego_yaw) + abs(way_yaw)
			# 	if e_heading > 180:
			# 		e_heading = 360 - e_heading
			e_heading = ego_yaw - self.waypoints_ahead[0, 2]
			if np.sign(e_heading) * np.sign(self.route[self.traj_index2, 2] - ego_yaw) >= 0:
				e_heading = 0.0
			if e_heading < 2: e_heading=0
			self.e_d_heading = (e_heading - self.e_heading) / dt
			self.e_heading = e_heading

			slip = self.velocity_local[2]
			e_slip = self.velocity_local[2] - self.waypoints_ahead[0, 5]
			if np.sign(e_slip) * np.sign(self.route[self.traj_index2, 5] - self.velocity_local[2]) >= 0:
				e_slip = 0.0
			self.e_d_slip = (e_slip - self.e_slip) / dt
			self.e_slip = e_slip

			e_vx = self.velocity_local[0] - self.waypoints_ahead[0, 3]
			if np.sign(e_vx) * np.sign(self.route[self.traj_index2, 3] - self.velocity_local[0]) >= 0:
				e_vx = 0.0
			self.e_d_vx = (e_vx - self.e_vx) / dt
			self.e_vx = e_vx

			e_vy = self.velocity_local[1] - self.waypoints_ahead[0, 4]
			if np.sign(e_vy) * np.sign(self.route[self.traj_index2, 4] - self.velocity_local[1]) >= 0:
				e_vy = 0.0
			self.e_d_vy = (e_vy - self.e_vy) / dt
			self.e_vy = e_vy

			# control = self.world.player.get_control()

			steer = control.steer
			brake = control.brake

			self.waypoints_history.append(
				np.array([location.x, location.y, ego_yaw, steer, brake, self.velocity_local[2]]))

			vx = self.velocity_local[0]
			vy = self.velocity_local[1]
			vz = self.velocity_local[3]
			e_d_slip = self.e_d_slip
			if vx * vx + vy * vy < 4:  # if the speed is too small we ignore the error of slip angle
				e_slip = 0
				e_d_slip = 0
			ex = location.x - self.waypoints_ahead[0, 0]
			ey = location.y - self.waypoints_ahead[0, 1]
			ez = location.z - self.waypoints_ahead[0, 11]
			# state = [steer, brake, self.e_dis, self.e_d_dis, self.e_heading, self.e_d_heading, e_slip, e_d_slip,
			# self.e_vx, self.e_d_vx, self.e_vy, self.e_d_vy]
			state = [steer, brake, location.x, location.y, location.z, vx, vy, vz, ego_yaw,ego_pitch, slip,
			         ex, ey, ez, self.e_heading, self.e_d_dis]
			[state.append(k[0]) for k in self.waypoints_ahead_local]  # wx
			[state.append(k[1]) for k in self.waypoints_ahead_local]  # wy
			[state.append(k[2]) for k in self.waypoints_ahead_local]  # wz
			[state.append(k[3]) for k in self.waypoints_ahead_local]  # w_heading
			# print(len(state))
			self.tg = time.time()

			return state

	def getReward(self, state, steer_history, brake_history):
		e_dis = self.minDis  # current distance
		e_d_dis = state[15]  # distance change rate, self.e_d_dis = (self.minDis - self.e_dis) / dt
		# e_slip = state[6]
		e_heading = state[14]
		# std_steer0 = np.array(steer_history)
		# std_steer = std_steer0.std()

		# std_brake = np.array(brake_history)
		# std_brake = std_brake.std()

		# r_dis = np.exp(-0.5 * e_dis)
		r_dis = 2 * np.exp(-2 * e_dis) - 1  # e_dis range: 0 ~ self.away
		r_d_dis = 1 / (1 + np.exp(0.05 * e_d_dis))  # e_d_dis range: -speed ~ speed
		r_heading = 1 - 2 * np.tanh(abs(e_heading / 40))  # e_heading range: -360 ~ 360

		# if abs(e_slip) < 90:
		#     r_slip = np.exp(-0.1 * abs(e_slip))
		# elif (e_slip) >= 90:
		#     r_slip = -np.exp(-0.1 * (180 - e_slip))
		# else:
		#     r_slip = -np.exp(-0.1 * (e_slip + 180))
		#
		# r_std_steer = np.exp(-2 * std_steer)
		# r_std_brake = np.exp(-2 * std_brake)

		# tiny_steer = 0.02
		# steer = std_steer0[np.abs(std_steer0) > tiny_steer]
		# count = 0
		# for i in range(len(steer) - 1):
		# 	if steer[i] * steer[i + 1] < 0:
		# 		count = count + 1
		# t_c = 1 / (10*np.exp(abs(count-1)))  # ly: steer reverse times
		schedule = min(self.traj_index / self.route_len,1)  # range: 0~1
		r_sche = min(2 * (schedule - 0.5), 0)
		# print(self.route_length.shape[0])

		reward = (0.6 * r_dis + 0.4 * r_heading + 0.0 * r_sche + 0.0 * r_d_dis)

		# if schedule > 0.8:
		# 	reward = reward + 90
		# elif schedule > 0.6:
		# 	reward = reward + 70
		# elif schedule > 0.4:
		# 	reward = reward + 30
		# elif self.traj_index <= 4:
		# 	reward = 0
		# if self.traj_index < 2:
		# 	reward = 0
		if self.destinationFlag:
			reward = reward + 20
		# if self.away[1]:
		# 	reward = reward - 8
		if self.collisionFlag or self.away[0]:
			reward = reward - 20

		# print('reward:%.2f, schedule:%.2f, r_dis:%.2f, r_d_dis:%.2f, r_heading:%.2f'%(15 * reward,schedule,r_dis,r_d_dis,r_heading))
		return int(750*self.fixed_seconds) * reward #15 for fixed_seconds=0.02

	# def getReward2(self, state, steer_history, brake_history):
	# 	# self.waypoints_history.append(np.array([location.x, location.y, ego_yaw, steer, brake, self.velocity_local[2]]))
	# 	player_state = self.waypoints_history[-1]
	# 	desti_p = self.route[-1, 0:3]
	# 	end_wp = self.mp.get_waypoint(carla.Location(x=desti_p[0],y=desti_p[1]))
	# 	end_wp = end_wp.transform.location
	# 	obstacle_spawn_p = carla.Transform()
	# 	spawn_point = self.param['spawn']
	# 	obstacle_spawn_p.rotation.yaw = spawn_point.rotation.yaw
	# 	obstacle_spawn_p.location.x = spawn_point.location.x + self.param['obstacle_ahead'] * np.cos(
	# 		np.pi * self.param['spawn'].rotation.yaw / 180.0)
	# 	obstacle_spawn_p.location.y = spawn_point.location.y + self.param['obstacle_ahead'] * np.sin(
	# 		np.pi * self.param['spawn'].rotation.yaw / 180.0)
	#
	# 	dis_y_obst = abs(player_state[1] - obstacle_spawn_p.location.y)
	# 	dis_x_obst = abs(player_state[0] - obstacle_spawn_p.location.x)
	# 	dis_y_desti = abs(player_state[1] - end_wp.y)
	# 	dis_x_desti = abs(player_state[0] - end_wp.x)
	# 	dis_y_range = abs(end_wp.y - spawn_point.location.y)
	# 	dis_x_range = abs(end_wp.x - spawn_point.location.x)
	#
	# 	r_obs_x = 2 * np.exp(-0.8 * dis_x_desti) - 1
	# 	r_desti = (dis_y_range - dis_y_desti) / dis_y_range
	# 	reward = 1 * r_obs_x + 0 * r_desti
	#
	# 	# 1/(gamma ** total_steps)=1/(0.97**83)=12.5, so the finish reward should be larger than 12.5 * daily_reward
	# 	if self.destinationFlag:
	# 		reward = reward + 20
	# 	if self.away[0]:
	# 		reward = reward - 15
	# 	if self.collisionFlag:
	# 		reward = reward - 20
	#
	# 	reward = 12 * reward if r_desti > 0.35 else 5 * reward
	#
	# 	# print('reward:%.2f, r_obs_x:%.2f, r_desti:%.2f'%(reward,r_obs_x,r_desti))
	# 	return reward



	def drawPoints(self):
		# will cause carla crash
		draw_waypoints(self.world.world, self.route)

	def render(self):
		# show ROS client window by pygame
		# self.hud.tick(self.clock, self.e_dis, self.e_heading, self.velocity_local[2])
		self.clock.tick()
		self.world.tick(self.clock)
		self.world.render(self.display)
		pygame.display.flip()

	# def velocity_world2local(self, yaw):
	# 	velocity_world = self.world.player.get_velocity()
	# 	vx = velocity_world.x
	# 	vy = velocity_world.y
	# 	yaw = np.radians(yaw)
	#
	# 	local_x = float(vx * np.cos(yaw) - vy * np.sin(yaw))
	# 	local_y = float(vy * np.cos(yaw) + vx * np.sin(yaw))
	# 	if local_x != 0:
	# 		slip_angle = np.arctan(local_y / local_x) / 3.1415926 * 180
	# 	else:
	# 		slip_angle = 0
	#
	# 	velocity_local = [local_x, local_y, slip_angle]
	# 	return velocity_local

	def velocity_world2(self, yaw):
		velocity_world = self.world.player.get_velocity()
		vx = velocity_world.x
		vy = velocity_world.y
		vz = velocity_world.z
		# yaw = np.radians(yaw)
		slip_angle = 0.0
		if abs(vx) + abs(vy) > 0.2:
			v_yaw = 180 * np.arctan2(vy, vx) / np.pi
			if v_yaw<0:
				v_yaw = 360 + v_yaw
			slip_angle =  v_yaw - yaw
			# print('v_yaw,yaw:',v_yaw,yaw)
		else:
			slip_angle = 0

		velocity_w = [vx, vy, slip_angle,vz]
		return velocity_w

	# def velocity_local2world(self, velocity_local, yaw):
	# 	vx = velocity_local[0]
	# 	vy = velocity_local[1]
	#
	# 	world_x = vx * np.cos(yaw) - vy * np.sin(yaw)
	# 	world_y = vy * np.cos(yaw) + vx * np.sin(yaw)
	#
	# 	return carla.Vector3D(world_x, world_y, 0)

	def collisionDetect(self):
		if self.world.collision_sensor.history:
			return True
		else:
			return False

	def getAction(self, actionID=4, steer=0, brake=0):
		control = carla.VehicleControl(
			brake=brake,
			steer=steer,
			throttle=0.0)
		return control

	def getNearby(self, egoLocation):

		self.waypoints_ahead = []
		# self.waypoints_neighbor = []
		# egoLocation = self.world.player.get_location()
		dx_array = self.route_x - egoLocation.x
		dy_array = self.route_y - egoLocation.y
		dis_array = dx_array * dx_array + dy_array * dy_array  # ly: omit the np.sqrt()
		minDis = np.amin(dis_array)
		_ = np.where(dis_array == minDis)
		index = _[0][0]  # index for the min distance to all waypoints.

		# self.drived_distance = self.route_length[index]
		self.waypoints_ahead = self.route[index:, :]

		# if index >= 20:
		# 	index_st = index - 20
		# else:
		# 	index_st = 0
		# self.waypoints_neighbor = self.route[index_st:, :]
		self.traj_index = index
		dz = self.route[index,11] - egoLocation.z
		self.minDis = np.sqrt(dis_array[index] + dz**2)
		if index == 0:
			self.traj_index2 = 1
			# self.minDis = np.sqrt(dis_array[index])
		elif index == self.route.shape[0] - 1:
			self.traj_index2 = index - 1
			# self.minDis = np.sqrt(dis_array[index])
		else:
			if dis_array[index - 1] <= dis_array[index + 1]:
				self.traj_index2 = index - 1
			else:
				self.traj_index2 = index + 1
		if index >= self.route_len:
			self.minDis = self.distance_point_to_line(self.route_x[self.route_len +2], self.route_y[self.route_len +2],
			                                          self.route_x[self.route_len +5], self.route_y[self.route_len +5],
			                                          egoLocation.x, egoLocation.y)
		if self.minDis < 0.15:
			self.minDis = 0

	def distance_point_to_line(self, x1, y1, x2, y2, x0, y0):
		# point:(x0,y0), line:(x1,y1)~(x2,y2)
		return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
	# def routeAppend(self, egoLocation):


	def getLocalFutureWay(self, egoLocation, yaw):
		# transfer the future waypoints (#10) to the local coordinate.
		# x, y, slip (degree)
		future_wp = 8  # get 8 future waypoints
		k = future_wp
		ways = self.waypoints_ahead[0::8, :]  # cpd:filter to 1m between way pts
		# print(self.traj_index-self.route_len,self.minDis,yaw-self.waypoints_ahead[0,2])
		if self.traj_index > self.route_len and self.minDis < 2.0 and abs(yaw-self.waypoints_ahead[0,2]) < 20:
			self.destinationFlag = True
		elif self.testFlag and self.traj_index >= self.route_len:
			self.destinationFlag = True
		elif self.testFlag and self.traj_index > 0.8*self.route_len and abs(yaw-self.param['spawn'].rotation.yaw)<3:
			self.destinationFlag = True
		if ways.shape[0] < future_wp:
			k = ways.shape[0]
			# print('almost success')
			location = carla.Location(x=self.waypoints_ahead[-1, 0], y=self.waypoints_ahead[-1, 1], z=self.waypoints_ahead[-1, 11])
			map_wp = self.mp.get_waypoint(location)
		self.waypoints_ahead_local = []
		# yaw = np.radians(yaw)
		# y_sin = np.sin(yaw)
		# y_cos = np.cos(yaw)

		for w in ways[0:k]:  # add reference waypoint
			wx = w[0]
			wy = w[1]
			wz = w[11]
			w_heading = w[2]
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y
			dz = wz - egoLocation.z
			# nx = dx * y_cos - dy * y_sin
			# ny = dy * y_cos + dx * y_sin
			d_heading = w_heading - yaw
			self.waypoints_ahead_local.append(np.array([wx, wy, wz, w_heading]))
		for i in range(future_wp - k):  # add map waypoint
			wi = map_wp.next(1 + i)[0]
			wx = wi.transform.location.x
			wy = wi.transform.location.y
			wz = wi.transform.location.z
			d_heading = wi.transform.rotation.yaw - yaw
			# w_heading = wi.transform.rotation.yaw
			dx = wx - egoLocation.x
			dy = wy - egoLocation.y
			dz = wz - egoLocation.z
			# nx = dx * y_cos - dy * y_sin
			# ny = dy * y_cos + dx * y_sin
			self.waypoints_ahead_local.append(np.array([wx, wy,wz, w_heading]))

	def state_tr(self,instate):
		# state = [steer, brake, location.x, location.y, location.z, vx, vy, vz, ego_yaw,ego_pitch, slip,
		# 	         ex, ey, ez, self.e_heading, self.e_d_dis] + future:[x*8,y*8,dz*8,heading*8]
		outstate = copy.copy(instate)
		x = instate[2] - self.param['spawn'].location.x
		y = instate[3] - self.param['spawn'].location.y
		r = np.radians(self.param['spawn'].rotation.yaw)
		sinr = np.sin(r)
		cosr = np.cos(r)
		outstate[2] = x * cosr + y * sinr #x
		outstate[3] = (y * cosr - x * sinr) * self.LR
		if abs(self.route[0,11]) > 0.1:
			outstate[4] = instate[4] - self.route[0,11]
		outstate[5] = instate[5] * cosr + instate[6] * sinr #vx
		outstate[6] = (instate[6] * cosr - instate[5] * sinr) * self.LR
		outstate[8] = self.de360to_180(instate[8] - self.param['spawn'].rotation.yaw) * self.LR #ego_yaw
		outstate[10] = (instate[10]) * self.LR #slip
		outstate[12] = (instate[12]) * self.LR
		outstate[14] = (instate[14]) * self.LR #e_heading
		# if self.param['id']=='7': outstate[15] = 0*instate[14]
		for i in range(8):
			wx = instate[16+i] - self.param['spawn'].location.x
			wy = instate[24+i] - self.param['spawn'].location.y
			outstate[16+i] = wx * cosr + wy * sinr #future x
			outstate[24+i] = (wy * cosr - wx * sinr) * self.LR #future y
			if abs(self.route[0,11]) > 0.1:
				outstate[32+i] = instate[32+i] - self.route[0,11] #future z
			outstate[40+i] = self.de360to_180(instate[40+i] - self.param['spawn'].rotation.yaw) * self.LR #future heading
		return outstate

	def de360to_180(self,in_d):
		#in:-360~360(diff of two degrees in [0,360]), out:-180-180
		out_d = in_d
		if in_d >180:
			out_d = in_d - 360
		elif in_d < -180:
			out_d = in_d + 360
		return out_d
# def getLocalHistoryWay(self, egoLocation, yaw):
# 	# x, y, steer, slip (degree)
# 	ways = self.waypoints_history
# 	yaw = np.radians(yaw)
# 	self.waypoints_history_local = []
# 	if len(ways) < 5:
# 		for i in range(5 - len(ways)):
# 			self.waypoints_history_local.append(np.array([0, 0, 0, 0]))
# 	y_sin = np.sin(yaw)
# 	y_cos = np.cos(yaw)
# 	for w in ways:
# 		wx = w[0]
# 		wy = w[1]
# 		w_steer = w[2]
# 		w_slip = w[3]
# 		dx = wx - egoLocation.x
# 		dy = wy - egoLocation.y
#
# 		nx = dx * np.cos(yaw) - dy * np.sin(yaw)
# 		ny = dy * np.cos(yaw) + dx * np.sin(yaw)
# 		self.waypoints_history_local.append(np.array([nx, ny, w_steer, w_slip]))

# def vgf_direction(self, egoLocation):
# 	way_x = self.waypoints_ahead[0, 0]
# 	way_y = self.waypoints_ahead[0, 1]
# 	yaw = np.radians(self.waypoints_ahead[0, 2])
#
# 	dx = egoLocation.x - way_x
# 	dy = egoLocation.y - way_y
#
# 	nx = dx * np.cos(yaw) - dy * np.sin(yaw)
# 	ny = dy * np.cos(yaw) + dx * np.sin(yaw)
#
# 	if ny < 0:
# 		return True
# 	else:
# 		return False
