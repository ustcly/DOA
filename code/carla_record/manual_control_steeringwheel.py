#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function
from .experiments_parameters import choose_parameter

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import copy
import csv

try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time

if sys.version_info >= (3, 0):
	from configparser import ConfigParser
else:
	from ConfigParser import RawConfigParser as ConfigParser

try:
	import pygame
	from pygame.locals import KMOD_CTRL
	from pygame.locals import KMOD_SHIFT
	from pygame.locals import K_0
	from pygame.locals import K_9
	from pygame.locals import K_BACKQUOTE
	from pygame.locals import K_BACKSPACE
	from pygame.locals import K_COMMA
	from pygame.locals import K_DOWN
	from pygame.locals import K_ESCAPE
	from pygame.locals import K_F1
	from pygame.locals import K_LEFT
	from pygame.locals import K_PERIOD
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SLASH
	from pygame.locals import K_SPACE
	from pygame.locals import K_TAB
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_c
	from pygame.locals import K_d
	from pygame.locals import K_h
	from pygame.locals import K_m
	from pygame.locals import K_p
	from pygame.locals import K_q
	from pygame.locals import K_r
	from pygame.locals import K_s
	from pygame.locals import K_w
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
	rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
	name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
	presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
	return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
	name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
	return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
	def __init__(self, client, carla_world, hud, actor_filter, param_ID=1, collectFlg=0, fixed_seconds=0.02, sync_mode=False,testFlag = False):
		# ==============================================================================
		# -- scene parameters ----------------------------------------------------------
		# ==============================================================================
		self.param = choose_parameter(param_ID)  # -- 99  for test
		# ==============================================================================
		self.world = carla_world
		self.mp = carla_world.get_map()
		self.client = client
		# print('client v:',self.client.get_client_version())
		self.hud = hud
		self.testFlag=testFlag
		self.player = None
		self.obstacle = None
		self.obs2 = None
		self.collision_sensor = None
		self.lane_invasion_sensor = None
		self.gnss_sensor = None
		self.camera_manager = None
		self.rec = None
		self._weather_presets = find_weather_presets()
		self._weather_index = 0
		self.attempt = 0
		self.flg = 0  # 0 for on process, 1 for success, 2 for fail
		self._actor_filter = actor_filter
		self.steer_k_copy = 0
		self.world.on_tick(hud.on_world_tick)
		self.collectFlg = collectFlg
		self.bp = self.world.get_blueprint_library()
		self.fixed_seconds = fixed_seconds
		self.sync_mode = sync_mode

		self.world.set_weather(self.param['weather'])
		if collectFlg or not sync_mode:
			settings = self.world.get_settings()
			settings.synchronous_mode = False
			settings.fixed_delta_seconds = self.fixed_seconds  # 0.01--1ms
			self.world.apply_settings(settings)
		else:
			settings = self.world.get_settings()
			settings.synchronous_mode = True
			settings.fixed_delta_seconds = self.fixed_seconds
			self.world.apply_settings(settings)
			self.world.tick()
		self.restart()

	def restart(self, testvehicle = False, friction = 4.0, mass = 1410, vehicle ='audi.a2'):
		# Keep same camera config if the camera manager exists.
		cam_index = self.camera_manager.index if self.camera_manager is not None else 0
		cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

		if testvehicle:
			self._actor_filter = 'vehicle.' + vehicle
		blueprint = self.bp.find(self._actor_filter)
		blueprint.set_attribute('role_name', 'hero')

		if blueprint.has_attribute('color'):
			# color = random.choice(blueprint.get_attribute('color').recommended_values)
			blueprint.set_attribute('color', '255,0,0')
		# Spawn the player.
		# spawn_points = self.world.get_map().get_spawn_points()
		# spawn_point = random.choice(spawn_points)
		spawn_point = self.param['spawn']  # carla.Transform()
		# if not self.collectFlg:
		#     spawn_point.location.x = spawn_point.location.x - 0.2 * np.cos(np.pi * spawn_point.rotation.yaw/180.0)
		#     spawn_point.location.y = spawn_point.location.y - 0.2 * np.sin(np.pi * spawn_point.rotation.yaw/180.0)
		#            spawn_point.location.x = 195.67#195.6 # Town1:-60, Town5: 195.67
		#            spawn_point.location.y = -100.8#-100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		#            spawn_point.location.z = 1.0
		#            spawn_point.rotation.yaw = 89.0#Town1:0, Town5:91
		obstacle_spawn_p = carla.Transform()
		# oh = self.param['obstacle_ahead'] + 6
		obstacle_spawn_p.rotation.yaw = spawn_point.rotation.yaw
		obstacle_spawn_p.location.z = spawn_point.location.z + 0.5
		obstacle_spawn_p.location.x = spawn_point.location.x + self.param['obstacle_ahead'] * np.cos(
			np.pi * self.param['spawn'].rotation.yaw / 180.0)
		obstacle_spawn_p.location.y = spawn_point.location.y + self.param['obstacle_ahead'] * np.sin(
			np.pi * self.param['spawn'].rotation.yaw / 180.0)

		obs2_spawn_p = carla.Transform()
		obs2_spawn_p.rotation.yaw = spawn_point.rotation.yaw
		obs2_spawn_p.location.z = spawn_point.location.z
		obs2_spawn_p.location.x = spawn_point.location.x + (self.param['ob2'] + self.param['obstacle_ahead'])* np.cos(
        np.pi * spawn_point.rotation.yaw / 180.0)
		obs2_spawn_p.location.y = spawn_point.location.y + (self.param['ob2'] + self.param['obstacle_ahead']) * np.sin(
        np.pi * spawn_point.rotation.yaw / 180.0)
		if self.param['id']=='5':
			_wp_o2 = self.mp.get_waypoint(obs2_spawn_p.location).get_right_lane()
			wp_o2 = _wp_o2.get_right_lane().transform
		elif self.param['id']=='6':
			_wp_o2 = self.mp.get_waypoint(obs2_spawn_p.location).get_right_lane()
			wp_o2 = _wp_o2.get_right_lane().transform
		elif self.param['id']=='7':
			_wp_o2 = self.mp.get_waypoint(obs2_spawn_p.location).get_left_lane()
			wp_o2 = _wp_o2.get_left_lane().transform
		elif self.param['id']=='8':
			_wp_o2 = self.mp.get_waypoint(obs2_spawn_p.location)
			wp_o2 = _wp_o2.get_left_lane().transform
		elif self.param['id']=='11':
			_wp_o2 = self.mp.get_waypoint(obs2_spawn_p.location)
			wp_o2 = _wp_o2.get_right_lane().transform

		wp_o2.location.x = wp_o2.location.x + (wp_o2.location.x-_wp_o2.transform.location.x)*0.15
		wp_o2.location.y = wp_o2.location.y + (wp_o2.location.y-_wp_o2.transform.location.y)*0.15
		obs_bp = self.bp.find('vehicle.tesla.cybertruck')

		wp_p = self.mp.get_waypoint(spawn_point.location).transform
		wp_o1 = self.mp.get_waypoint(obstacle_spawn_p.location).transform
		if self.param['id'] == '6':
			wp_p.rotation.pitch = 3.38
			wp_p.location.z += 0.02
			# print('ssss')
		elif vehicle in {'jeep.wrangler_rubicon','nissan.patrol_2021','mercedes.sprinter'}:
			wp_p.location.z += 0.02
		elif vehicle in {'lincoln.mkz_2020'}:
			wp_p.location.z += 0.01
		else:
			wp_p.location.z += 0.002 #0.002
		wp_o1.location.z += 0.5
		wp_o2.location.z += 0.5

		if self.player is not None:
			self.destroy()
			self.attempt = self.attempt + 1
			self.player = self.world.try_spawn_actor(blueprint, wp_p)
			self.obstacle = self.world.try_spawn_actor(obs_bp, wp_o1)
			if self.testFlag:
				self.obs2 = self.world.try_spawn_actor(obs_bp, wp_o2)
		while self.player is None:
			self.player = self.world.try_spawn_actor(blueprint, wp_p)
			wp_p.location.z += 0.002
			print('player_spawn_z:',wp_p.location.z)
		while self.obstacle is None:
			self.obstacle = self.world.try_spawn_actor(obs_bp, wp_o1)
			wp_o1.location.z += 0.1
			print('obs1_spawn_z:',wp_p.location.z)
		if self.testFlag:
			while self.obs2 is None:
				self.obs2 = self.world.try_spawn_actor(obs_bp, wp_o2)
				wp_o2.location.z += 0.1
				print('ob2_spawn_z:',wp_p.location.z)
		if not self.collectFlg and self.sync_mode:
			self.world.tick()

		self.flg = 0
		if self.collectFlg:
			self.rec = Record(self.attempt, self.param)

		physics = self.player.get_physics_control()
		wheels = physics.wheels
		for i in range(4):
			tire = wheels[i]
			tire.tire_friction = friction
			wheels[i] = tire
		physics.wheels = wheels
		if not testvehicle:
			physics.mass = mass
		self.player.apply_physics_control(physics)
		# self.fri = friction
		# if not self.collectFlg and self.sync_mode:
		#     self.world.tick()

		# Set up the sensors.
		self.collision_sensor = CollisionSensor(self, self.player, self.hud)
		self.lane_invasion_sensor = LaneInvasionSensor(self, self.player, self.hud)
		self.gnss_sensor = GnssSensor(self, self.player)
		self.camera_manager = CameraManager(self, self.player, self.hud)
		self.camera_manager.transform_index = cam_pos_index
		self.camera_manager.set_sensor(cam_index, notify=False)
		actor_type = get_actor_display_name(self.player)
		self.hud.notification(actor_type)
		# if not self.collectFlg and self.sync_mode:
		#     self.world.tick()

		# init_control = self.player.get_control()
		# init_control.gear = 1
		# init_control.steer = 0.0
		# init_control.throttle = 0.3
		# self.player.apply_control(init_control)
		init_velo = carla.Vector3D()
		if abs(wp_p.rotation.pitch) < 1:
			vp = 0
		else:
			vp = wp_p.rotation.pitch
		if vp > 180:
			vp = vp - 360
		v_kmh = self.param['speed'] / 3.6
		init_velo.x = v_kmh * np.cos(np.pi * spawn_point.rotation.yaw / 180.0)*np.cos(np.pi * vp / 180.0)
		init_velo.y = v_kmh * np.sin(np.pi * spawn_point.rotation.yaw / 180.0)*np.cos(np.pi * vp / 180.0)
		init_velo.z = np.sign(vp) * np.sqrt(v_kmh **2 -init_velo.x **2-init_velo.y**2)*0.99
		# print(vp,np.sign(vp),init_velo.z)
		_i = 0
		if self.collectFlg:
			time.sleep(0.5)
		while abs(self.player.get_velocity().y)+abs(self.player.get_velocity().x) < 3:
			if _i >= 1:
				print('target_velocity tried:',_i)
			self.player.set_target_velocity(init_velo)
			if not self.collectFlg and self.sync_mode:
				self.world.tick()
			else:
				time.sleep(0.008)
			_i = _i+1


	def next_weather(self, reverse=False):
		self._weather_index += -1 if reverse else 1
		self._weather_index %= len(self._weather_presets)
		preset = self._weather_presets[self._weather_index]
		self.hud.notification('Weather: %s' % preset[1])
		self.player.get_world().set_weather(preset[0])

	def tick(self, clock):
		self.hud.tick(self, clock)

	def render(self, display):
		self.camera_manager.render(display)
		self.hud.render(display)

	def destroy(self):
		sensors = [
			self.camera_manager.sensor,
			self.collision_sensor.sensor,
			self.lane_invasion_sensor.sensor,
			self.gnss_sensor.sensor]
		for sensor in sensors:
			if sensor is not None:
				sensor.stop()
				sensor.destroy()
		self.camera_manager.sensor = None
		self.camera_manager.index = None
		if self.player or self.obstacle is not None:
			self.obstacle.destroy()
			if self.testFlag:
				self.obs2.destroy()
			self.player.destroy()
			# batch = []
			# batch.append(self.player.destroy())
			# batch.append(self.obstacle.destroy())
			# results = self.client.apply_batch_sync(batch, True)
		if self.rec is not None:
			self.rec.close()
		if not self.collectFlg and self.sync_mode:
			self.world.tick()
		# actors = self.world.get_actors()
		# print("Destroyed. now sensors:",actors.filter('sensor*'))
		# print("Destroyed. now vehicles:",actors.filter('vehicle*'))


# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
	def __init__(self, world, start_in_autopilot):
		self._autopilot_enabled = start_in_autopilot
		if isinstance(world.player, carla.Vehicle):
			self._control = carla.VehicleControl()
			world.player.set_autopilot(self._autopilot_enabled)
		elif isinstance(world.player, carla.Walker):
			self._control = carla.WalkerControl()
			self._autopilot_enabled = False
			self._rotation = world.player.get_transform().rotation
		else:
			raise NotImplementedError("Actor type not supported")
		self._steer_cache = 0.0
		world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

		# initialize steering wheel
		pygame.joystick.init()

		joystick_count = pygame.joystick.get_count()
		if joystick_count > 1:
			raise ValueError("Please Connect Just One Joystick")

		self._joystick = pygame.joystick.Joystick(0)
		self._joystick.init()
		self.steer_K = 1.0

		self._parser = ConfigParser()
		self._parser.read('wheel_config.ini')
		self._steer_idx = int(
			self._parser.get('G29 Racing Wheel', 'steering_wheel'))
		self._throttle_idx = int(
			self._parser.get('G29 Racing Wheel', 'throttle'))
		self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
		self._brake_idx2 = int(self._parser.get('G29 Racing Wheel', 'brake2'))
		self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
		self._gearF_idx = int(self._parser.get('G29 Racing Wheel', 'gearF'))
		self._gearR_idx = int(self._parser.get('G29 Racing Wheel', 'gearR'))
		self.adds = int(self._parser.get('G29 Racing Wheel', 'add'))
		self.minss = int(self._parser.get('G29 Racing Wheel', 'mins'))
		self._handbrake_idx = 0
		# self._handbrake_idx = int(
		# 	self._parser.get('G29 Racing Wheel', 'handbrake'))

	def parse_events(self, world, clock):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return True
			elif event.type == pygame.JOYBUTTONDOWN:
				if event.button == 0:
					world.restart()
				elif event.button == 1:
					world.hud.toggle_info()
				elif event.button == 2:
					world.camera_manager.toggle_camera()
				elif event.button == 3:
					world.next_weather()
				elif event.button == 4:
					self._handbrake_idx = 0
				elif event.button == 5:
					self._handbrake_idx = 1
				elif event.button == self._reverse_idx:
					self._control.gear = 1 if self._control.reverse else -1
				elif event.button == self._gearF_idx:
					self._control.gear = 1
				elif event.button == self._gearR_idx:
					self._control.gear = -1
				elif event.button == self.adds:
					self.steer_K += 0.5 if self.steer_K < 15 else 0
				elif event.button == self.minss:
					self.steer_K -= 0.5 if self.steer_K > 0.5 else 0
				elif event.button == 23:
					world.camera_manager.next_sensor()
				world.steer_k_copy = self.steer_K

			elif event.type == pygame.KEYUP:
				if self._is_quit_shortcut(event.key):
					return True
				elif event.key == K_BACKSPACE:
					world.restart()
				elif event.key == K_F1:
					world.hud.toggle_info()
				elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
					world.hud.help.toggle()
				elif event.key == K_TAB:
					world.camera_manager.toggle_camera()
				elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
					world.next_weather(reverse=True)
				elif event.key == K_c:
					world.next_weather()
				elif event.key == K_BACKQUOTE:
					world.camera_manager.next_sensor()
				elif event.key > K_0 and event.key <= K_9:
					world.camera_manager.set_sensor(event.key - 1 - K_0)
				elif event.key == K_r:
					world.camera_manager.toggle_recording()
				if isinstance(self._control, carla.VehicleControl):
					if event.key == K_q:
						self._control.gear = 1 if self._control.reverse else -1
					elif event.key == K_m:
						self._control.manual_gear_shift = not self._control.manual_gear_shift
						self._control.gear = world.player.get_control().gear
						world.hud.notification('%s Transmission' %
											   ('Manual' if self._control.manual_gear_shift else 'Automatic'))
					elif self._control.manual_gear_shift and event.key == K_COMMA:
						self._control.gear = max(-1, self._control.gear - 1)
					elif self._control.manual_gear_shift and event.key == K_PERIOD:
						self._control.gear = self._control.gear + 1
					elif event.key == K_p:
						self._autopilot_enabled = not self._autopilot_enabled
						world.player.set_autopilot(self._autopilot_enabled)
						world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

		if not self._autopilot_enabled:
			if isinstance(self._control, carla.VehicleControl):
				self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
				self._parse_vehicle_wheel()
				self._control.reverse = self._control.gear < 0
			elif isinstance(self._control, carla.WalkerControl):
				self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
			world.player.apply_control(self._control)


	def _parse_vehicle_keys(self, keys, milliseconds):
		self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
		steer_increment = 5e-4 * milliseconds
		if keys[K_LEFT] or keys[K_a]:
			self._steer_cache -= steer_increment
		elif keys[K_RIGHT] or keys[K_d]:
			self._steer_cache += steer_increment
		else:
			self._steer_cache = 0.0
		self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
		self._control.steer = round(self._steer_cache, 1)
		self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
		self._control.hand_brake = keys[K_SPACE]

	def _parse_vehicle_wheel(self):
		numAxes = self._joystick.get_numaxes()
		jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
		# print (jsInputs)
		jsButtons = [float(self._joystick.get_button(i)) for i in
					 range(self._joystick.get_numbuttons())]

		# Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
		# For the steering, it seems fine as it is
		# K1 = self.steer_K  # #default: 0.55, 1.0
		js_steer = jsInputs[self._steer_idx]
		if js_steer > 0.25:
			js_steer = 0.25
		elif js_steer <-0.25:
			js_steer = -0.25
		# steerCmd = self.steer_K * math.tan(3.14159 * jsInputs[self._steer_idx])
		steerCmd = math.tan(3.14159 * js_steer)

		K2 = 1.6  # 1.6
		throttleCmd = K2 + (2.05 * math.log10(
			-0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
		if throttleCmd <= 0:
			throttleCmd = 0
		elif throttleCmd > 1:
			throttleCmd = 1

		js_brake = min(jsInputs[self._brake_idx], jsInputs[self._brake_idx2])
		if js_brake < 0:
			js_brake = 0
		brakeCmd = math.log10(9*(1-js_brake)+1)
		# brakeCmd = 1.6 + (2.05 * math.log10(
		# 	-0.7 * min(jsInputs[self._brake_idx], jsInputs[self._brake_idx2]) + 1.4) - 1.2) / 0.92
		if brakeCmd <= 0:
			brakeCmd = 0
		elif brakeCmd > 1:
			brakeCmd = 1
		# print("js_steer:",jsInputs[self._steer_idx],",js_brake:",jsInputs[self._brake_idx])
		self._control.steer = steerCmd
		self._control.brake = brakeCmd
		self._control.throttle = throttleCmd


		# toggle = jsButtons[self._reverse_idx]
		self._control.hand_brake = bool(self._handbrake_idx)
		# self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

	def _parse_walker_keys(self, keys, milliseconds):
		self._control.speed = 0.0
		if keys[K_DOWN] or keys[K_s]:
			self._control.speed = 0.0
		if keys[K_LEFT] or keys[K_a]:
			self._control.speed = .01
			self._rotation.yaw -= 0.08 * milliseconds
		if keys[K_RIGHT] or keys[K_d]:
			self._control.speed = .01
			self._rotation.yaw += 0.08 * milliseconds
		if keys[K_UP] or keys[K_w]:
			self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
		self._control.jump = keys[K_SPACE]
		self._rotation.yaw = round(self._rotation.yaw, 1)
		self._control.direction = self._rotation.get_forward_vector()

	@staticmethod
	def _is_quit_shortcut(key):
		return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
	def __init__(self, width, height, collectFlg=0):
		self.dim = (width, height)
		font = pygame.font.Font(pygame.font.get_default_font(), 20)
		font_name = 'courier' if os.name == 'nt' else 'mono'
		fonts = [x for x in pygame.font.get_fonts() if font_name in x]
		default_font = 'ubuntumono'
		mono = default_font if default_font in fonts else fonts[0]
		mono = pygame.font.match_font(mono)
		self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
		self._notifications = FadingText(font, (width, 40), (0, height - 40))
		self.help = HelpText(pygame.font.Font(mono, 24), width, height)
		self.server_fps = 0
		self.frame = 0
		self.simulation_time = 0
		self._show_info = True
		self._info_text = []
		self._server_clock = pygame.time.Clock()
		self.rec = []
		self.collectFlg = collectFlg

	def on_world_tick(self, timestamp):
		self._server_clock.tick()
		self.server_fps = self._server_clock.get_fps()
		self.frame = timestamp.frame
		self.simulation_time = timestamp.elapsed_seconds

	def tick(self, world, clock):
		self.rec = []
		self._notifications.tick(world, clock)
		if not self._show_info:
			return
		t = world.player.get_transform()
		v = world.player.get_velocity()
		c = world.player.get_control()
		f = world.player.get_physics_control()
		mp = world.mp
		w = mp.get_waypoint(t.location)
		heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
		heading = 'S' if abs(t.rotation.yaw) > 90.5 else ''
		heading = heading + 'E' if 179.5 > t.rotation.yaw > 0.5 else heading
		heading = heading + 'W' if -0.5 > t.rotation.yaw > -179.5 else heading
		colhist = world.collision_sensor.get_collision_history()
		# laninv = world.lane_invasion_sensor.
		collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
		max_col = max(1.0, max(collision))
		collision = [x / max_col for x in collision]
		# vehicles = world.world.get_actors().filter('vehicle.*')
		tire = f.wheels[2]

		drift = 0.0
		if abs(v.x) + abs(v.y) > 0.2 and c.gear >= 0:
			drift = 180 * np.arctan2(v.y, v.x) / np.pi - t.rotation.yaw
		self._info_text = [
			'Server:  % 16.0f FPS' % self.server_fps,
			'[Sync_mode]' if world.sync_mode else 'Client:  % 16.0f FPS' % clock.get_fps(),
			'',
			'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
			'Friction: % 16.1f ' % (tire.tire_friction),
			'Param:     % 18s' % (world.param['id']+' ('+world.param['sc']+')'),
			'Map:     % 20s' % mp.name.split('/')[-1],
			'',
			'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
			u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
			'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
			# 'Height:  % 18.0f m' % t.location.z,
			'Waypoint: (% 5.1f, % 5.1f)% 1.0f\N{DEGREE SIGN}' % (
				w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw),
			'Drift: % 16.0f\N{DEGREE SIGN} ' % (drift),
			'Steer: %5.2f' % (c.steer),
			# 'Steer_K: %5.2f' % (world.steer_k_copy),
			'']
		if isinstance(c, carla.VehicleControl):
			self._info_text += [
				('Steer:', c.steer, -1.0, 1.0),
				('Throttle:', c.throttle, 0.0, 1.0),
				('Brake:', c.brake, 0.0, 1.0),
				('Reverse:', c.reverse),
				('Hand brake:', c.hand_brake),
				'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
		elif isinstance(c, carla.WalkerControl):
			self._info_text += [
				('Speed:', c.speed, 0.0, 5.556),
				('Jump:', c.jump)]
		self._info_text += [
			'',
			'Collision:',
			collision]
		# if len(vehicles) > 1:
		#     self._info_text += ['Nearby vehicles:']
		#     distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
		#     vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
		#     for d, vehicle in sorted(vehicles):
		#         if d > 200.0:
		#             break
		#         vehicle_type = get_actor_display_name(vehicle, truncate=22)
		#         self._info_text.append('% 4dm %s' % (d, vehicle_type))

		# -----------record---------
		# headers = ['world_x', 'world_y', 'world_heading', 'world_vx', 'world_vy', 'slip_angle', 'yaw_rate', 'steer',
		# 				   'throttle', 'hand_brake', 'brake', 'world_z', 'world_vz', 'pitch']
		if self.collectFlg:
			av = world.player.get_angular_velocity()
			# print(av)
			self.rec.append(t.location.x)
			self.rec.append(t.location.y)
			self.rec.append(t.rotation.yaw)
			self.rec.append(v.x)
			self.rec.append(v.y)
			self.rec.append(drift)
			self.rec.append(av.z)
			self.rec.append(c.steer)
			self.rec.append(c.throttle)
			self.rec.append(c.hand_brake)
			self.rec.append(c.brake)
			self.rec.append(t.location.z)
			self.rec.append(v.z)
			self.rec.append(t.rotation.pitch)

		#  -----------success flag--------------
		colli = sum(colhist.values())
		lan_inv_hist = world.lane_invasion_sensor.get_history()
		tmp = str(lan_inv_hist)
		lan_inv_times = len(lan_inv_hist)
		lan_fail_times = tmp.count('Solid') + tmp.count('Grass') + tmp.count('Curb')
		# d: distance within lane
		d = w.lane_width / 2 - world.player.bounding_box.extent.y - math.sqrt(
			(t.location.x - w.transform.location.x) ** 2 + (t.location.y - w.transform.location.y) ** 2)
		if colli > 0 or lan_fail_times > 0:  # --fail
			world.flg = 1
		else:  # --success or in progress
			yaw_err = abs(w.transform.rotation.yaw - t.rotation.yaw)
			finish_dist = math.sqrt((t.location.x - world.param['spawn'].location.x) ** 2 + (
					t.location.y - world.param['spawn'].location.y) ** 2 + (
											t.location.z - world.param['spawn'].location.z) ** 2) - world.param[
							  'obstacle_ahead']
			if yaw_err < 3 and finish_dist > 10 and drift < 1 and d > 0:  # --finish
				world.flg = 2
		if self.collectFlg:
			world.rec.wr(self.rec, world.flg)

	def toggle_info(self):
		self._show_info = not self._show_info

	def notification(self, text, seconds=2.0):
		self._notifications.set_text(text, seconds=seconds)

	def error(self, text):
		self._notifications.set_text('Error: %s' % text, (255, 0, 0))

	def render(self, display):
		if self._show_info:
			info_surface = pygame.Surface((220, self.dim[1]))
			info_surface.set_alpha(100)
			display.blit(info_surface, (0, 0))
			v_offset = 4
			bar_h_offset = 100
			bar_width = 106
			for item in self._info_text:
				if v_offset + 18 > self.dim[1]:
					break
				if isinstance(item, list):
					if len(item) > 1:
						points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
						pygame.draw.lines(display, (255, 136, 0), False, points, 2)
					item = None
					v_offset += 18
				elif isinstance(item, tuple):
					if isinstance(item[1], bool):
						rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
						pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
					else:
						rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
						pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
						f = (item[1] - item[2]) / (item[3] - item[2])
						if item[2] < 0.0:
							rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
						else:
							rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
						pygame.draw.rect(display, (255, 255, 255), rect)
					item = item[0]
				if item:  # At this point has to be a str.
					surface = self._font_mono.render(item, True, (255, 255, 255))
					display.blit(surface, (8, v_offset))
				v_offset += 18
		self._notifications.render(display)
		self.help.render(display)


# ==============================================================================
# -- record -----------------------------------------------------------------------
# ==============================================================================
class Record(object):
	def __init__(self, attempt, param):
		headers = ['world_x', 'world_y', 'world_heading', 'world_vx', 'world_vy', 'slip_angle', 'yaw_rate', 'steer',
				   'throttle', 'hand_brake', 'brake', 'world_z', 'world_vz', 'pitch']
		os.makedirs('./csv/24param' + param['id'], exist_ok=True)
		t = time.localtime()
		self.save_name = ('./csv/24param' + param['id'] + '/' + time.strftime("%m%d%H%M%S", t) + 'param' + param[
			'id'] + '_traj' + str(attempt))
		# FailFlag = rec.collisionFlag or rec.destinationFlag or rec.awayFlag
		self.file = open(self.save_name + '.csv', 'w')
		self.writer = csv.writer(self.file)
		self.writer.writerow(headers)

	def wr(self, rec, flg):
		if flg == 0:  # 0 for on process, 1 for fail, 2 or above for success
			self.writer.writerow(rec)
		elif flg == 1:
			self.close()
			if os.path.exists(self.save_name + '.csv'):
				os.rename(self.save_name + '.csv', self.save_name + '_fail.csv')
				print('[' + self.save_name + '_fail.csv] saved')
		elif flg == 2:
			self.close()
			if os.path.exists(self.save_name + '.csv'):
				os.rename(self.save_name + '.csv', self.save_name + '_success.csv')
				print('[' + self.save_name + '_success.csv] saved')
		else:
			print('flag error')

	def close(self):
		if self.file is not None:
			self.file.close()


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
	def __init__(self, font, dim, pos):
		self.font = font
		self.dim = dim
		self.pos = pos
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)

	def set_text(self, text, color=(255, 255, 255), seconds=2.0):
		text_texture = self.font.render(text, True, color)
		self.surface = pygame.Surface(self.dim)
		self.seconds_left = seconds
		self.surface.fill((0, 0, 0, 0))
		self.surface.blit(text_texture, (10, 11))

	def tick(self, _, clock):
		delta_seconds = 1e-3 * clock.get_time()
		self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
		self.surface.set_alpha(500.0 * self.seconds_left)

	def render(self, display):
		display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
	def __init__(self, font, width, height):
		lines = __doc__.split('\n')
		self.font = font
		self.dim = (680, len(lines) * 22 + 12)
		self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
		self.seconds_left = 0
		self.surface = pygame.Surface(self.dim)
		self.surface.fill((0, 0, 0, 0))
		for n, line in enumerate(lines):
			text_texture = self.font.render(line, True, (255, 255, 255))
			self.surface.blit(text_texture, (22, n * 22))
			self._render = False
		self.surface.set_alpha(220)

	def toggle(self):
		self._render = not self._render

	def render(self, display):
		if self._render:
			display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
	def __init__(self, world, parent_actor, hud):
		self.sensor = None
		self.history = []
		self._parent = parent_actor
		self.hud = hud
		bp = world.bp.find('sensor.other.collision')
		self.sensor = world.world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

	def get_collision_history(self):
		history = collections.defaultdict(int)
		for frame, intensity in self.history:
			history[frame] += intensity
		return history

	@staticmethod
	def _on_collision(weak_self, event):
		self = weak_self()
		if not self:
			return
		actor_type = get_actor_display_name(event.other_actor)
		self.hud.notification('Collision with %r' % actor_type)
		impulse = event.normal_impulse
		intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
		self.history.append((event.frame, intensity))
		if len(self.history) > 4000:
			self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
	def __init__(self, world, parent_actor, hud):
		self.sensor = None
		self._parent = parent_actor
		self.hud = hud
		self.history = []
		bp = world.bp.find('sensor.other.lane_invasion')
		self.sensor = world.world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

	@staticmethod
	def _on_invasion(weak_self, event):
		self = weak_self()
		if not self:
			return
		lane_types = set(x.type for x in event.crossed_lane_markings)
		text = ['%r' % str(x).split()[-1] for x in lane_types]
		self.hud.notification('Crossed line %s' % ' and '.join(text))
		self.history.append((event.frame, lane_types))

	def get_history(self):
		return self.history


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
	def __init__(self, world, parent_actor):
		self.sensor = None
		self._parent = parent_actor
		self.lat = 0.0
		self.lon = 0.0
		bp = world.bp.find('sensor.other.gnss')
		self.sensor = world.world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
		# We need to pass the lambda a weak reference to self to avoid circular
		# reference.
		weak_self = weakref.ref(self)
		self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

	@staticmethod
	def _on_gnss_event(weak_self, event):
		self = weak_self()
		if not self:
			return
		self.lat = event.latitude
		self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
	def __init__(self, world, parent_actor, hud):
		self.sensor = None
		self.surface = None
		self._parent = parent_actor
		self.hud = hud
		self.recording = False
		self._camera_transforms = [
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
			carla.Transform(carla.Location(x=1.6, z=1.7))]
		self.transform_index = 1
		self.sensors = [
			['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
		self.world = world
		bp_library = self.world.bp
		for item in self.sensors:
			bp = bp_library.find(item[0])
			if item[0].startswith('sensor.camera'):
				bp.set_attribute('image_size_x', str(hud.dim[0]))
				bp.set_attribute('image_size_y', str(hud.dim[1]))
			elif item[0].startswith('sensor.lidar'):
				bp.set_attribute('range', '50')
			item.append(bp)
		self.index = None

	def toggle_camera(self):
		self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
		self.sensor.set_transform(self._camera_transforms[self.transform_index])

	def set_sensor(self, index, notify=True):
		index = index % len(self.sensors)
		needs_respawn = True if self.index is None \
			else self.sensors[index][0] != self.sensors[self.index][0]
		if needs_respawn:
			if self.sensor is not None:
				self.sensor.destroy()
				self.surface = None
			self.sensor = self.world.world.spawn_actor(
				self.sensors[index][-1],
				self._camera_transforms[self.transform_index],
				attach_to=self._parent)
			# We need to pass the lambda a weak reference to self to avoid
			# circular reference.
			weak_self = weakref.ref(self)
			self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
		if notify:
			self.hud.notification(self.sensors[index][2])
		self.index = index

	def next_sensor(self):
		self.set_sensor(self.index + 1)

	def toggle_recording(self):
		self.recording = not self.recording
		self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

	def render(self, display):
		if self.surface is not None:
			display.blit(self.surface, (0, 0))

	@staticmethod
	def _parse_image(weak_self, image):
		self = weak_self()
		if not self:
			return
		if self.sensors[self.index][0].startswith('sensor.lidar'):
			points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
			points = np.reshape(points, (int(points.shape[0] / 4), 4))
			lidar_data = np.array(points[:, :2])
			lidar_data *= min(self.hud.dim) / 100.0
			lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
			lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
			lidar_data = lidar_data.astype(np.int32)
			lidar_data = np.reshape(lidar_data, (-1, 2))
			lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
			lidar_img = np.zeros(lidar_img_size)
			lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
			self.surface = pygame.surfarray.make_surface(lidar_img)
		else:
			image.convert(self.sensors[self.index][1])
			array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
			array = np.reshape(array, (image.height, image.width, 4))
			array = array[:, :, :3]
			array = array[:, :, ::-1]
			self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
		if self.recording:
			image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
	pygame.init()
	pygame.font.init()
	world = None
	param = choose_parameter(args.paramid)

	try:
		client = carla.Client(args.host, args.port)
		client.set_timeout(2.0)

		display = pygame.display.set_mode(
			(args.width, args.height),
			pygame.HWSURFACE | pygame.DOUBLEBUF)

		hud = HUD(args.width, args.height, collectFlg=1)
		#        world = World(client.get_world(), hud, args.filter)
		world = World(client,client.load_world(param['map']), hud, param['ego'], param_ID=args.paramid, collectFlg=1, fixed_seconds=0.005)
		controller = DualControl(world, args.autopilot)

		clock = pygame.time.Clock()
		while True:
			clock.tick_busy_loop(120)
			if controller.parse_events(world, clock):
				return
			world.tick(clock)
			world.render(display)
			pygame.display.flip()

	finally:

		if world is not None:
			world.destroy()

		pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
	argparser = argparse.ArgumentParser(
		description='CARLA Manual Control Client')
	argparser.add_argument(
		'-v', '--verbose',
		action='store_true',
		dest='debug',
		help='print debug information')
	argparser.add_argument(
		'--host',
		metavar='H',
		default='127.0.0.1',
		help='IP of the host server (default: 127.0.0.1)')
	argparser.add_argument(
		'-p', '--port',
		metavar='P',
		default=2000,
		type=int,
		help='TCP port to listen to (default: 2000)')
	argparser.add_argument(
		'-a', '--autopilot',
		action='store_true',
		help='enable autopilot')
	argparser.add_argument(
		'--res',
		metavar='WIDTHxHEIGHT',
		default='1600x900',
		help='window resolution (default: 1280x720)')
	argparser.add_argument(
		'--paramid',
		metavar='PATTERN',
		default=11,
		help='paramid (default: 5)')
	# argparser.add_argument(
	#     '--param',
	#     metavar='PARAMETER',
	#     default=99,  # 99 for test
	#     help='PARAMETER ID')
	args = argparser.parse_args()

	args.width, args.height = [int(x) for x in args.res.split('x')]

	log_level = logging.DEBUG if args.debug else logging.INFO
	logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

	logging.info('listening to server %s:%s', args.host, args.port)

	print(__doc__)

	try:

		game_loop(args)

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')


if __name__ == '__main__':
	main()
