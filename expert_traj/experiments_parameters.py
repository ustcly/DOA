import numpy as np
import carla
import random


def choose_parameter(num):
	param = {'spawn': carla.Transform(), 'weather': carla.WeatherParameters()}
	if num == 0:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.audi.tt'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.ClearNoon  # none affect to friction.
		param['friction'] = 1.7
		param['speed'] = 80.0  # km/h
		param['obstacle_ahead'] = 30.0  # m
		return param
	############################################################
	############################################################
	elif num == 5:
		# flat, (a)
		param['id'] = str(num)
		param['sc'] = 'a'
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.03
		param['spawn'].rotation.yaw = 90  # Town1:0, Town5: 90
		param['LR'] = 1  # drift to right as 1, left is -1
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 120#115.0  # km/h
		param['obstacle_ahead'] = 21 + 5.5 +5# pure distance + 6 for the center of two cars. 30
		param['ob2'] = 17+2
		param['lane2']=2   #3.5m each lane
		return param
	elif num == 11:
		# flat, (b)
		param['id'] = str(num)
		param['sc'] = 'b'
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = -229.34  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = 37.54  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 10.3
		param['spawn'].rotation.yaw = 270  # Town1:0, Town5: 90
		param['LR'] = -1  # drift to right as 1, left is -1
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 120.0  # km/h
		param['obstacle_ahead'] = 20 + 5.5 # m
		param['ob2'] = 8+2
		param['lane2']=1
		return param
	elif num == 6:
		# Going upslop ,(c)
		param['id'] = str(num)
		param['sc'] = 'c'
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = -20  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = 194.57  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.47
		param['spawn'].rotation.yaw = 180  # Town1:0, Town5: 90
		param['LR'] = 1  # drift to right as 1, left is -1
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 122.0  # km/h
		param['obstacle_ahead'] = 19 + 5.5 # m
		param['ob2'] = 14+2
		param['lane2']=2
		return param
	elif num == 7:
		# Going downslop (d)
		param['id'] = str(num)
		param['sc'] = 'd'
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = -70  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = 208.59  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 7#5.9
		param['spawn'].rotation.yaw = 0  # Town1:0, Town5: 90
		param['LR'] = -1  # drift to right as 1, left is -1
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 115.0  # km/h
		param['obstacle_ahead'] = 20 + 5.5  # m
		param['ob2'] = 14+2
		param['lane2']=-2
		return param
	elif num == 8:
		# Going downslop (e)
		param['id'] = str(num)
		param['sc'] = 'e'
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = -70 # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -190.0 # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 9.72
		param['spawn'].rotation.yaw = 0  # Town1:0, Town5: 90
		param['LR'] = 1  # drift to right as 1, left is -1
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 115.0  # km/h
		param['obstacle_ahead'] = 20 + 5.5   # m
		param['ob2'] = 8+2
		param['lane2']=-1
		return param
	############################################################
	############################################################
	elif num == 1:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.audi.tt'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 2.0
		param['speed'] = 70.0  # km/h
		param['obstacle_ahead'] = 20.0  # m
		return param
	elif num == 2:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 1.7
		param['speed'] = 75.0  # km/h
		param['obstacle_ahead'] = 25.0  # m
		return param
	elif num == 3:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.dodge.charger_2020'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 2.3
		param['speed'] = 85.0  # km/h
		param['obstacle_ahead'] = 30.0  # m
		return param
	elif num == 4:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.dodge.charger_2020'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.0
		param['speed'] = 95.0  # km/h
		param['obstacle_ahead'] = 30.0  # m
		return param
	elif num == 9:
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = 85.6  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -145.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 10
		param['spawn'].rotation.yaw = 270.16  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.5
		param['speed'] = 100.0  # km/h
		param['obstacle_ahead'] = 30.0  # m
		return param
	elif num == 10:
		param['id'] = str(num)
		param['map'] = 'Town03'
		param['ego'] = 'vehicle.ford.mustang'  # ford.mustang
		param['spawn'].location.x = 85.6  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -145.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 10
		param['spawn'].rotation.yaw = 270.16  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 4.5
		param['speed'] = 100.0  # km/h
		param['obstacle_ahead'] = 30.0  # m
		return param
	else:  # --  99 for test
		param['id'] = str(num)
		param['map'] = 'Town05'
		param['ego'] = 'vehicle.audi.tt'  # ford.mustang
		param['spawn'].location.x = 195.67  # 195.6 # Town1:-60, Town5: 195.67
		param['spawn'].location.y = -100.8  # -100.8 #Town1(R Lane:140.5, L Lane:137.1) Town5:-100.8
		param['spawn'].location.z = 0.1
		param['spawn'].rotation.yaw = 89.0  # Town1:0, Town5: 90
		param['weather'] = carla.WeatherParameters.MidRainSunset  # none affect to friction.
		param['friction'] = 2.0
		param['speed'] = 70.0  # km/h
		param['obstacle_ahead'] = 20.0  # m
		return param
