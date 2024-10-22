import sys
from environment import *
import time
import random
import pygame
import csv
import os
import torch
# from tools import SAC_Actor
from SAC_auto_alpha import *
from tools import getHeading, bool2num

# np.random.seed(1234)
def action_range(action,BSrange):
	action = np.clip(action, -1, 1)
	#input action: -1 ~ 1
	a=[0.0,0.0]
	brake_range = BSrange[0]
	steer_range = BSrange[1]
	a[0] = float((action[0] +1)/2* (steer_range[1] - steer_range[0]) + steer_range[0])
	a[1] = float((action[1] +1)/2* (brake_range[1] - brake_range[0]) + brake_range[0])

	return a
def action_rerange(action,BSrange):
	brake_range = BSrange[0]
	steer_range = BSrange[1]
	action[0] = (action[0] - steer_range[0]) / (steer_range[1] - steer_range[0]) * 2 - 1
	action[1] = (action[1] - brake_range[0]) / (brake_range[1] - brake_range[0]) * 2 - 1
	#output action: -1 ~ 1
	return action

def main():
	BSrange = [(0, 0.4), (-1, 1)]  # Boundary of brake values and steering angle values
	parser = argparse.ArgumentParser()
	parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient 0.005
	# parser.add_argument('--target_update_interval', default=1, type=int)
	parser.add_argument('--gradient_steps', default=1, type=int)  # update times,default=1,

	parser.add_argument('--learning_rate', default=3e-4, type=float)  # default=3e-4
	parser.add_argument('--gamma', default=0.97, type=float)  # discount gamma,common 0.99. here 0.96 means will consider future 56 steps
	# https://zhuanlan.zhihu.com/p/345353294

	parser.add_argument('--capacity', default=20000, type=int)  # replay buffer size,default=400000,cpd50000
	parser.add_argument('--iteration', default=26000, type=int)  # num of  games
	parser.add_argument('--batch_size', default=256, type=int)  # mini batch size, default=512

	parser.add_argument('--load', default=True, type=bool)  # load model
	parser.add_argument('--param_ID', default=5, type=int)  # parameter ID in experiments_parameters.py
	parser.add_argument('--sync_mode', default=True, type=bool)  # carla sync_mode
	parser.add_argument('--method', default='DOA')  # DOA,DOAP
	# parser.add_argument('--test_set', default='vehicle')  # friction, mass, vehicle

	args = parser.parse_args()

	pygame.init()
	pygame.font.init()
	print('INIT with parameter ID: ', args.param_ID)
	global env

	# ==============================================================================
	t_fri= np.arange(2.5, 4.6, 0.5)
	t_para = [5,11,6,7,8]#
	t_v = ['ford.mustang','audi.a2','lincoln.mkz_2020','jeep.wrangler_rubicon','nissan.patrol_2021','mercedes.sprinter']
	i = 0
	succ = 0

	for para in t_para:
		args.param_ID = para
		env = environment(BSrange=BSrange, args=args, traj_num=args.param_ID,
		                  testFlag = True, model='sac')  # ly: -----------------------------environment

		action_dim = 2
		control = carla.VehicleControl()
		state = env.getState(control)
		state_dim = len(state)
		print('action_dimension:', action_dim, ' ; state_dimension:', state_dim)
		# Initializing the Agent for SAC and load the trained weights
		agent = SACAgent(args=args, state_dim=state_dim,
		                 action_dim=action_dim)  # ly: -----------------------------agent

		# 5 (a): 'Model_p5', 2110  # old: 'model_p5', 2300
		# 11 (b): 'Model_p11', 2420
		# 6 (c): 'Model_p6', 860
		# 7 (d): 'Model_p7', 1410,1570
		m_path = '../models'
		if args.method == 'DOA':
			if args.param_ID in {5}:
				agent.load(m_path, 2110)
			elif args.param_ID in {7,8}:
				agent.load(m_path, 1410)
			elif args.param_ID in {6}:
				agent.load(m_path, 860)
			elif args.param_ID in {11}:
				agent.load(m_path, 2420)
		elif args.method =='DOAP':
			if args.param_ID in {5}:
				agent.load(m_path, 2110)
			else:
				agent.load(m_path, 2420)
		else:
			print('wrong method')

		# actor = SAC_Actor(state_dim=state_dim, action_dim = action_dim).to(device)


		sec_K = int(200 * env.fixed_seconds)
		max_step = int(env.route_len * 1.5/sec_K)

		# define the headers of the csv file to be saved
		# state = [steer, brake, location.x, location.y, location.z, vx, vy, vz, ego_yaw,ego_pitch, slip,
		# 		# 			         ex, ey, ez, self.e_heading, self.e_d_dis] + future:[x*8,y*8,dz*8,heading*8]
		headers = ['step','steer', 'brake', 'state_x', 'state_y', 'state_z', 'vx', 'vy', 'vz', 'state_yaw', 'pitch', 'slip',
		           'Dis_O1', 'Dis_O2', 'traj_index', 'reward',  'collisionFlag', 'desitinationFlag', 'awayFlag']

		pth = './test/SSR_'+args.method+'/'
		os.makedirs(pth, exist_ok=True)

		for veh in t_v:  # ====each times of games
			for fri in t_fri:
				destinationFlag = False
				collisionFlag = False
				awayFlag = [False, False]
				carla_startFlag = False
				state = env.reset(traj_num=args.param_ID, testFlag = True, friction=fri, testvehicle=True,vehicle =veh)
				save_path = 'P%.0f_'%para + veh +'_%.1f'%(fri)
				save_file = open(pth+save_path + '.csv', 'w')
				writer = csv.writer(save_file)
				writer.writerow(headers)

				# first_step_pass = False
				step = 0
				speed = 0.0
				cte = 0.0
				hae = 0.0
				time_cost = 0.0
				max_reward = None
				min_reward = None
				ep_r = 0.0  ###expectation of reward R

				for step in range(max_step):  # ====each step in one round
					env.render()  # ly:--includes world.tick()
					# plt.clf()
					carla_startFlag = True

					if step == 0:
						steer = 0.0
						brake = 0.5
						hand_brake = False
					else:
						# ly: TEST
						action = agent.test(tState)
						action = action_range(action,BSrange)
						# action = np.reshape(action, [1, 2])

						steer = action[0]
						brake = action[1]
					# if i%5==0:
					# 	agent.writer.add_scalar('TEST/Control/iteration_'+str(i)+'/steer', steer, global_step = step)
					# 	agent.writer.add_scalar('TEST/Control/iteration_'+str(i)+'/brake', brake, global_step = step)
					# print(step,steer,brake)
					next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer,
					                                                                                 brake=brake)
					ep_r = ep_r + reward
					_next_state = np.reshape(next_state, [1, state_dim])
					next_state = _next_state
					tState = next_state


					max_s_flag = 1 if step == max_step-1 else 0
					endFlag = collisionFlag or destinationFlag or max_s_flag

					# prepare the state information to be saved
					# In env.getState: state = [steer, brake, location.x, location.y, location.z, vx, vy, vz, ego_yaw,ego_pitch, slip,
					# 			         ex, ey, ez, self.e_heading, self.e_d_dis] + future:[x*8,y*8,dz*8,heading*8]
					# headers = ['step','steer', 'brake', 'state_x', 'state_y', 'state_z', 'vx', 'vy', 'vz', 'state_yaw', 'pitch', 'slip',
					# 	           'Dis_O1', 'Dis_O2', 'traj_index', 'reward',  'collisionFlag', 'desitinationFlag', 'awayFlag']
					x = next_state[0, 2]
					y = next_state[0, 3]
					z = next_state[0, 4]
					lo1=env.world.obstacle.get_location()
					lo2=env.world.obs2.get_location()
					dis1 = np.sqrt((x-lo1.x)**2+(y-lo1.y)**2+(z-lo1.z)**2)
					dis2 = np.sqrt((x-lo2.x)**2+(y-lo2.y)**2+(z-lo2.z)**2)
					traj_index = env.traj_index
					cf = bool2num(collisionFlag)
					df = bool2num(destinationFlag)
					af = bool2num(awayFlag[1])

					writer.writerow([step]+list(next_state[0, 0:11])+[dis1,dis2,traj_index,reward,cf,df,af])
					# writer.writerow(next_state)

					if endFlag:
						print(save_path,'End flag: ', '[Collision]' if collisionFlag else '',
						      '\033[32m[Destination & success]\033[0m' if destinationFlag else '',
						      '[Away]' if awayFlag[1] else '',
						      '[Away_0]' if awayFlag[0] else '',
						      '[Max steps]' if max_s_flag else '','.')
						break
				save_file.close()
				if not destinationFlag:
					time.sleep(0.002)
					os.rename(pth+save_path+'.csv', pth+save_path+'_fail.csv')
				else:
					succ +=1
				i = i +1
		if env.world is not None:
			env.world.destroy()
		print('i=',i,',succ=',succ,',rate:',succ/i)

	# END:
if __name__ == "__main__":
	global env
	try:
		main()
	except KeyboardInterrupt:
		print('######### KeyboardInterrupt, Cleanning ##########')
		# T_all = time.time() - T
		# T_hours = int(T_all / 3600)
		# T_minutes = int((T_all % 3600) / 60)
		# print(f'Total time: {T_hours} hours {T_minutes} minutes')
		# plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.png')
		# plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.svg')
		# plt.show()
		# plt.close(fig)
	finally:
		if env.world is not None:
			env.world.destroy()
			settings = env.world.world.get_settings()
			settings.synchronous_mode = False
			settings.fixed_delta_seconds = None
			env.world.world.apply_settings(settings)
		pygame.quit()
		print("world.destroy, pygame.quit")
