import sys
from environment import *
# from SACAgent import *
from SAC_auto_alpha import *
import time
import random
import pygame
import numpy as np
from collections import deque
from torch import cuda
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

# from agents.navigation.basic_agent import BasicAgent

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

########SAC#######
def main():

	BSrange = [(0, 0.2), (-1, 1)]  # Boundary of brake values and steering angle values
	parser = argparse.ArgumentParser()

	parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient 0.005
	# parser.add_argument('--target_update_interval', default=1, type=int)
	parser.add_argument('--gradient_steps', default=1, type=int)  # update times,default=1,

	parser.add_argument('--learning_rate', default=3e-4, type=float)  # default=3e-4
	parser.add_argument('--gamma', default=0.97, type=float)  # discount gamma,common 0.99. here 0.96 means will consider future 56 steps
	# https://zhuanlan.zhihu.com/p/345353294

	parser.add_argument('--capacity', default=40000, type=int)  # replay buffer size,default=400000,cpd50000
	parser.add_argument('--iteration', default=1600, type=int)  # num of  games
	parser.add_argument('--batch_size', default=256, type=int)  # mini batch size, default=512

	parser.add_argument('--load', default = False, type=bool)  # load model
	parser.add_argument('--param_ID', default = 5 , type=int)  # parameter ID in experiments_parameters.py
	parser.add_argument('--sync_mode', default=True, type=bool)  # carla sync_mode

	args = parser.parse_args()

	pygame.init()
	pygame.font.init()
	print('INIT with parameter ID: ', args.param_ID)
	global env

	# ==============================================================================
	env = environment(BSrange=BSrange, args=args, traj_num=args.param_ID,
	                  model='sac')  # ly: -----------------------------environment

	action_dim = 2
	control = carla.VehicleControl()
	state = env.getState(control)
	state_dim = len(state)
	print('action_dimension:', action_dim, ' ; state_dimension:', state_dim)

	agent = SACAgent(args=args, state_dim=state_dim,
	                 action_dim=action_dim)  # ly: -----------------------------agent

	if args.load:
		# 5 (a): 'Model_p5', 2110  # old: 'model_p5', 2300
		# 11 (b): 'Model_p11', 2420
		# 6 (c): 'Model_p6', 860
		# 7 (d): 'Model_p7', 700
		# 8 (e): 'model_p8',
		agent.load('Model_p7', 700)# modeldir, epoch

	print("====================================")
	print("Collection Experience...")
	print("====================================")
	epr_history = deque(maxlen=8)
	destinationFlag = False
	epr_history.append([0, 0, 0, 0]) #[success_flag, ep_r, max_r, min_r]
	collisionFlag = False
	awayFlag = [False, False]
	carla_startFlag = False
	T = time.time()

	sec_K = int(200 * env.fixed_seconds)
	max_step = int(env.route_len * 1.5/sec_K)
	# update_times = 20 # as well as update interval
	test_interval = 10
	update_start = False
	Test_flag = False

	# plt_range = int(args.iteration/test_interval)+3
	plt_x = []
	plt_y = []
	plt_min = []
	plt_max = []
	plt_i = 0
	plt.ion()
	fig, ax = plt.subplots()
	line, = ax.plot(plt_x, plt_y)
	ax.set_xlabel('Test Iteration')
	ax.set_ylabel('Epoch Reward')
	ax.set_title('Real-time Rewards')
	plt.show(block=False)

	for i in range(args.iteration):  # ====each times of games
		if update_start and i % test_interval == 0:
			Test_flag = True
			state = env.reset(traj_num=args.param_ID,testFlag=True)
			print('\n\033[33m**############# RESET and TESTING ',plt_i,' ############**\033[0m')
		else:
			Test_flag = False
			state = env.reset(traj_num=args.param_ID)
			print('\n----- RESET and Exploring: %d' % i)

		t0 = time.time()

		# first_step_pass = False
		step = 0
		speed = 0.0
		cte = 0.0
		hae = 0.0
		time_cost = 0.0
		max_reward = None
		min_reward = None
		ep_r = 0.0  ###expectation of reward R


		if i == 0:
			for _s in range(int(env.route_len/sec_K)):
				# put reference to replay buffer.
				env.render()
				steer = env.route[_s * sec_K,7] * env.LR
				brake = env.route[_s * sec_K,10]
				action = np.array([steer, brake])
				next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer,brake=brake)
				next_state = np.reshape(next_state, [1, state_dim])
				if _s > 0:
					action = action_rerange(action,BSrange)
					agent.replay_buffer.store(tState, action, reward, next_state, int(destinationFlag))
				tState = next_state
			continue



		for step in range(max_step):  # ====each step in one round
			env.render()  # ly:--includes world.tick()
			# plt.clf()
			carla_startFlag = True

			#######------------------Exploration--------------------#########
			if not Test_flag:  # Explore and put in replay buffer
				# ly: take an action each time
				# if step: #time.time()-t0 >= 0.05:
				if step < 1:
					action = np.random.uniform(-1,1,2)
				elif i < 1000:
					P = np.random.binomial(1,i/(i+50),1)
					action_A = agent.choose_action(tState)
					steer = env.route[min(step * sec_K,env.route_len-1),7] * env.LR
					brake = env.route[min(step * sec_K,env.route_len-1),10]
					action_R = np.array([steer, brake])
					action = P * action_A + (1-P)*action_R + np.random.normal(0,1/np.ceil(i/100),2)
				else:
					action = agent.choose_action(tState)
				action = action_range(action,BSrange) # with clip

				steer = action[0]  # [0, 0]
				brake = action[1]  # [0, 1]
				# print("TRAIN: i=", i, ", step step=", step, "; mapped steer: %.3f" % (steer),
				# 	  ", brake: %.3f" % (brake))
				# if i%5==0:
				# 	agent.writer.add_scalar('Control/iteration_'+str(i)+'/steer', steer, global_step = step)
				# 	agent.writer.add_scalar('Control/iteration_'+str(i)+'/brake', brake, global_step = step)

				next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer,brake=brake)
				next_state = np.reshape(next_state, [1, state_dim])
				ep_r = ep_r + reward

				if step > 0:
					action = action_rerange(action,BSrange)
					agent.replay_buffer.store(tState, action, reward, next_state, int(destinationFlag))
				tState = next_state
			#######------------------Test--------------------#########
			else: # ly: Test.

				if step == 0:
					steer = 0.0
					brake = 0.0
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

				next_state, reward, collisionFlag, destinationFlag, awayFlag, control = env.step(steer=steer,
				                                                                                 brake=brake)
				if plt_i % 8 == 0:
					print("step: ", step, "; mapped steer: %.3f" % (steer),
				      ", brake: %.3f" % (brake), "; reward: %.2f" % (reward))
				ep_r = ep_r + reward
				_next_state = np.reshape(next_state, [1, state_dim])
				next_state = _next_state
				tState = next_state

			# Continue in [each step]
			if max_reward is None or max_reward < reward:
				max_reward = reward
			if min_reward is None or min_reward > reward:
				min_reward = reward
			# print("Reward: %.2f" % reward)
			max_s_flag = 1 if step == max_step-1 else 0
			endFlag = collisionFlag or destinationFlag or awayFlag[0] or max_s_flag

			vx = env.velocity_local[0]
			vy = env.velocity_local[1]
			# speed = speed + np.sqrt(vx * vx + vy * vy)
			# cte = cte + tState[0, 2]
			# hae = hae + abs(tState[0, 4])

			if endFlag:
				print('End flag: ', '[Collision]' if collisionFlag else '',
				      '\033[32m[Destination & success]\033[0m' if destinationFlag else '',
				      '[Away]' if awayFlag[1] else '',
				      '[Max steps]' if max_s_flag else '','.')
				break

			# first_step_pass = True
		# end of [each step]

		time_cost = time.time() - t0
		print("ep_R= %.2f, maxR= %.2f, minR= %.2f" % (ep_r, max_reward, min_reward))
		# speed = speed / step
		# cte = cte / step
		# hae = hae / step
		update_start = True if agent.replay_buffer.num_transition > args.batch_size * 6 else False

		#######------------------Learn--------------------#########
		if update_start and not Test_flag:
			# if agent.replay_buffer.num_transition < args.capacity:
			# 	k = 1 + agent.replay_buffer.num_transition / args.capacity
			# else:
			# 	k = 2
			# u_times = int(k * update_times)
			# kk = '> Buffer_max; _' if k>=2 else '; ___'
			# print('\n\033[34m*************TRAIN**************\033[0m')
			if agent.replay_buffer.num_transition < args.capacity:
				u_times = int(2 * agent.replay_buffer.num_transition/args.batch_size)
				kk = '; ___'
			else:
				u_times = int(2 * args.capacity/args.batch_size)
				kk = '> Buffer_max; _'
			print("Explored_steps:", str(agent.replay_buffer.num_transition),kk,'[Train', u_times,
			      "times]___")
			for _u in range(u_times):
				agent.learn()


		#######------------------Analysis test result--------------------#########
		if Test_flag or i == args.iteration-1:
			###------plot--------#
			plt_y.append(ep_r)
			plt_x.append(plt_i)
			plt_max.append(max_reward)
			plt_min.append(min_reward)
			line.set_xdata(plt_x)
			line.set_ydata(plt_y)

			ax.fill_between(plt_x, plt_min, plt_max, color='#FFDDDD')

			ax.relim()
			ax.autoscale_view()
			fig.canvas.draw()
			fig.canvas.flush_events()
			# plt.pause(0.1)
			plt_i = plt_i + 1

			###------Analysis--------#
			if destinationFlag and not (collisionFlag or awayFlag[1]):
				epr_history.append([1, ep_r, max_reward, min_reward])
			else:
				epr_history.append([0, ep_r, max_reward, min_reward])

			epr_his = np.array(epr_history)
			if sum(epr_his[:, 0]) > 6:  # If continus 8 destinationFlag in test
				var_ep = np.var(epr_his[:, 1])
				var_max = np.var(epr_his[:, 2])
				var_min = np.var(epr_his[:, 3])
				var_r = var_ep + var_max + var_min
				print('\033[32mContinues succeed 8 times in TEST.\033[0m Standard Deviation of reward:')
				print('SD_ep: %.2f, SD_max: %.2f, SD_min: %.2f'%(np.sqrt(var_ep),np.sqrt(var_max),np.sqrt(var_min))) #Standard Deviation
				if ep_r > 900 or i >= args.iteration - 8: # reward1: ep_r > 1100, reward2: ep_r > 380
					agent.save(i, args.param_ID, plt_y)
					if var_r < 0.004 * np.mean(epr_his[:, 1]) or i == args.iteration - 1:  # If variance is less than 1 then finish
						print('######### Training Finished ##########')
						print('Total iteration is (final_i + 1): ', i+1)
						T_all = time.time() - T
						T_hours = int(T_all / 3600)
						T_minutes = int((T_all % 3600) / 60)
						print(f'Total time: {T_hours} hours {T_minutes} minutes')
						# plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.png')
						plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.svg')
						plt.show()
						plt.close(fig)
						break
			elif i >= args.iteration - 2*test_interval:
				agent.save(i, args.param_ID, plt_y)
				plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.svg')
			# elif destinationFlag:
			# 	print(ep_r)
			# 	agent.save(i, args.param_ID, plt_y)
				# plt.savefig('p'+str(args.param_ID)+time.strftime("%m%d%H%M%S", time.localtime()) +'.svg')

	# agent.writer.add_scalar('Metrics/ep_r', ep_r, global_step=i)
	# agent.writer.add_scalar('Metrics/time_cost', time_cost, global_step=i)
	# agent.writer.add_scalar('Metrics/avg_speed', speed, global_step=i)
	# agent.writer.add_scalar('Metrics/avg_cross_track_error', cte, global_step=i)
	# agent.writer.add_scalar('Metrics/avg_heading_error', hae, global_step=i)
	# agent.writer.add_scalar('Metrics/reward_every_second', ep_r/time_cost, global_step=i)
	#
	# agent.writer.add_scalar('Physics/Tire_friction', env.tire_friction, global_step = i)
	# agent.writer.add_scalar('Physics/Mass', env.mass, global_step=i)
	#
	# if i % 10 ==0 and agent.replay_buffer.num_transition > 3000:
	# 	agent.writer.add_scalar('Metrics_test/ep_r', ep_r, global_step=i)
	# 	agent.writer.add_scalar('Metrics_test/time_cost', time_cost, global_step=i)
	# 	agent.writer.add_scalar('Metrics_test/avg_speed', speed, global_step=i)
	# 	agent.writer.add_scalar('Metrics_test/avg_cross_track_error', cte, global_step=i)
	# 	agent.writer.add_scalar('Metrics_test/avg_heading_error', hae, global_step=i)
	# 	agent.writer.add_scalar('Metrics_test/reward_every_second', ep_r/time_cost, global_step=i)
	#
	# 	agent.writer.add_scalar('Physics_test/Tire_friction', env.tire_friction, global_step = i)
	# 	agent.writer.add_scalar('Physics_test/Mass', env.mass, global_step=i)

	# cuda.empty_cache()
	# print(cuda.list_gpu_processes())

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
