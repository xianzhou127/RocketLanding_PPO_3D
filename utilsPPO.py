import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import numpy as np

import matplotlib.pyplot as plt
import os

from matplotlib.ticker import MultipleLocator
class BetaActor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(BetaActor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.alpha_head = nn.Linear(net_width, action_dim)
		self.beta_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))

		alpha = F.softplus(self.alpha_head(a)) + 1.0
		beta = F.softplus(self.beta_head(a)) + 1.0

		return alpha,beta

	def get_dist(self,state):
		alpha,beta = self.forward(state)
		dist = Beta(alpha, beta)
		return dist

	def deterministic_act(self, state):
		alpha, beta = self.forward(state)
		mode = (alpha) / (alpha + beta)
		return mode

class GaussianActor_musigma(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(GaussianActor_musigma, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.sigma_head = nn.Linear(net_width, action_dim)

	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		sigma = F.softplus( self.sigma_head(a) )
		return mu,sigma

	def get_dist(self, state):
		mu,sigma = self.forward(state)
		dist = Normal(mu,sigma)
		return dist

	def deterministic_act(self, state):
		mu, sigma = self.forward(state)
		return mu


class GaussianActor_mu(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, log_std=0):
		super(GaussianActor_mu, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.mu_head = nn.Linear(net_width, action_dim)
		self.mu_head.weight.data.mul_(0.1)
		self.mu_head.bias.data.mul_(0.0)

		self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

	def forward(self, state):
		a = torch.relu(self.l1(state))
		a = torch.relu(self.l2(a))
		mu = torch.sigmoid(self.mu_head(a))
		return mu

	def get_dist(self,state):
		mu = self.forward(state)
		action_log_std = self.action_log_std.expand_as(mu)
		action_std = torch.exp(action_log_std)

		dist = Normal(mu, action_std)
		return dist

	def deterministic_act(self, state):
		return self.forward(state)


class Critic(nn.Module):
	def __init__(self, state_dim,net_width):
		super(Critic, self).__init__()

		self.C1 = nn.Linear(state_dim, net_width)
		self.C2 = nn.Linear(net_width, net_width)
		self.C3 = nn.Linear(net_width, 1)

	def forward(self, state):
		v = torch.tanh(self.C1(state))
		v = torch.tanh(self.C2(v))
		v = self.C3(v)
		return v

def str2bool(v):
	'''transfer str to bool for argparse'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
		return False
	else:
		print('Wrong Input.')
		raise


def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	return  2*(a-0.5)*max_action

def Reward_adapter(r, EnvIdex):
	# For BipedalWalker
	if EnvIdex == 0 or EnvIdex == 1:
		if r <= -100: r = -1
	# For Pendulum-v0
	elif EnvIdex == 3:
		r = (r + 8) / 8
	return r

def evaluate_policy(env, agent, max_action, turns, render = False, plot = False, record = False):           
	
	dt = 0.02
	m = 25000.0             
	g = 9.81                
	rocket_length = 40.0    
	L = rocket_length / 2.0                 
	I = (1.0 / 12.0) * m * (rocket_length**2)
	H = 40
	MAX_THRUST_ACC = 2.0 * g  
	MIN_THRUST_ACC = 0.4 
	MAX_GIMBAL = np.deg2rad(20.0)
	MAX_GIMBAL_SPEED = np.deg2rad(30.0)
	target_state = np.array([0.0, 0.2 + H/2, 0.0, 0.0, 0.0, 0.0])
	state = np.array([20.0, 400.0, -10.0, -40.0, np.deg2rad(90), 0.0])
	state = np.array([np.float32(108.8021), np.float64(401.968798828125), np.float32(19.011578),   5, np.float64(-1.8786757459513606), np.float64(0.009725803833688767)])
	env_state = {
            'x': state[0], 'y': state[1], 'vx': state[2], 'vy': state[3],
            'theta': state[4], 'vtheta': state[5],
            'phi': 0, 'f': 0, 't': 0, 'action_': np.zeros(2)
        }
	success = np.array([0,0,0,0,0])
	final_x_list = []
	final_y_list = []
	total_scores = 0
	turn_scores = 0
	state_init = np.array([0,0,0,0,0,0])
	state_final = np.array([0,0,0,0,0,0])
	for j in range(turns):
		if plot:
			s, info = env.reset(state_dict = env_state)
		else:
			s, info = env.reset()
		# state_init = [s[0]*100, s[1]*500+target_state[1], s[2]*20, info['vy'], s[4]* np.deg2rad(90), s[5]* np.deg2rad(5)]
		done = False
		history_x = []
		history_u = []
		obs = []
		step = 0
		turn_scores = 0
		while not done:
			if render: env.render()  # 如果需要，在这里渲染
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			# print(act)
			# act = np.array([0,0,1])
			s_next, r, dw, tr, info = env.step(act)
			turn_scores += r
			total_scores += r
			step += 1
			s = s_next
			done = (dw or tr)
			
			if plot:
				state = [s_next[0]*100, s_next[1]*500+target_state[1], s_next[2]*20, info['vy'], s_next[4]* np.deg2rad(90), s_next[5]* np.deg2rad(5)]
				throttle_pct = MIN_THRUST_ACC + (1.0 - MIN_THRUST_ACC) * (act[0] + 1.0) / 2.0
				f = throttle_pct * MAX_THRUST_ACC
				phi_target = act[1] * MAX_GIMBAL
				a = [f,phi_target]

				history_u.append(a.copy())
				history_x.append(state.copy())

			if done:
				# obs = [ s[0] * 100, s[1] * 100, s[2] * 500, 
		   		# 		s[3] * 20 , s[4] * 20 , s[5] * 10, 
				# 		s[6] * (np.pi / 4.0), s[7] * (np.pi / 4.0), s[8] * (np.pi / 4.0), 
				# 		s[9] * np.deg2rad(30), s[10] * np.deg2rad(30), s[11] * np.deg2rad(30),]
				print(  "xyz: ", s[0] * 100, s[1] * 100, s[2] * 500, 
		   			 	"Vxyz: ", s[3] * 20 , s[4] * 20 , s[5] * 10, 
						"theta xyz", np.rad2deg(s[8] * (np.pi / 4.0)), np.rad2deg(s[7] * (np.pi / 4.0)) , np.rad2deg(s[6] * (np.pi / 4.0)),
						"Vtheta xyz", np.rad2deg(s[9] * np.deg2rad(30)), np.rad2deg(s[10] * np.deg2rad(30)), np.rad2deg(s[11] * np.deg2rad(30)),
						"turn_scores ", turn_scores)
				break

			

		if record:
			# [新增] 2. 单个回合结束，提取并保存该回合的最终落点
			# 根据你原有的状态还原逻辑，实际坐标为 X: s_next[0]*50， Y: s_next[1]*300 + target_state[1]
			final_pos_x = s_next[0] * 100
			final_pos_y = s_next[1] * 500 + target_state[1]
			final_x_list.append(final_pos_x)
			final_y_list.append(final_pos_y)
			landing_state = info['landing_state']
			state_final = [s_next[0]*100, s_next[1]*500+target_state[1], s_next[2]*20, info['vy'], s_next[4]* np.deg2rad(90), s_next[5]* np.deg2rad(5)]

			if landing_state == 1:
				success[0] += 1
			elif landing_state == 2:
				success[1] += 1
			elif landing_state == 3:
				success[2] += 1
			elif landing_state == -1:
				success[3] += 1
				print('\nstate_fail:',state_final)                
			elif landing_state == -2:
				success[4] += 1
				print('\nstate_fail:',state_final)  

			print(success,'state_init:',state_init)

		if plot:
			history_x = np.array(history_x)
			history_u = np.array(history_u)
			t = np.arange(step-1) * dt

			# 加长画布，适应 3x2 的网格
			plt.figure(figsize=(12, 12))

			# 1. 轨迹图
			plt.subplot(3, 2, 1)
			plt.plot(history_x[:, 0], history_x[:, 1], 'b-', label='Trajectory')
			plt.plot(target_state[0], target_state[1], 'ro', markersize=10, label='Target')
			plt.plot(history_x[0, 0], history_x[0, 1], 'go', markersize=10, label='Start')
			plt.xlabel('X Position (m)'); plt.ylabel('Y Position (m)')
			plt.legend(); plt.grid(True)

			# 2. 姿态角
			plt.subplot(3, 2, 2)
			plt.plot(t, np.rad2deg(history_x[:, 4]), 'g-', label='Angle (deg)')
			plt.xlabel('Time (s)'); plt.ylabel('Angle (deg)')
			plt.legend(); plt.grid(True)

			# 3. 推力actual_thrust_kN
			plt.subplot(3, 2, 3)
			actual_thrust_kN = history_u[:, 0] / g
			plt.plot(t, actual_thrust_kN, 'r-', label='Thrust (g)')
			plt.axhline(1, color='k', linestyle='--', label='Hover Thrust')
			plt.axhline( MIN_THRUST_ACC * MAX_THRUST_ACC / g, color='gray', linestyle=':', label='Min Thrust')
			plt.xlabel('Time (s)'); plt.ylabel('Thrust (g)')
			plt.legend(); plt.grid(True)
			
			# 4. 摆角 (Gimbal)
			plt.subplot(3, 2, 4)
			plt.plot(t, np.rad2deg(history_u[:, 1]), 'purple', label='Gimbal Angle (deg)')
			plt.axhline(20.0, color='gray', linestyle=':')
			plt.axhline(-20.0, color='gray', linestyle=':')
			plt.xlabel('Time (s)'); plt.ylabel('Gimbal Angle (deg)')
			plt.legend(); plt.grid(True)

			# 5. 横向速度 Vx
			plt.subplot(3, 2, 5)
			plt.plot(t, history_x[:, 2], 'c-', label='Vx (m/s)')
			plt.axhline(0, color='gray', linestyle='--')
			plt.xlabel('Time (s)'); plt.ylabel('Vx (m/s)')
			plt.legend(); plt.grid(True)

			# 6. 垂直速度 Vy
			plt.subplot(3, 2, 6)
			plt.plot(t, history_x[:, 3], 'm-', label='Vy (m/s)')
			plt.axhline(0, color='gray', linestyle='--')
			plt.xlabel('Time (s)'); plt.ylabel('Vy (m/s)')
			plt.legend(); plt.grid(True)

			plt.tight_layout() # 自动调整子图间距，防止文字重叠
			plt.show()

	if render: env.close()
	return total_scores/turns
