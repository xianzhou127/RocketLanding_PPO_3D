from utilsPPO import BetaActor, GaussianActor_musigma, GaussianActor_mu, Critic
import numpy as np
import copy
import torch
import math


class PPO_agent(object):
	def __init__(self, **kwargs):
		# Init hyperparameters for PPO agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)

		# Choose distribution for the actor
		if self.Distribution == 'Beta':
			self.actor = BetaActor(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_ms':
			self.actor = GaussianActor_musigma(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		elif self.Distribution == 'GS_m':
			self.actor = GaussianActor_mu(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		else: print('Dist Error')
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

		# Build Critic
		self.critic = Critic(self.state_dim, self.net_width).to(self.dvc)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

		# 记得在 opt 中传入 num_envs=11，或者在这里直接写死 self.num_envs = 11
		self.num_envs = 10 

		# Build Trajectory holder (全部增加 self.num_envs 维度)
		self.s_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim), dtype=np.float32)
		self.a_hoder = np.zeros((self.T_horizon, self.num_envs, self.action_dim), dtype=np.float32)
		self.r_hoder = np.zeros((self.T_horizon, self.num_envs, 1), dtype=np.float32)
		self.s_next_hoder = np.zeros((self.T_horizon, self.num_envs, self.state_dim), dtype=np.float32)
		self.logprob_a_hoder = np.zeros((self.T_horizon, self.num_envs, self.action_dim), dtype=np.float32)
		self.done_hoder = np.zeros((self.T_horizon, self.num_envs, 1), dtype=np.bool_)
		self.dw_hoder = np.zeros((self.T_horizon, self.num_envs, 1), dtype=np.bool_)

	def select_action(self, state, deterministic):
		with torch.no_grad():
			# 判断传入的 state 是单条数据(1D)还是批量数据(2D)
			is_batched = len(np.shape(state)) > 1
			
			if not is_batched:
				# 评估模式：单条数据，增加 batch 维度变成 (1, state_dim)
				state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.dvc)
			else:
				# 训练模式：多进程传进来的已经是 (8, state_dim) 的矩阵，直接用
				state_tensor = torch.FloatTensor(state).to(self.dvc)

			if deterministic:
				# only used when evaluate the policy. Making the performance more stable
				a = self.actor.deterministic_act(state_tensor)
				
				if not is_batched:
					return a.cpu().numpy()[0], None  # 返回 shape (adim,)
				else:
					return a.cpu().numpy(), None     # 返回 shape (batch, adim)
			else:
				# only used when interact with the env
				dist = self.actor.get_dist(state_tensor)
				a = dist.sample()
				a = torch.clamp(a, 0, 1)
				logprob_a = dist.log_prob(a).cpu().numpy() # 保持原有的 batch 维度
				
				if not is_batched:
					# 单环境：提取第一行，并抹平对数概率
					return a.cpu().numpy()[0], logprob_a[0].flatten()
				else:
					# 多环境：直接返回整个矩阵 (batch, adim)
					# 这样在 main 函数的 for i in range(num_envs) 循环中，
					# 提取 a[i] 和 logprob_a[i] 就刚好对应单条数据
					return a.cpu().numpy(), logprob_a


	def train(self):
		self.entropy_coef = max(self.entropy_coef * self.entropy_coef_decay, 1e-4)

		'''1. Prepare PyTorch data from Numpy data (此时依然保持 3D: T_horizon, num_envs, dim)'''
		s = torch.from_numpy(self.s_hoder).to(self.dvc)
		a = torch.from_numpy(self.a_hoder).to(self.dvc)
		r = torch.from_numpy(self.r_hoder).to(self.dvc)
		s_next = torch.from_numpy(self.s_next_hoder).to(self.dvc)
		logprob_a = torch.from_numpy(self.logprob_a_hoder).to(self.dvc)
		done = torch.from_numpy(self.done_hoder).float().to(self.dvc)
		dw = torch.from_numpy(self.dw_hoder).float().to(self.dvc)

		'''2. 计算 Advantage 和 TD Target (必须在 3D 状态下进行，防止不同环境的时间线串扰)'''
		with torch.no_grad():
			# PyTorch 的 Linear 层支持直接传入 3D 张量，它会自动对最后一维操作
			vs = self.critic(s)        # 输出形状: (T_horizon, num_envs, 1)
			vs_ = self.critic(s_next)  # 输出形状: (T_horizon, num_envs, 1)

			# 计算 TD error
			deltas = r + self.gamma * (1.0 - dw) * vs_ - vs 
			
			# GAE 计算：在时间维度上逆序遍历
			adv = torch.zeros_like(deltas)
			adv_accum = 0
			# 注意：这里的 self.lamda 请根据你类里的实际变量名调整，有些叫 self.lam
			for t in reversed(range(self.T_horizon)):
				adv_accum = deltas[t] + self.gamma * self.lambd * (1.0 - done[t]) * adv_accum
				adv[t] = adv_accum
			
			# 此时 adv 和 vs 都是 (T_horizon, num_envs, 1)，相加不会有任何维度冲突！
			td_target = adv + vs 

		'''3. 展平所有张量到 2D，准备丢入网络进行 Mini-batch 更新'''
		# 用 .view(-1, dim) 直接将 (T_horizon, num_envs, dim) 拍扁成 (T_horizon * num_envs, dim)
		s = s.view(-1, self.state_dim)
		a = a.view(-1, self.action_dim)
		logprob_a = logprob_a.view(-1, self.action_dim)
		adv = adv.view(-1, 1)
		td_target = td_target.view(-1, 1)

		'''4. 优势函数标准化 (强推！在展平后的全局 Batch 上做标准化，训练会非常稳定)'''
		adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

		actor_losses = []
		critic_losses = []
		entropies = []
		kl_divs = []
		clip_fracs = []


		"""Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
		a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
		c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))

		target_kl = 0.02 # 设定的目标 KL 散度（经验值 0.01 ~ 0.03）
		
		for i in range(self.K_epochs):

			#Shuffle the trajectory, Good for training
			perm = np.arange(s.shape[0])
			np.random.shuffle(perm)
			perm = torch.LongTensor(perm).to(self.dvc)
			s, a, td_target, adv, logprob_a = \
				s[perm].clone(), a[perm].clone(), td_target[perm].clone(), adv[perm].clone(), logprob_a[perm].clone()

			'''update the actor'''
			for i in range(a_optim_iter_num):
				index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
				distribution = self.actor.get_dist(s[index])
				dist_entropy = distribution.entropy().sum(1, keepdim=True)
				logprob_a_now = distribution.log_prob(a[index])
				ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))

				# --- 新增：提取并计算 KL 散度和 Clip 比例 ---
				with torch.no_grad():
					approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
					clip_frac = (torch.abs(ratio - 1.0) > self.clip_rate).float().mean().item()

				surr1 = ratio * adv[index]
				surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv[index]
				a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

				self.actor_optimizer.zero_grad()
				a_loss.mean().backward()
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
				self.actor_optimizer.step()

			# --- 新增：收集当前 batch 的数据 ---
				actor_losses.append(a_loss.mean().item())
				entropies.append(dist_entropy.mean().item())
				kl_divs.append(approx_kl)
				clip_fracs.append(clip_frac)

			'''update the critic'''
			for i in range(c_optim_iter_num):
				index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
				c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
				for name,param in self.critic.named_parameters():
					if 'weight' in name:
						c_loss += param.pow(2).sum() * self.l2_reg

				self.critic_optimizer.zero_grad()
				c_loss.backward()
				self.critic_optimizer.step()

			# --- 新增：收集 current batch 的 Critic Loss ---
			critic_losses.append(c_loss.item())

			# --- 新增：Early Stopping 逻辑 ---
            # 计算这一个 epoch 中所有 mini-batch 的平均 KL
            # 注意：这里的 kl_divs 是你在 actor 更新循环里收集的列表
			current_epoch_kl = np.mean(kl_divs[-a_optim_iter_num:]) 
			if current_epoch_kl > 1.5 * target_kl:
				print(f"Early stopping at epoch {i} due to reaching max kl: {current_epoch_kl:.4f}")
				break # 强行终止当前轨迹的后续 epoch 更新，保护网络

		return {
			"a_loss": np.mean(actor_losses),
			"c_loss": np.mean(critic_losses),
			"entropy": np.mean(entropies),
			"kl": np.mean(kl_divs),
			"clip_frac": np.mean(clip_fracs)
		}

	def put_data(self, s, a, r, s_next, logprob_a, done, dw, idx):
		self.s_hoder[idx] = s
		self.a_hoder[idx] = a
		self.r_hoder[idx] = r
		self.s_next_hoder[idx] = s_next
		self.logprob_a_hoder[idx] = logprob_a
		self.done_hoder[idx] = done
		self.dw_hoder[idx] = dw

	def save(self,EnvName, timestep, time):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}{}.pth".format(EnvName,timestep,time))
		torch.save(self.critic.state_dict(), "./model/{}_q_critic{}{}.pth".format(EnvName,timestep,time))

	def load(self,EnvName, timestep, time):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}{}.pth".format(EnvName, timestep,time)))
		self.critic.load_state_dict(torch.load("./model/{}_q_critic{}{}.pth".format(EnvName, timestep,time)))



