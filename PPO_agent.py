from datetime import datetime
import os, shutil
import argparse
import torch
import gymnasium as gym
import numpy as np

from utilsPPO import str2bool, Action_adapter, Reward_adapter, evaluate_policy
from PPO import PPO_agent

from rocket import RocketEnv

torch.set_num_threads(2)  # 或者设置为 1
'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=1, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=True, help='render or None')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--runstime', type=str, default="2026-04-25 23_37", help='which data to save') #单独设置
parser.add_argument('--ModelIdex', type=int, default=328000, help='which model to load,0')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Distribution', type=str, default='Beta', help='Should be one of Beta ; GS_ms  ;  GS_m')
parser.add_argument('--Max_train_steps', type=int, default=int(5e9), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e5), help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.9995, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.15, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=512, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-5, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
parser.add_argument('--a_optim_batch_size', type=int, default=6144, help='lenth of sliced trajectory of actor')
parser.add_argument('--c_optim_batch_size', type=int, default=6144, help='lenth of sliced trajectory of critic')
parser.add_argument('--entropy_coef', type=float, default=3e-3, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=1.0, help='Decay rate of entropy_coef')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)

def main():
    # EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v4','HalfCheetah-v4','BipedalWalker-v3','BipedalWalkerHardcore-v3']
    # BrifEnvName = ['PV1', 'LLdV2', 'Humanv4', 'HCv4','BWv3', 'BWHv3']

    EnvName = ['hover','landing']
    BrifEnvName = ['hover', 'land']
    max_steps = 100/0.02 # 100s / 0.02 (dt)
    # PPO_dt = 0.02
    # env_dt = 0.002

    do_render = None
    if opt.render: do_render = 'human'

    # 定义环境工厂函数，给每个子进程独立创建环境
    def make_env():
        def _init():
            # 实例化你改好的 Rocket 环境
            env = RocketEnv(xml_path="rocket/rocket.xml", render_mode=do_render, frame_skip=10, max_steps=max_steps)
            return env
        return _init

    # 利用多线程优势，开启 10 个并行环境
    num_envs = 10   
    if not opt.render:
        envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(num_envs)])

    # 评估环境依然保留单线程即可
    eval_env = RocketEnv(xml_path="rocket/rocket.xml", render_mode=do_render, frame_skip=10, max_steps=max_steps)

    opt.state_dim = 17
    opt.action_dim = 3
    
    opt.max_action = 1.0 
    opt.min_action = -1.0
    opt.max_steps = max_steps   
    print('Env:',EnvName[opt.EnvIdex],'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,
          '  max_a:',opt.max_action,'  min_a:',opt.min_action, 'max_steps', opt.max_steps)
    
    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    timelater = ' ' + opt.runstime

    if opt.write:
        from torch.utils.tensorboard import SummaryWriter

        if opt.Loadmodel:
            writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timelater
            print(f"Read to Resume Training. Logging to: {writepath}")
        else:
            writepath = 'runs/{}'.format(BrifEnvName[opt.EnvIdex]) + timenow
            if os.path.exists(writepath): 
                shutil.rmtree(writepath)

        print(f"====> TensorBoard 日志绝对路径: {os.path.abspath(writepath)} <====")
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_agent(**vars(opt)) 
    ep_ret = np.zeros(num_envs)
    ep_len = np.zeros(num_envs)
    traj_lenth, total_steps = 0, 0
    next_eval_step = opt.eval_interval + opt.ModelIdex * 1000
    next_save_step = opt.save_interval + opt.ModelIdex * 1000
    
    if opt.Loadmodel: 
        agent.load(BrifEnvName[opt.EnvIdex], opt.ModelIdex, timelater)
        total_steps = opt.ModelIdex*1000

    if opt.render:
        ep_r = evaluate_policy(eval_env, agent, opt.max_action, 10, render=True,plot = False, record = False)
        print(f'Env:{EnvName[opt.EnvIdex]}, Episode Reward:{ep_r}')
    else:
        s, infos = envs.reset()
    
        while total_steps < opt.Max_train_steps:
            # 1. 动作推断
            a, logprob_a = agent.select_action(s, deterministic=False) 
            act = Action_adapter(a, opt.max_action) 
            
            # 2. 与环境交互
            s_next, r, terminated, truncated, info = envs.step(act)
            
            ep_ret += r  
            ep_len += 1       
            dones = np.logical_or(terminated, truncated)
            dw = np.logical_and(terminated, np.logical_not(truncated))

            # 【修改 2】严格遵循 Gymnasium 官方的向量化环境 final_observation 提取规范
            real_s_next = s_next.copy()
            if '_final_observation' in info:
                for i in range(num_envs):
                    # info['_final_observation'] 是一个布尔数组，表示哪些环境真的 done 了
                    if info['_final_observation'][i]:
                        real_s_next[i] = info['final_observation'][i]

            # 3. 存入经验回放池
            agent.put_data(
                s, a, r.reshape(-1, 1), real_s_next, 
                logprob_a, dones.reshape(-1, 1), dw.reshape(-1, 1), 
                idx=traj_lenth
            )

            # 记录标量数据
            for i in range(num_envs):
                if dones[i]:
                    if opt.write:
                        writer.add_scalar('Rollout/EpRet', ep_ret[i], global_step=total_steps)
                        writer.add_scalar('Rollout/EpLen', ep_len[i], global_step=total_steps)
                    ep_ret[i] = 0.0
                    ep_len[i] = 0

            # 步数推进
            traj_lenth += 1
            total_steps += num_envs
            
            # 4. 网络更新
            if traj_lenth >= opt.T_horizon:
                train_metrics = agent.train()
                traj_lenth = 0 
                
                if opt.write:
                    writer.add_scalar('Train/Actor_Loss', train_metrics['a_loss'], global_step=total_steps)
                    writer.add_scalar('Train/Critic_Loss', train_metrics['c_loss'], global_step=total_steps)
                    writer.add_scalar('Train/Entropy', train_metrics['entropy'], global_step=total_steps)
                    writer.add_scalar('Train/Approx_KL', train_metrics['kl'], global_step=total_steps)
                    writer.add_scalar('Train/Clip_Fraction', train_metrics['clip_frac'], global_step=total_steps)
                    writer.flush()

            s = s_next # 更新状态

            # 5. 评估与保存
            if total_steps >= next_eval_step:
                score = evaluate_policy(eval_env, agent, opt.max_action, turns=1, render=False) 
                
                if opt.write: 
                    writer.add_scalar('Eval_Reward_Avg', score, global_step=total_steps)
                    writer.flush()
                    
                print(f"EnvName: {EnvName[opt.EnvIdex]} | steps: {int(total_steps/1000)}k | score: {score:.2f}")
                next_eval_step += opt.eval_interval

            if total_steps >= next_save_step:
                save_time = timelater if opt.Loadmodel else timenow
                agent.save(BrifEnvName[opt.EnvIdex], int(total_steps/1000), save_time)
                next_save_step += opt.save_interval

        envs.close()

if __name__ == '__main__':
    main()
