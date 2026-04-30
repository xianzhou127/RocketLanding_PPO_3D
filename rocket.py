import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R # 确保在文件开头导入

class RocketEnv(gym.Env):
    """
    基于MuJoCo的火箭垂直着陆PPO环境
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, xml_path="./rocket/rocket.xml", frame_skip=10, render_mode=None, max_steps = 5000):
        super().__init__()

        # 保存渲染模式
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # 1. 加载MuJoCo模型和数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 控制频率设置：MuJoCo底层的dt通常很小（如0.002s），RL控制频率不需要那么高
        # frame_skip决定了执行一次RL动作，MuJoCo在底层步进多少次
        self.frame_skip = frame_skip 
        self.dt = self.model.opt.timestep * self.frame_skip
        # print(self.dt)

        # 动态计算 0.8g 和 2.0g 的推力边界
        total_mass = mujoco.mj_getTotalmass(self.model)
        gravity = 9.81
        hover_thrust = total_mass * gravity
        
        self.min_thrust = 0.8 * hover_thrust
        self.max_thrust = 2.0 * hover_thrust

        self.max_gimbal_speed = np.deg2rad(50.0) # 最大转速 50度/秒
        self.max_delta = self.max_gimbal_speed * self.dt

        # 1. 舵机转速限制 (假设 50度/秒)
        self.max_gimbal_speed = np.deg2rad(50.0) 
        self.max_delta_angle = self.max_gimbal_speed * self.dt 

        # 2. 涡轮泵响应限制
        # 假设发动机从最小推力到最大推力需要 0.5 秒
        engine_response_time = 0.5 
        max_thrust_change_per_sec = (self.max_thrust - self.min_thrust) / engine_response_time
        # 每一步允许的最大推力变化量
        self.max_delta_thrust = max_thrust_change_per_sec * self.dt
        
        # 2. 定义动作空间 (Action Space)
        # 假设有3个控制量：主发动机推力(1维), 偏航/俯仰万向节角度(2维)
        # 神经网络输出通常归一化在 [-1, 1] 之间
        self.action_dim = 3 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        
        # 3. 定义状态/观测空间 (Observation Space)
        # 通常包括：位置(x,y,z), 姿态(四元数或欧拉角), 线速度(vx,vy,vz), 角速度(wx,wy,wz)
        # 这里假设总共有 13 个维度的状态变量
        self.observation_dim = 17
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        # 1. 将当前的 last_action 归档为 prev_action，专门留给 Reward 计算差值用
        self.prev_action = self.last_action.copy()
        
        # 2. 更新当前动作，让 _get_obs 能把它打包进 s_{t+1}，供下一帧决策使用
        self.last_action = action.copy()

        # 1. 动作逆映射：将 [-1, 1] 的动作映射到物理限制范围内（例如推力 0~1000kN）
        real_action = self._scale_action(action)
        self.data.ctrl[:] = real_action
        
        # 2. 推进仿真
        for _ in range(self.frame_skip):
            # 注入气动代码  
            self._apply_aerodynamics()
            mujoco.mj_step(self.model, self.data)

            THRESHOLD = 1e6
            if np.max(np.abs(self.data.qacc)) > THRESHOLD or np.isnan(self.data.qacc).any():
                print(f"\n[致命错误] 仿真在时间 t={self.data.time:.4f} 崩溃！")

            # 5. 检查终止条件（成功着陆或坠毁）
            terminated = self._check_terminated()
            truncated = False
            if self.current_step >= self.max_steps: truncated = True # 超时截断
            if terminated or truncated: break
        
        # 3. 获取新状态
        obs = self._get_obs()
        # print(obs)

        # 4. 计算奖励
        reward = self._compute_reward(obs)

        info = {} # 用于记录额外调试信息，如触地速度、剩余燃料等
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 【新增】状态维护变量重置
        self.current_step = 0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.current_gimbal_pitch = 0.0
        self.current_gimbal_yaw = 0.0
        self.current_thrust = self.min_thrust
        # --- 初始化标志位 ---
        self.already_crash = False
        self.already_landing = False
        self.landing_state = 0
        
        # 重置MuJoCo内部状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机化初始状态（高度、初始速度、初始姿态偏差等）
        # 这对于PPO的鲁棒性至关重要
        # self.data.qpos[2] = np.random.uniform(100.0, 150.0) # 假设 z 轴是高度
        # self.data.qvel[:] = np.random.uniform(-0.1, 0.1, size=self.data.qvel.shape)
        # self.data.qpos[2] = 500 # 假设 z 轴是高度
        # self.data.qvel[:] = 0

        # self.data.xpos[:] = np.array([0,0,500])
        self.data.qpos[0] = np.random.uniform(-120.0, 120.0) #x
        self.data.qpos[1] = np.random.uniform(-120.0, 120.0) #y
        self.data.qpos[2] = np.random.uniform(400.0, 600.0) #z
        # self.data.qpos[2] = np.random.uniform(2000.0, 2500.0) #z
        # self.data.qpos[2] = 15 #z
        
        self.data.qvel[0] = np.random.uniform(-12.0, 12.0)
        # self.data.qvel[0] = 10
        self.data.qvel[1] = np.random.uniform(-12.0, 12.0)
        self.data.qvel[2] = np.random.uniform(-60.0, -20.0)
        
        self.data.qvel[3]  = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        self.data.qvel[4]  = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        self.data.qvel[5]  = np.random.uniform(-np.deg2rad(5), np.deg2rad(5))
        # 随机生成 Roll, Pitch, Yaw 欧拉角
        # 注意：通常火箭着陆时不需要巨大的初始 Roll (滚转)，重点随机化 Pitch 和 Yaw
        init_roll = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))  # 偏航角可以大一点
        init_pitch = np.random.uniform(-np.deg2rad(40), np.deg2rad(40)) 
        init_yaw = np.random.uniform(-np.deg2rad(40), np.deg2rad(40))

        # init_roll = np.random.uniform(-np.deg2rad(50), np.deg2rad(50)) 
        # init_pitch = np.random.uniform(-np.deg2rad(10), np.deg2rad(10)) 
        # init_yaw = np.random.uniform(-np.deg2rad(50), np.deg2rad(50)) # 偏航角可以大一点

        # init_roll = 0
        # init_pitch = 0 
        # init_yaw = 0 # 偏航角可以大一点
        
        # 使用 scipy 将欧拉角转换为四元数
        r = R.from_euler('xyz', [init_yaw, init_pitch, init_roll], degrees=False)
        quat_scipy = r.as_quat() # 格式是 [x, y, z, w]

        # ⚠️ 致命易错点：MuJoCo 的四元数格式是 [w, x, y, z]！必须手动换位！
        quat_mujoco = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
        
        # 将四元数写入 qpos 的对应位置 (freejoint 的位置占 3 格，四元数占 4 格)
        self.data.qpos[3:7] = quat_mujoco
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        # print(obs[6:9])
        info = {}
        
        return obs, info

    def _get_obs(self):
        """
        融合了运动学制导与 LQR 势能评估的 3D 高级观测状态提取
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket_main")
        
        # --- 1. 获取绝对物理量 ---
        pos = self.data.xpos[body_id].copy()
        x, y, z = pos[0], pos[1], pos[2]
        
        vel = self.data.qvel[:3].copy()
        vx, vy, vz = vel[0], vel[1], vel[2] # 注意：Z是垂直高度轴
        
        ang_vel = self.data.qvel[3:6].copy()
        wx, wy, wz = ang_vel[0], ang_vel[1], ang_vel[2]

        # 提取欧拉角 (Roll, Pitch, Yaw)
        quat_mujoco = self.data.xquat[body_id].copy()
        quat_scipy = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]] # [x, y, z, w]
        roll, pitch, yaw = R.from_quat(quat_scipy).as_euler('zyx', degrees=False)

        # --- 2. 🚀 物理级运动学速度引导 (Kinematic Glide Slope) ---
        target_z =2.0 # 假设着陆时质心高度为 12m
        TOUCHDOWN_VELOCITY = 0.0    
        DESIRED_DECELERATION = 4.0  
        MAX_FALL_SPEED = 80.0       
        
        altitude = max(0.0, z - (target_z + 1.0))
        
        # 计算 Z 轴理想下落速度
        ideal_vz = -np.sqrt(TOUCHDOWN_VELOCITY**2 + 2.0 * DESIRED_DECELERATION * altitude)
        target_vz = max(-MAX_FALL_SPEED, ideal_vz) 
        
        MAX_ACCEPTABLE_VZ_ERROR = 10.0  
        vz_error = (vz - target_vz) / MAX_ACCEPTABLE_VZ_ERROR

        # --- 3. 矢量特征归一化 ---
        dx = x / 50.0   
        dy = y / 50.0
        dz = (z - target_z) / 500.0
        # dz = (z - target_z) / 2000.0
        
        vx_norm = vx / 30.0
        vy_norm = vy / 30.0
        vz_norm = vz / 10.0
        
        roll_norm = roll / (np.pi / 4.0)
        pitch_norm = pitch / (np.pi / 4.0)
        yaw_norm = yaw / (np.pi / 4.0)
        
        wx_norm = wx / np.deg2rad(30)
        wy_norm = wy / np.deg2rad(30)
        wz_norm = wz / np.deg2rad(30)

        # --- 4. 标量特征 (Critic 的单调势能评估) ---
        dist_norm = np.sqrt(dx**2 + dy**2 + dz**2) / np.sqrt(3)
        distxy_norm = np.sqrt(dx**2 + dy**2) / np.sqrt(2)
        v_abs_norm = np.sqrt(vx_norm**2 + vy_norm**2 + vz_norm**2) / np.sqrt(3)
        vxy_abs_norm = np.sqrt(vx_norm**2 + vy_norm**2) / np.sqrt(2)
        angle_abs_norm = np.sqrt(pitch_norm**2 + yaw_norm**2) / np.sqrt(2) # 重点惩罚俯仰和偏航

        # LQR 二次型代价特征
        X_T_Q_X_norm = (
            5.0 * dx**2 + 5.0 * dy**2 + 1.0 * dz**2 +
            1.0 * vx_norm**2 + 1.0 * vy_norm**2 + 2.0 * vz_error**2 + 
            5.0 * pitch_norm**2 + 5.0 * yaw_norm**2 + 
            1.0 * wx_norm**2 + 1.0 * wy_norm**2 + 1.0 * wz_norm**2
        ) / 10.0

        # --- 5. 物理约束特征 ---
        # 获取当前步数和最后一次动作 (需要在 step 和 reset 中维护)
        t_norm = getattr(self, 'current_step', 0) / self.max_steps
        
        last_u = getattr(self, 'last_action', np.zeros(3))
        last_u_pitch = last_u[0]
        last_u_yaw   = last_u[1]
        last_u_thrust= last_u[2]

        # --- 6. 拼接输出 ---
        obs = np.array([
            # 矢量区 (12个): 位置3, 速度3, 角度3, 角速度3
            dx, dy, dz, 
            vx_norm, vy_norm, vz_norm, 
            roll_norm, pitch_norm, yaw_norm, 
            wx_norm, wy_norm, wz_norm,
            # 标量区 (1个)
            X_T_Q_X_norm,
            # 约束区 (4个)
            t_norm, last_u_pitch, last_u_yaw, last_u_thrust
        ], dtype=np.float32)

        # obs = np.array([
        #     # 矢量区 (12个): 位置3, 速度3, 角度3, 角速度3
        #     dx, dy, dz, 
        #     vx_norm, vy_norm, vz_norm, 
        #     roll_norm, pitch_norm, yaw_norm, 
        #     wx_norm, wy_norm, wz_norm,
        #     # 标量区 (4个)
        #     distxy_norm, vxy_abs_norm, angle_abs_norm, X_T_Q_X_norm,
        #     # 约束区 (4个)
        #     t_norm, last_u_pitch, last_u_yaw, last_u_thrust
        # ], dtype=np.float32)
        

        # print(obs)

        return obs

    def _scale_action(self, action):
        """将 PPO 输出的 [-1, 1] 映射到 XML 中定义的致动器范围，并加入执行机构动态延迟"""
        scaled_action = np.zeros(3)
        
        # 最大万向节偏转角 (你原本代码里的 0.26 弧度，约 15度)
        MAX_GIMBAL_ANGLE = 0.26 
        
        # --- 1. 万向节俯仰 (Pitch) 动态控制 ---
        pitch_target = action[0] * MAX_GIMBAL_ANGLE
        delta_pitch = pitch_target - self.current_gimbal_pitch
        actual_delta_pitch = np.clip(delta_pitch, -self.max_delta, self.max_delta)
        self.current_gimbal_pitch += actual_delta_pitch
        scaled_action[0] = self.current_gimbal_pitch
        
        # --- 2. 万向节偏航 (Yaw) 动态控制 ---
        yaw_target = action[1] * MAX_GIMBAL_ANGLE
        delta_yaw = yaw_target - self.current_gimbal_yaw
        actual_delta_yaw = np.clip(delta_yaw, -self.max_delta, self.max_delta)
        self.current_gimbal_yaw += actual_delta_yaw
        scaled_action[1] = self.current_gimbal_yaw
        
        # # --- 3. 主推力 (Thrust) ---
        # # 真实液体火箭发动机也有节流阀响应延迟 (Throttle Lag)，
        # # 如果目前不需要那么复杂的引擎建模，推力保持原样的直接映射即可。
        # scaled_action[2] = self.min_thrust + (action[2] + 1.0) / 2.0 * (self.max_thrust - self.min_thrust)

        # --- 3. 🚀 主推力 (Thrust) 涡轮泵延迟控制 ---
        # 先计算出网络“期望”的目标推力
        target_thrust = self.min_thrust + (action[2] + 1.0) / 2.0 * (self.max_thrust - self.min_thrust)
        
        # 计算推力差值
        delta_thrust = target_thrust - self.current_thrust
        
        # 限制这一步能改变的最大推力幅度
        # (注意：有些发动机减速比加速快，这里为了简便先设为对称限制)
        actual_delta_thrust = np.clip(delta_thrust, -self.max_delta_thrust, self.max_delta_thrust)
        
        # 更新当前真实推力
        self.current_thrust += actual_delta_thrust
        scaled_action[2] = self.current_thrust
        
        return scaled_action

    def _compute_reward(self, obs): 
        # --- 1. 直接从 obs 中提取预处理好的势能 ---
        # obs[15] 就是你刚才算好的 X_T_Q_X_norm
        # 注意：你在 obs 里除以了 18.0，为了和之前的惩罚力度保持一致，这里可以乘回来 (可选)
        X_T_Q_X = obs[12] * 10

        vel = self.data.qvel[:3].copy()
        vx, vy, vz = vel[0], vel[1], vel[2] # 注意：Z是垂直高度轴
        
        # 提取最后一次动作 (根据你的 _get_obs，last_u 顺序是 pitch, yaw, thrust)
        last_action = self.prev_action
        action = self.last_action 

        # ==========================================
        # 2. 核心势能与生存逻辑 (完全复用)
        # ==========================================
        # 势能函数 (取负的 sqrt)
        potential_penalty = - np.sqrt(X_T_Q_X)
        
        # 生存奖励 (将曲面末端抬高)
        C_survival = 0.5 * np.exp(-0.5 * abs(potential_penalty))
        
        # 💥 防自杀补丁 1
        step_penalty_weight = 0.5 
        r_state = (step_penalty_weight * potential_penalty) + C_survival
        
        # ==========================================
        # 3. 动作阻尼
        # ==========================================
        # 假设 action 也是 [pitch_cmd, yaw_cmd, thrust_cmd]
        u_delta = 0.5 * np.sum((action - last_action)**2) 
        r_action = - u_delta
        
        # 单步过程奖励
        r_process = r_state + r_action

        # ==========================================
        # 4. 终端奖励判定
        # ==========================================
        terminal_reward = 0
        if self.current_step >= self.max_steps:
            terminal_reward = -1000.0 + 500 * np.exp(-0.1 * abs(potential_penalty))

        elif self.already_crash:
                # 💥 防自杀补丁 2
            terminal_reward = -500.0 + 500.0 * np.exp(-0.1 * abs(potential_penalty))
            if self.landing_state == -3:
                terminal_reward += -2000
         
        elif self.already_landing:
            base_success = 1500.0
            time_bonus = (self.max_steps - self.current_step) * C_survival
            precision_bonus = 2000.0 * np.exp(-0.5 * abs(potential_penalty))
            vz_bonus = 1000 * np.exp(-0.5 * abs(vz))
            
            if self.landing_state == 1:
                base_success += 2000
            elif self.landing_state == 2:
                base_success += 1000
            elif self.landing_state == 3:
                base_success += 500
            
            terminal_reward = base_success + time_bonus + precision_bonus + vz_bonus

        reward = r_process + terminal_reward
        
        return reward

    def _check_terminated(self):
        """定义什么是回合结束 (3D 宽容触地判定)"""
            
        # --- 1. 获取绝对物理量 (从底层数据获取，避免受 obs 归一化影响) ---
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket_main")
        
        pos = self.data.xpos[body_id]
        x, y, z = pos[0], pos[1], pos[2]
        
        vel = self.data.qvel[:3]
        vx, vy, vz = vel[0], vel[1], vel[2]
        
        ang_vel = self.data.qvel[3:6]
        wx, wy, wz = ang_vel[0], ang_vel[1], ang_vel[2]

        quat_mujoco = self.data.xquat[body_id]
        quat_scipy = [quat_mujoco[1], quat_mujoco[2], quat_mujoco[3], quat_mujoco[0]]
        roll, pitch, yaw = R.from_quat(quat_scipy).as_euler('zyx', degrees=False)

        # --- 3. 触地判定逻辑 ---
        # 假设目标触地时质心高度为 target_z (你在 obs 里设为了 12.0)
        target_z = 3.0 
        
        # 只要高度降到目标高度以下，就视为“砸”到地面了
        if z <= target_z:
            crash = False
            
            # A. 速度限制 (垂直方向放宽到 15.0，水平面 X 和 Y 漂移放宽到 10.0)
            if abs(vz) >= 15.0: 
                crash = True
            if abs(vx) >= 10.0 or abs(vy) >= 10.0:
                crash = True
                
            # B. 落点限制 (水平面距离目标点的半径，假设为 20.0 米)
            target_r = getattr(self, 'target_r', 20.0) 
            distance_from_target = np.sqrt(x**2 + y**2)
            # if distance_from_target >= target_r:
            #     crash = True
                
            # C. 角度限制 (放宽到 10 度)
            angle_limit = 10.0 / 180.0 * np.pi
            # 💡 提示：对于火箭，滚转(roll)通常不致命，核心惩罚俯仰(pitch)和偏航(yaw)
            if abs(pitch) >= angle_limit or abs(yaw) >= angle_limit:
                crash = True
                
            # D. 角速度限制 (放宽到 10度/秒)
            ang_vel_limit = 10.0 / 180.0 * np.pi
            if abs(wx) >= ang_vel_limit or abs(wy) >= ang_vel_limit or abs(wz) >= ang_vel_limit:
                crash = True

            # 更新环境状态，供 Reward 函数结算使用
            if crash:
                self.already_crash = True
            else:
                self.already_landing = True
                self.landing_state = 4
                # 你可以根据精度分级，比如距离小于 5m 给顶级奖励，小于 20m 给次级奖励
                if abs(vz) <= 3.0 and abs(vx) <= 2.0 and abs(vy) <= 2.0 and \
                abs(pitch) <= np.deg2rad(3) and abs(yaw) <= np.deg2rad(3) and \
                abs(wx) <= np.deg2rad(5) and abs(wy) <= np.deg2rad(5) and abs(wz) <= np.deg2rad(5):
                    if distance_from_target <= 10.0:
                        self.landing_state = 1 # 完美着陆
                    elif distance_from_target <= 50.0:
                        self.landing_state = 2 # 基本着陆
                    else:
                        self.landing_state = 3 # 基本幸存
                return True
            
            self.landing_state = -1
            # print(self.landing_state)

            return True # 只要碰到地板，回合必然结束
            
        # --- 4. 空中出界保护 (防飞丢补丁) ---
        # 如果火箭往天上乱飞或者水平飞出几千米，直接掐断，节约训练算力
        phi = np.sqrt(x**2 + y**2) / np.sqrt(x**2 + y**2 + z**2)

        # if roll > 0 and roll < np.pi/2 and (a < 0 or b < 0) or \
        # roll > np.pi/2 and roll < np.pi and (a < 0 or b > 0) or \
        # roll > np.pi and roll < np.pi * 3/2 and (a > 0 or b > 0) or \
        # roll > np.pi * 3/2 and roll < np.pi * 2 and (a > 0 or b < 0):
            # self.already_crash = True
            # self.landing_state = -3
            # return True
            
        if z > 5000.0 or abs(x) > 5000.0 or abs(y) > 5000.0 or \
        abs(wx) >= np.deg2rad(900) or abs(wy) >= np.deg2rad(900) or abs(wz) >= np.deg2rad(900):
            self.already_crash = True
            self.landing_state = -3
            return True
        if self.current_step >= self.max_steps:
            self.landing_state = -2
            # print(self.landing_state)

        return False

    def _apply_aerodynamics(self):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket_main")

        # 先清零，避免残留
        self.data.qfrc_applied[:] = 0.0

        # 读取整机总质心：world body 的 subtree_com 就是整个模型的总质心
        p_com_total = self.data.subtree_com[0].copy()

        alt = max(float(p_com_total[2]), 0.0)
        if alt < 0: alt = 0.0
        
        rho = 1.225 * np.exp(-alt / 7400.0)

        # 世界坐标系下的线速度
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel6,
            0  # world orientation
        )
        v_world = vel6[3:].copy()   # 官方文档：顺序是 (rot:lin)
        w_body = vel6[:3].copy()   # 如果你当前绑定这里是角速度分量
        # print(w_body)
        # print(v_world)
        # v_world = self.data.qvel[:3] 
        v_mag = np.linalg.norm(v_world)
        
        if v_mag < 0.01:
            self.data.xfrc_applied[body_id] = np.zeros(6)
            return

        # 旋转矩阵 (Body to World)
        R = self.data.xmat[body_id].reshape(3, 3)
        
        # 速度转换到机体坐标系 [v_bx, v_by, v_bz]
        # 假设 Z 轴 (v_body[2]) 是火箭轴向
        v_body = R.T @ v_world

        # ==========================================
        # 【新增】1. 计算真正的 3D 总攻角 (Total Angle of Attack)
        # 范围在 [0, pi] 之间。0 表示正向飞行，pi 表示尾部朝下（降落状态）
        # ==========================================
        # 防止浮点数精度问题导致 arccos 报错
        cos_alpha = np.clip(v_body[2] / v_mag, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        # 2. 几何参数
        D = 3.6
        H = 46.0
        
        # ==========================================
        # 【修改】3. 动态迎风面积投影 (匹配你的 2D 逻辑)
        # ==========================================
        # 当 alpha=0 或 pi 时，sin(alpha)=0 (侧面不受风)
        # 当 alpha=pi/2 时，sin(alpha)=1 (侧面积完全迎风)
        A_side_eff = H * D * abs(np.sin(alpha))
        # print(A_side_eff)
        
        # 轴向同理
        A_axial_eff = np.pi * (D / 2.0)**2 * abs(np.cos(alpha))

        Cd_axial = 0.4
        Cd_side = 1.2
        # 定义压心 (Cp) 在机体坐标系下相对于质心 (CoG) 的位置。
        # 假设栅格舵展开后，整体压心在质心上方 15 米处
        dist_cp_cg = 8
        r_cp = np.array([0.0, 0.0, dist_cp_cg])

        # # 【修改】失去栅格舵后，阻力系数大幅下降
        # Cd_axial = 0.4  # 底部发动机端面平坦，依然有阻力，但没有顶部栅格舵大
        # Cd_side = 0.5   # 光滑圆柱体的侧向阻力系数远低于带栅格舵的形态 (原来是 1.2)
        # # 【修改】压心 (Cp) 大幅下移，靠近几何中心
        # # 假设压心现在只在质心上方 6 米处 (原来是 15 米)
        # dist_cp_cg = 6.0 
        # r_cp = np.array([0.0, 0.0, dist_cp_cg])

        # 4. 计算阻力 (使用总动压计算力的大小，再分配到各轴)
        # 动压 q = 0.5 * rho * v^2
        q = 0.5 * rho * v_mag**2
        
        # 计算阻力标量
        F_drag_axial_mag = q * A_axial_eff * Cd_axial
        F_drag_side_mag = q * A_side_eff * Cd_side

        # 将阻力方向与速度方向相反
        F_body = np.zeros(3)
        # X 和 Y 轴构成了法向速度平面
        v_side_mag = np.sqrt(v_body[0]**2 + v_body[1]**2)
        if v_side_mag > 1e-4:
            F_body[0] = -F_drag_side_mag * (v_body[0] / v_side_mag)
            F_body[1] = -F_drag_side_mag * (v_body[1] / v_side_mag)
            
        # Z 轴是轴向
        F_body[2] = -F_drag_axial_mag * np.sign(v_body[2])

        # 5. 气动转矩
        # 简单气动阻尼
        C_damp_pitch = 3.0e5
        C_damp_yaw   = 3.0e5

        tau_damp_body = np.array([
            -C_damp_pitch * w_body[0],
            -C_damp_yaw   * w_body[1],
            0.0
        ])

        F_moment_body = np.array([F_body[0], F_body[1], 0.0])
        tau_body = np.cross(r_cp, F_moment_body) + tau_damp_body

        # 6. 转换回世界坐标系并施加
        F_world = R @ F_body
        tau_world = R @ tau_body
        # print("F_world =", F_world)
        # print("tau_world =", tau_world)
        # print("tau_damp_body =", tau_damp_body)
        mujoco.mj_applyFT(
            self.model,
            self.data,
            F_world,
            tau_world,
            p_com_total,      # 世界坐标里的施力点
            body_id,          # 该点附着在哪个 body 上；这里用 rocket_main 即可
            self.data.qfrc_applied
        )

        # self.data.xfrc_applied[body_id][:3] = F_world
        # self.data.xfrc_applied[body_id][3:] = tau_world

    def render(self):
        """实现标准 Gym 的渲染接口"""
        
        # === 模式 1：返回像素矩阵（用于拼接视频） ===
        if self.render_mode == "rgb_array":
            # 1. 初始化离屏渲染器 (如果还没有的话)
            if not hasattr(self, 'renderer') or self.renderer is None:
                # 设置单张画面的分辨率
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            # 2. 渲染第一个视角：全景
            self.renderer.update_scene(self.data, camera="tracking_view")
            img_main = self.renderer.render()

            # 3. 渲染第二个视角：发动机特写
            self.renderer.update_scene(self.data, camera="engine_view")
            img_engine = self.renderer.render()

            # 4. 拼接画面 (使用 numpy)
            # hstack: 左右拼接 (宽度变成 1280)
            # vstack: 上下拼接 (高度变成 960)
            combined_img = np.hstack((img_main, img_engine)) 
            
            # 你也可以做画中画 (Picture-in-Picture)：
            # import cv2
            # img_engine_small = cv2.resize(img_engine, (160, 120))
            # img_main[10:130, 10:170] = img_engine_small
            # combined_img = img_main

            return combined_img

        # === 模式 2：人类观看交互模式 ===
        elif self.render_mode == "human":
            if not hasattr(self, 'viewer') or self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        """关闭环境并清理资源"""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
            self.viewer = None