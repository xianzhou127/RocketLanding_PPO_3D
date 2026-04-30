import numpy as np
import time
from rocket import RocketEnv  # 确保你的环境文件名为 rocket.py
import mujoco
def main():
    print("初始化 Gymnasium 火箭环境...")
    # 开启人类渲染模式，这会自动触发我们刚写的 render() 方法【
    env = RocketEnv(xml_path="rocket/rocket.xml", render_mode="human", frame_skip=10)
    
    # 1. 环境重置，获取初始观测
    obs, info = env.reset()
    print(f"✅ 环境重置成功！观测空间维度: {obs.shape}")
    print(f"📊 初始状态 (高度Z = {obs[2]:.2f}m)")

    # 悬停推力估算：质量约 27500kg，重力 9.81，悬停需约 270kN
    # 我们的推力动作空间被 _scale_action 映射：[-1, 1] -> [0, 850kN]
    # 因此，动作值 a 满足: (a + 1)/2 * 850 = 270 => a 约等于 -0.36
    hover_action_val = -0.36

    # 2. 与环境交互循环
    test_steps = 3000
    for step in range(test_steps):
        # 构造一个符合 [-1.0, 1.0] 规范的随机/测试动作
        # action[0]: 俯仰 (Pitch)
        # action[1]: 偏航 (Yaw)
        # action[2]: 主推力 (Thrust)
        
        if step < 200:
            # 前 100 步：推力不足以起飞，只测试万向节摆动
            # pitch_cmd = 0.5 * np.sin(step * 0.1)
            # yaw_cmd = 0.5 * np.cos(step * 0.1)
            pitch_cmd = 0
            yaw_cmd = 0
            thrust_cmd = -1  # 映射后极小，产生火焰但不起飞
            action = np.array([pitch_cmd, yaw_cmd, thrust_cmd], dtype=np.float32)
        else:
            # 100 步后：推力加大 (略大于悬停)，伴随轻微摆动让火箭升空
            pitch_cmd = 0
            action = np.array([pitch_cmd, 0.0, -1], dtype=np.float32)

        # ==========================================
        # 最关键的一行：RL 智能体与环境的交互
        # ==========================================
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染画面
        env.render()
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rocket_main")
        # 打印状态以便观察
        if step % 20 == 0:
            pos = env.data.xpos[body_id]
            x, y, z = pos[0], pos[1], pos[2]
            vel = env.data.qvel[:3]
            vx, vy, vz = vel[0], vel[1], vel[2]
            print(f"Step: {step:3d} | 高度: {z:5.1f}m | 垂直速度: {vx:5.1f}m/s | 动作推力指令: {action[2]:.2f}")

        # 如果满足了你 _check_terminated 定义的坠毁或着陆条件
        if terminated or truncated:
            print(f"💥 回合在 step {step} 结束 (可能坠毁或触发边界)！重置环境...")
            obs, info = env.reset()
            time.sleep(1) # 停顿一下方便观察

        # 控制渲染速度，避免画面过快
        # frame_skip=10, 内部 timestep=0.002, 所以 RL 的 dt=0.02s
        # time.sleep(0.02)

    print("测试完成。")
    env.close()

if __name__ == "__main__":
    main()