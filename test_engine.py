import mujoco
import mujoco.viewer
import numpy as np
import time

def main():
    # 1. 加载模型 (确保文件名正确)
    model = mujoco.MjModel.from_xml_path("rocket/rocket.xml")
    data = mujoco.MjData(model)

    # 诊断并计算悬停推力
    total_mass = mujoco.mj_getTotalmass(model)
    hover_thrust = total_mass * 9.81

    # 【新增】获取火焰几何体的 ID，以便我们后续高效访问其属性
    fire_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fire_geom")

    print("\n" + "="*40)
    print(f"🚀 火箭总质量: {total_mass:.2f} kg")
    print("开始仿真...前3秒将在地面测试万向节，3秒后起飞！")
    print("这一次，你将在底部看到动态的火焰特效！")
    print("="*40 + "\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        # viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track_cam")
        # viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        
        while viewer.is_running():
            while data.time < time.time() - start_time:
                t = data.time
                
                # 万向节摆动测试 (频率加快)
                data.ctrl[0] = 0.2 * np.sin(np.pi * t * 2)
                data.ctrl[1] = 0.2 * np.cos(np.pi * t * 2)

                # 推力测试时序
                if t < 3.0:
                    # 前3秒：推重比 0.5
                    current_thrust = hover_thrust * 0.5
                else:
                    # 3秒后：推重比 1.5
                    current_thrust = hover_thrust * 1.5
                
                data.ctrl[2] = current_thrust

                # ==========================================
                # 【新增】核心代码：动态火焰缩放
                # ==========================================
                # 1. 计算推力的归一化比例 (假设 XML 中 ctrlrange 是 0 到 850000)
                max_xml_thrust = model.actuator_ctrlrange[2][1]
                thrust_ratio = np.clip(current_thrust / max_xml_thrust, 0.0, 1.0)
                
                # 2. 动态调整几何体大小
                # size[0] 是半径, size[1] 是半轴长
                # 我们让半径在 [0.05, 0.4] 之间变化，长度在 [0.1, 3.0] 之间变化
                # 即使推力为 0，也会留一点点极小的火焰模拟“飞行员光”
                new_radius = 0.05 + 0.35 * thrust_ratio
                new_length = 0.1 + 2.9 * thrust_ratio
                model.geom_size[fire_geom_id, 0] = new_radius
                model.geom_size[fire_geom_id, 1] = new_length

                # 3. 动态调整位置，让火焰底部对准 thrust_point
                # new_length 是半轴长，所以几何体的中心应该在 site 下方 new_length 米处
                model.geom_pos[fire_geom_id, 2] = -new_length

                # 4. 可选：推力越小，火焰越透明
                # new_alpha = 0.1 + 0.6 * thrust_ratio
                # model.geom_rgba[fire_geom_id, 3] = new_alpha
                # ==========================================
                    
                mujoco.mj_step(model, data)
            
            viewer.sync()

if __name__ == "__main__":
    main()