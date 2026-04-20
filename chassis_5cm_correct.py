import mujoco
import mujoco.viewer
import time
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path('chassis.xml')
data = mujoco.MjData(model)

# --- 修正1：大幅降低基础速度 ---
# MuJoCo 物理单位是米，200太大，改为 0.5 - 2.0 之间的小数更合理
base_speed = 0.8 

try:
    # 确保 XML 中的 actuator 名称正确
    drive_motor_1 = model.actuator('drive_motor_1').id
    drive_motor_2 = model.actuator('drive_motor_2').id
    drive_motor_3 = model.actuator('drive_motor_3').id
    print(f"电机ID获取成功: 1={drive_motor_1}, 2={drive_motor_2}, 3={drive_motor_3}")
except Exception as e:
    print(f"获取电机ID失败，请检查XML actuator名称: {e}")
    exit(1)

# 参数配置
forward_duration = 2.0  # 缩短时间方便观察
turn_duration = 1.0     
wheel_radius = 0.1      # --- 修正2：添加轮子半径 ---
L = 0.2                 # 轮子中心到车体中心的距离

# 三个轮子角度 (分别对应 90°, -30°, 210°)
angles = [np.pi/2, -np.pi/6, 5*np.pi/6] 

# 运动学逆解矩阵 (针对3轮全向底盘)
# 格式: [v_x, v_y, w] -> [wheel1, wheel2, wheel3]
def compute_wheel_speeds(vx, vy, w):
    """
    计算三个轮子的速度。
    vx, vy: 线速度 (m/s)
    w: 角速度 (rad/s)
    """
    speeds = []
    for angle in angles:
        # 标准全向轮公式: omega = (vx*cos(theta) + vy*sin(theta) + L*w) / r
        # 这里省略除以 r，直接在 ctrl 中体现
        vel = vx * np.cos(angle) + vy * np.sin(angle) + L * w
        speeds.append(vel)
    return np.array(speeds)

def move_forward():
    # 前进: vx=base_speed, vy=0, w=0
    speeds = compute_wheel_speeds(base_speed, 0, 0)
    data.ctrl[drive_motor_1] = speeds[0]
    data.ctrl[drive_motor_2] = speeds[1]
    data.ctrl[drive_motor_3] = speeds[2]
    # print(f"前进: {speeds[0]:.3f}, {speeds[1]:.3f}, {speeds[2]:.3f}")

def turn_left():
    # 左转: vx=0, vy=0, w=正 (逆时针)
    speeds = compute_wheel_speeds(0, 0, base_speed/2) # 除以2防止转太快
    data.ctrl[drive_motor_1] = speeds[0]
    data.ctrl[drive_motor_2] = speeds[1]
    data.ctrl[drive_motor_3] = speeds[2]
    # print(f"左转: {speeds[0]:.3f}, {speeds[1]:.3f}, {speeds[2]:.3f}")

def stop():
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0

# 获取初始位置
mujoco.mj_forward(model, data)
start_x = data.qpos[0]
start_y = data.qpos[1]
print(f"起始位置: x={start_x:.4f}, y={start_y:.4f}")

# 启动仿真
with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.002 # 稍微调大步长
    step_count = 0
    
    try:
        for i in range(4):
            print(f"\n--- 第 {i+1} 边 ---")
            
            # 前进阶段
            print("前进中...")
            start_time = time.time()
            while time.time() - start_time < forward_duration:
                move_forward()
                mujoco.mj_step(model, data)
                viewer.sync()

            # 停止
            stop()
            viewer.sync()
            time.sleep(0.2) # 短暂停顿

            # 左转阶段
            print("左转中...")
            start_time = time.time()
            while time.time() - start_time < turn_duration:
                turn_left()
                mujoco.mj_step(model, data)
                viewer.sync()

            # 停止
            stop()
            viewer.sync()
            time.sleep(0.2)

        # 结束
        print(f"\n任务完成！最终位置: x={data.qpos[0]:.2f}, y={data.qpos[1]:.2f}")
        # 保持窗口打开
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    except KeyboardInterrupt:
        stop()
        print("程序已停止")