import mujoco
import mujoco.viewer
import time
import numpy as np

# ======================
# 加载模型
# ======================
model = mujoco.MjModel.from_xml_path("chassis.xml")
data = mujoco.MjData(model)

# 获取电机ID
MOTOR_NAMES = ["drive_motor_1", "drive_motor_2", "drive_motor_3"]
motor_ids = [model.actuator(name).id for name in MOTOR_NAMES]

# ======================
# 物理参数配置
# ======================
WHEEL_RADIUS = 0.0174625  # 轮子半径 (从你的XML中获取的尺寸)
L = 0.1                   # 轮子中心到车体中心的距离 (根据你的模型布局估算)
# 三轮全向运动学逆解矩阵 (针对 120度 分布的轮子)
# 公式：[车轮速度] = M * [线速度x, 线速度y, 角速度z]
# 这里假设轮子1在左侧，角度为 150度；轮子2在右前，角度为 -90度 (30度)；轮子3在左前，角度为 -210度 (150度)
# 矩阵 M 的计算基于: wheel_i_vel = (cos(theta_i)*v_x + sin(theta_i)*v_y + L*omega) / R
M = np.array([
    [np.cos(np.pi/6),  np.sin(np.pi/6),  L],  # Wheel 2 (右前)
    [np.cos(5*np.pi/6), np.sin(5*np.pi/6), L],  # Wheel 3 (左前)
    [np.cos(3*np.pi/2), np.sin(3*np.pi/2), L]   # Wheel 1 (左后)
]) / WHEEL_RADIUS

# ======================
# 高层控制函数
# ======================
def set_velocity(v_x, v_y, omega):
    """
    设置小车的期望速度
    :param v_x: 前后方向速度 (m/s)
    :param v_y: 左右方向速度 (m/s)
    :param omega: 旋转角速度 (rad/s)
    """
    target_speeds = M @ np.array([v_x, v_y, omega])
    # 限制最大速度，防止电机过载
    max_speed = 10.0
    target_speeds = np.clip(target_speeds, -max_speed, max_speed)
    
    data.ctrl[motor_ids[0]] = target_speeds[0]
    data.ctrl[motor_ids[1]] = target_speeds[1]
    data.ctrl[motor_ids[2]] = target_speeds[2]

def stop():
    data.ctrl[motor_ids] = 0

# ======================
# 主逻辑
# ======================
def run_square():
    print("Starting Square Path...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 定义正方形参数
        side_length = 0.5   # 边长 0.5米
        turn_radius = 0.0   # 原地转弯
        
        for side in range(4):
            print(f"Running side {side + 1}")
            
            # 1. 直线前进
            start_time = data.time
            while data.time - start_time < side_length / 2.0: # 假设前进速度为 2m/s，距离0.5m需0.25秒
                set_velocity(v_x=2.0, v_y=0.0, omega=0.0)
                mujoco.mj_step(model, data)
                viewer.sync()
            
            # 2. 准备转弯 (稍微减速)
            stop()
            time.sleep(0.2)
            
            # 3. 原地右转 90度
            # 旋转 90度 (pi/2 弧度)，角速度设为 2 rad/s
            turn_angle = np.pi / 2
            turn_duration = turn_angle / 2.0
            start_time = data.time
            while data.time - start_time < turn_duration:
                # 右转时，线速度为0，角速度为正 (逆时针为正，实际看坐标系方向)
                # 如果发现左转，把 -2.0 改为 2.0
                set_velocity(v_x=0.0, v_y=0.0, omega=-2.0) 
                mujoco.mj_step(model, data)
                viewer.sync()
                
            # 4. 动作停顿
            stop()
            time.sleep(0.2)

        print("Square Complete!")
        # 保持窗口打开，直到手动关闭
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    run_square()