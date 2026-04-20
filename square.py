import mujoco
import mujoco.viewer
import time
import numpy as np

# ======================
# 加载你的小车模型
# ======================
model = mujoco.MjModel.from_xml_path("chassis.xml")  # 你的模型文件名
data = mujoco.MjData(model)

# ======================
# 对应你小车的3个电机
# ======================
MOTOR_NAMES = ["drive_motor_1", "drive_motor_2", "drive_motor_3"]
motor_ids = [model.actuator(name).id for name in MOTOR_NAMES]

# ======================
# 速度参数（可调节）
# ======================
FORWARD_SPEED  = 4.0    # 前进速度
ROTATE_SPEED   = 2.0    # 自转速度
FORWARD_TIME   = 1.2    # 前进时间（控制正方形大小）
TURN_TIME      = 0.75   # 转90度时间
PAUSE_TIME     = 0.2    # 动作间停顿

# ======================
# 三轮全向小车运动控制
# ======================
def forward():
    """三轮同速 → 直线前进"""
    data.ctrl[motor_ids[0]] = FORWARD_SPEED
    data.ctrl[motor_ids[1]] = FORWARD_SPEED
    data.ctrl[motor_ids[2]] = FORWARD_SPEED

def rotate_right():
    """三轮反向 → 原地右转90度"""
    data.ctrl[motor_ids[0]] = -ROTATE_SPEED
    data.ctrl[motor_ids[1]] = -ROTATE_SPEED
    data.ctrl[motor_ids[2]] = -ROTATE_SPEED

def stop():
    data.ctrl[motor_ids[0]] = 0
    data.ctrl[motor_ids[1]] = 0
    data.ctrl[motor_ids[2]] = 0

# ======================
# 主逻辑：走正方形
# ======================
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("✅ 开始走正方形")

    for _ in range(4):
        # 1. 前进
        forward()
        start = time.time()
        while time.time() - start < FORWARD_TIME:
            mujoco.mj_step(model, data)
            viewer.sync()

        # 2. 停止
        stop()
        time.sleep(PAUSE_TIME)

        # 3. 右转90度
        rotate_right()
        start = time.time()
        while time.time() - start < TURN_TIME:
            mujoco.mj_step(model, data)
            viewer.sync()

        # 4. 停止
        stop()
        time.sleep(PAUSE_TIME)

    stop()
    print("✅ 正方形走完！")
    viewer.close()