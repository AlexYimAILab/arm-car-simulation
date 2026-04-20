import mujoco
import mujoco.viewer
import time
import numpy as np

# 直接加载arm.xml（积木块已在XML中）
model = mujoco.MjModel.from_xml_path('arm.xml')
data = mujoco.MjData(model)

# 获取执行器ID
rotation_joint = model.actuator('Rotation').id
pitch_joint = model.actuator('Pitch').id
elbow_joint = model.actuator('Elbow').id
wrist_pitch_joint = model.actuator('Wrist_Pitch').id
wrist_roll_joint = model.actuator('Wrist_Roll').id
jaw_joint = model.actuator('Jaw').id

print(f"执行器ID: Rotation={rotation_joint}, Pitch={pitch_joint}, Elbow={elbow_joint}, Wrist_Pitch={wrist_pitch_joint}, Wrist_Roll={wrist_roll_joint}, Jaw={jaw_joint}")

# 打印关节限位
print("\n关节限位:")
for i in range(model.nu):
    act = model.actuator(i)
    print(f"  {act.name}: ctrlrange = {act.ctrlrange}")

# 找到爪子body的id
jaw_body_id = model.body('Moving_Jaw_08d-v1').id
print(f"\n爪子 body id: {jaw_body_id}")

# 找到积木块body的id并打印位置
try:
    block_body_id = model.body('block').id
    print(f"积木块 body id: {block_body_id}")
except:
    print("未找到积木块")
    block_body_id = None

# 初始化
mujoco.mj_forward(model, data)

# 打印积木块和爪子位置
if block_body_id:
    block_pos = data.xpos[block_body_id]
    print(f"积木块位置: x={block_pos[0]:.3f}, y={block_pos[1]:.3f}, z={block_pos[2]:.3f}")

# 机械臂控制函数
def set_joint_position(positions):
    """设置所有关节位置 [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]"""
    data.ctrl[rotation_joint] = positions[0]
    data.ctrl[pitch_joint] = positions[1]
    data.ctrl[elbow_joint] = positions[2]
    data.ctrl[wrist_pitch_joint] = positions[3]
    data.ctrl[wrist_roll_joint] = positions[4]
    data.ctrl[jaw_joint] = positions[5]

def open_jaw():
    data.ctrl[jaw_joint] = 0.5

def close_jaw():
    data.ctrl[jaw_joint] = -0.5

def home_position():
    set_joint_position([0, 0, 0, 0, 0, 0])

def ready_to_grab():
    set_joint_position([1.57, -1.5, 1.5, 0, 0, 0])

def grab_low():
    set_joint_position([1.57, -2.0, 2.0, 0, 0, -0.15])

def lift_up():
    set_joint_position([1.57, -1.8, 1.8, 0, 0, -0.15])

def move_to_target():
    set_joint_position([0, -1.8, 1.8, 0, 0, -0.15])

def release_position():
    set_joint_position([0, -2.0, 2.0, 0, 0, 0])

def print_gripper_pos():
    """打印爪子末端位置"""
    jaw_pos = data.xpos[jaw_body_id]
    print(f"  爪子位置: x={jaw_pos[0]:.3f}, y={jaw_pos[1]:.3f}, z={jaw_pos[2]:.3f}")

# 初始化
mujoco.mj_forward(model, data)

print("\n开始机械臂控制...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.001
    
    try:
        # 归位
        print("1. 归位...")
        home_position()
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_gripper_pos()
        
        time.sleep(1)
        
        # 移动到积木上方准备抓取
        print("2. 移动到积木上方...")
        ready_to_grab()
        for _ in range(1000):
            mujoco.mj_step(model, data)
            viewer.sync()
        
        time.sleep(0.5)
        
        # 下降抓取
        print("3. 下降抓取...")
        grab_low()
        for _ in range(800):
            mujoco.mj_step(model, data)
            viewer.sync()
        
        time.sleep(0.5)
        
        # 抓取（闭合爪子）
        print("4. 抓取!")
        close_jaw()
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_gripper_pos()
        
        time.sleep(0.5)
        
        # 抬起
        print("5. 抬起...")
        lift_up()
        for _ in range(800):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_gripper_pos()
        
        time.sleep(0.5)
        
        # 移动到目标位置上方
        print("6. 移动到目标位置...")
        move_to_target()
        for _ in range(1200):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_gripper_pos()
        
        time.sleep(0.5)
        
        # 下降放置
        print("7. 下降放置...")
        release_position()
        for _ in range(800):
            mujoco.mj_step(model, data)
            viewer.sync()
        print_gripper_pos()
        
        # 松开爪子
        print("8. 松开爪子!")
        open_jaw()
        for _ in range(500):
            mujoco.mj_step(model, data)
            viewer.sync()
        
        time.sleep(1)
        
        # 返回归位
        print("9. 返回归位...")
        home_position()
        for _ in range(1000):
            mujoco.mj_step(model, data)
            viewer.sync()
        
        print("\n抓取搬运完成！")
        print("按 Ctrl+C 退出")
        
        # 保持显示
        while True:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        home_position()
        print("程序已停止")