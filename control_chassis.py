import mujoco
import mujoco.viewer
import time
import numpy as np

# 加载底盘模型
model = mujoco.MjModel.from_xml_path('chassis_corrected.xml')
data = mujoco.MjData(model)

# 电机ID
try:
    drive_motor_1 = model.actuator('drive_motor_1').id
    drive_motor_2 = model.actuator('drive_motor_2').id
    drive_motor_3 = model.actuator('drive_motor_3').id
    print(f"电机ID获取成功: drive_motor_1={drive_motor_1}, drive_motor_2={drive_motor_2}, drive_motor_3={drive_motor_3}")
except Exception as e:
    print(f"电机ID获取失败: {e}")
    print("可用的执行器:")
    for i in range(model.nu):
        print(f"执行器{i}: {model.actuator(i).name}")
    exit(1)

# 打印关节信息用于调试
print("\n=== 电机和关节信息 ===")
for i in range(model.nu):
    act = model.actuator(i)
    print(f"电机 {i}: {act.name}, gear: {act.gear}")

# 打印所有关节
print("\n所有关节:")
for i in range(model.njnt):
    jnt = model.jnt(i)
    print(f"关节 {i}: {model.jnt(i).name}, type: {jnt.type}, axis: {jnt.axis}")
print("=======================\n")

# 控制参数 - 使用合理的扭矩值
# 注意: 在MuJoCo中，motor执行器的ctrl值是力矩(扭矩)，不是速度！
# 对于这个小底盘，0.5-2.0 Nm的扭矩就足够了
speed = 1.0  # 扭矩值 (Nm)

def move_forward():
    data.ctrl[drive_motor_1] = -speed
    data.ctrl[drive_motor_2] = -speed
    data.ctrl[drive_motor_3] = -speed

def stop():
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0

print("开始运行底盘控制程序...")

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.001
    
    try:
        print("前进10秒...")
        move_forward()
        start_time = time.time()
        while time.time() - start_time < 10.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
        
        print("停止!")
        stop()
        
        # 保持显示
        while True:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        stop()
        print("程序已停止")