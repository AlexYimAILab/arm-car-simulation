import mujoco
import mujoco.viewer
import time
import numpy as np

model = mujoco.MjModel.from_xml_path('chassis.xml')
data = mujoco.MjData(model)

try:
    drive_motor_1 = model.actuator('drive_motor_1').id
    drive_motor_2 = model.actuator('drive_motor_2').id
    drive_motor_3 = model.actuator('drive_motor_3').id
    print(f"电机ID: drive_motor_1={drive_motor_1}, drive_motor_2={drive_motor_2}, drive_motor_3={drive_motor_3}")
except Exception as e:
    print(f"获取电机ID失败: {e}")
    exit(1)

speed = 200.0
target_distance = 0.05

mujoco.mj_forward(model, data)
start_x = data.qpos[0]
start_y = data.qpos[1]

def move_forward():
    data.ctrl[drive_motor_1] = speed
    data.ctrl[drive_motor_2] = speed
    data.ctrl[drive_motor_3] = speed

def stop():
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0

print(f"起始位置: x={start_x:.4f}, y={start_y:.4f}")
print(f"目标距离: {target_distance} 米 (5cm)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.001
    
    try:
        print("开始前进...")
        move_forward()
        
        while True:
            mujoco.mj_step(model, data)
            viewer.sync()
            
            current_x = data.qpos[0]
            current_y = data.qpos[1]
            distance = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            
            if distance >= target_distance:
                stop()
                print(f"已前进 {distance:.4f} 米，停止!")
                print(f"最终位置: x={current_x:.4f}, y={current_y:.4f}")
                break
            
            time.sleep(0.001)
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        stop()
        print("程序已停止")