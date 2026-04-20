import mujoco
import mujoco.viewer
import time
import numpy as np

import subprocess
import os

if not os.path.exists('chassis_v2.xml'):
    print("创建修复后的 chassis_v2.xml...")
    with open('chassis.xml', 'r') as f:
        lines = f.readlines()
    with open('chassis_v2.xml', 'w') as f:
        f.writelines(lines[:171])
    print("chassis_v2.xml 创建完成！")

model = mujoco.MjModel.from_xml_path('chassis_v2.xml')
data = mujoco.MjData(model)
print("模型加载成功")

try:
    drive_motor_1 = model.actuator('drive_motor_1').id
    drive_motor_2 = model.actuator('drive_motor_2').id
    drive_motor_3 = model.actuator('drive_motor_3').id
    print(f"电机ID: drive_motor_1={drive_motor_1}, drive_motor_2={drive_motor_2}, drive_motor_3={drive_motor_3}")
except Exception as e:
    print(f"获取电机ID失败: {e}")
    drive_motor_1 = 0
    drive_motor_2 = 1
    drive_motor_3 = 2
    print(f"使用默认电机ID: {drive_motor_1}, {drive_motor_2}, {drive_motor_3}")

mujoco.mj_forward(model, data)
start_x = data.qpos[0]
start_y = data.qpos[1]
print(f"起始位置: x={start_x:.4f}, y={start_y:.4f}")

speed = 500.0

def stop():
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0
    print("停止所有电机")

def move_forward_time(duration_seconds):
    global start_x, start_y, start_theta
    current_x = data.qpos[0]
    current_y = data.qpos[1]
    current_theta = data.qpos[3] if model.nq >= 4 else 0
    start_x = current_x
    start_y = current_y
    start_theta = current_theta
    
    print(f"开始前进 {duration_seconds} 秒...")
    print(f"起始位置: x={start_x:.4f}, y={start_y:.4f}, theta={start_theta:.4f}")
    
    base_body_id = model.body('base_plate_layer1-v5-1').id
    wheel1_body_id = model.body('4-Omni-Directional-Wheel_Single_Body-v1').id
    wheel2_body_id = model.body('4-Omni-Directional-Wheel_Single_Body-v1-1').id
    wheel3_body_id = model.body('4-Omni-Directional-Wheel_Single_Body-v1-2').id
    
    print(f"Body IDs - 底盘: {base_body_id}, 轮1: {wheel1_body_id}, 轮2: {wheel2_body_id}, 轮3: {wheel3_body_id}")
    
    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        data.ctrl[drive_motor_1] = -speed
        data.ctrl[drive_motor_2] = speed * 0.5
        data.ctrl[drive_motor_3] = speed * 0.5
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
        
        if int(time.time() - start_time) % 2 == 0 and time.time() - start_time > 0.1:
            base_pos = data.xpos[base_body_id]
            wheel1_pos = data.xpos[wheel1_body_id]
            print(f"底盘: ({base_pos[0]:.3f}, {base_pos[1]:.3f}), 轮1: ({wheel1_pos[0]:.3f}, {wheel1_pos[1]:.3f})", end='\r')
        
        current_x = data.qpos[0]
        current_y = data.qpos[1]
        current_theta = data.qpos[3] if model.nq >= 4 else 0
        elapsed = time.time() - start_time
        
        print(f"时间: {elapsed:.1f}s, 位置: x={current_x:.4f}, y={current_y:.4f}, theta={current_theta:.4f}", end='\r')
    
    stop()
    final_base_pos = data.xpos[base_body_id]
    final_wheel1_pos = data.xpos[wheel1_body_id]
    
    print(f"\n前进 {duration_seconds} 秒完成！")
    print(f"最终位置: x={current_x:.4f}, y={current_y:.4f}, theta={current_theta:.4f}")
    print(f"底盘世界坐标: ({final_base_pos[0]:.4f}, {final_base_pos[1]:.4f}, {final_base_pos[2]:.4f})")
    print(f"轮1世界坐标: ({final_wheel1_pos[0]:.4f}, {final_wheel1_pos[1]:.4f}, {final_wheel1_pos[2]:.4f})")
    print(f"底盘与轮1距离: {np.linalg.norm(final_base_pos - final_wheel1_pos):.4f} 米")
    distance = np.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
    print(f"总前进距离: {distance:.4f} 米")

print(f"模型执行器数量: {model.nu}")
print(f"模型自由度: {model.nv}")
print(f"模型关节数量: {model.njnt}")

print("启动虚拟环境...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.001
    model.opt.iterations = 50
    
    try:
        print("开始前进 10 秒...")
        move_forward_time(10.0)
        
        print("\n按 Ctrl+C 退出...")
        while True:
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        stop()
        print("\n程序已停止")