import mujoco
import mujoco.viewer
import time
import numpy as np

# 加载底盘模型
model = mujoco.MjModel.from_xml_path('chassis.xml')
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

# 获取关节ID
joint_1 = model.joint('ST3215_Servo_Motor-v1-2_Hub---Servo').id
joint_2 = model.joint('ST3215_Servo_Motor-v1-1_Hub-2---Servo').id
joint_3 = model.joint('ST3215_Servo_Motor-v1_Revolute-40').id
print(f"关节ID: joint_1={joint_1}, joint_2={joint_2}, joint_3={joint_3}")

# 打印关节信息
for i in range(model.njnt):
    joint = model.joint(i)
    if joint.type != 0:
        print(f"关节{i}: {joint.name}, type={joint.type}, range={joint.range}")

print("\n开始测试...")

# 初始化模拟
mujoco.mj_forward(model, data)

# 启动viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    model.opt.timestep = 0.001
    
    # 测试1: 施加大力矩
    print("\n=== 测试1: 施加大力矩 (500.0) ===")
    data.ctrl[drive_motor_1] = 500.0
    data.ctrl[drive_motor_2] = 500.0
    data.ctrl[drive_motor_3] = 500.0
    
    start_time = time.time()
    for step in range(5000):
        mujoco.mj_step(model, data)
        viewer.sync()
        if step % 500 == 0:
            print(f"Step {step}: qpos[{joint_1}]={data.qpos[joint_1]:.4f}, "
                  f"qvel[{joint_1}]={data.qvel[joint_1]:.4f}, "
                  f"qfrc_applied[{joint_1}]={data.qfrc_applied[joint_1]:.4f}")
        time.sleep(0.001)
    
    # 停止
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0
    print("停止")
    time.sleep(1)
    
    # 测试2: 检查摩擦力
    print("\n=== 测试2: 关闭摩擦力后的测试 ===")
    # 临时降低摩擦力
    for i in range(model.njnt):
        joint = model.joint(i)
        if joint.frictionloss != 0:
            joint.frictionloss = 0.0
            print(f"关节{i} ({joint.name}) 摩擦力设为 0")
    
    data.ctrl[drive_motor_1] = 50.0
    data.ctrl[drive_motor_2] = 50.0
    data.ctrl[drive_motor_3] = 50.0
    
    start_time = time.time()
    for step in range(5000):
        mujoco.mj_step(model, data)
        viewer.sync()
        if step % 500 == 0:
            print(f"Step {step}: qpos[0]={data.qpos[0]:.4f}, qpos[1]={data.qpos[1]:.4f}, qpos[2]={data.qpos[2]:.4f}")
            print(f"         关节1: qpos={data.qpos[joint_1]:.4f}, qvel={data.qvel[joint_1]:.4f}")
        time.sleep(0.001)
    
    print("\n测试完成")