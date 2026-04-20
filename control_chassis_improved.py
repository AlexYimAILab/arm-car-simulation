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
    # 打印所有可用的执行器
    print("可用的执行器:")
    for i in range(model.nu):
        print(f"执行器{i}: {model.actuator(i).name}")
    exit(1)

# 控制参数
# force_scale = 300.0  # 增大力度
# forward_duration = 5.0
# turn_duration = 2.0
# 将第26-27行的参数改为：
force_scale = 500.0  # 进一步增大力度
forward_duration = 10.0  # 延长前进时间
turn_duration = 3.0  # 延长转弯时间


# 全向轮底盘控制函数（根据关节轴向调整）
# 根据关节定义，所有关节都是绕Z轴旋转的hinge关节
def move_forward():
    """前进"""
    # 对于全向轮平台，通过不同轮子的速度组合实现前进
    data.ctrl[drive_motor_1] = force_scale
    data.ctrl[drive_motor_2] = force_scale
    data.ctrl[drive_motor_3] = force_scale
    print(f"前进: 电机1={force_scale}, 电机2={force_scale}, 电机3={force_scale}")

def move_backward():
    """后退"""
    data.ctrl[drive_motor_1] = -force_scale
    data.ctrl[drive_motor_2] = -force_scale
    data.ctrl[drive_motor_3] = -force_scale
    print(f"后退: 电机1={-force_scale}, 电机2={-force_scale}, 电机3={-force_scale}")

def turn_left():
    """左转"""
    # 全向轮平台的差速转向
    data.ctrl[drive_motor_1] = force_scale
    data.ctrl[drive_motor_2] = -force_scale
    data.ctrl[drive_motor_3] = -force_scale
    print(f"左转: 电机1={force_scale}, 电机2={-force_scale}, 电机3={-force_scale}")

def turn_right():
    """右转"""
    data.ctrl[drive_motor_1] = -force_scale
    data.ctrl[drive_motor_2] = force_scale
    data.ctrl[drive_motor_3] = force_scale
    print(f"右转: 电机1={-force_scale}, 电机2={force_scale}, 电机3={force_scale}")

def stop():
    """停止"""
    data.ctrl[drive_motor_1] = 0.0
    data.ctrl[drive_motor_2] = 0.0
    data.ctrl[drive_motor_3] = 0.0
    print(f"停止: 电机1=0, 电机2=0, 电机3=0")

# 运行模拟
print("开始运行底盘控制程序...")

# 初始化模拟
mujoco.mj_forward(model, data)

# 启动viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 调整物理参数
    model.opt.timestep = 0.001  # 更小的时间步长
    model.opt.iterations = 100  # 增加迭代次数以提高稳定性
    model.opt.noslip_iterations = 100
    
    # 主循环
    try:
        step_count_total = 0
        while True:
            # 前进5秒
            print("前进...")
            move_forward()
            start_time = time.time()
            step_count = 0
            while time.time() - start_time < forward_duration:
                mujoco.mj_step(model, data)
                viewer.sync()
                step_count += 1
                step_count_total += 1
                if step_count % 200 == 0:  # 每200步打印一次，因为时间步长变小了
                    print(f"位置: x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}")
                    print(f"速度: x={data.qvel[0]:.3f}, y={data.qvel[1]:.3f}, z={data.qvel[2]:.3f}")
                time.sleep(0.001)
            
            # 停止
            stop()
            start_time = time.time()
            while time.time() - start_time < 0.5:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)
            
            # 左转
            print("左转...")
            turn_left()
            start_time = time.time()
            step_count = 0
            while time.time() - start_time < turn_duration:
                mujoco.mj_step(model, data)
                viewer.sync()
                step_count += 1
                step_count_total += 1
                if step_count % 200 == 0:
                    print(f"位置: x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}")
                    print(f"速度: x={data.qvel[0]:.3f}, y={data.qvel[1]:.3f}, z={data.qvel[2]:.3f}")
                time.sleep(0.001)
            
            # 停止
            stop()
            start_time = time.time()
            while time.time() - start_time < 0.5:
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)
            
    except KeyboardInterrupt:
        # 停止所有电机
        stop()
        print("程序已停止")