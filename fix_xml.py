# 创建 fix_xml.py 运行
with open('chassis.xml', 'r') as f:
    content = f.read()

# 找到第一个 <mujoco 和最后一个 </mujoco>
start = content.find('<mujoco')
end = content.rfind('</mujoco>') + len('</mujoco>')

# 提取并保存
fixed = content[start:end]
with open('chassis_fixed.xml', 'w') as f:
    f.write(fixed)

print(f"已保存修复后的文件到 chassis_fixed.xml")