import re

# 读取文件
with open('chassis.xml', 'r') as f:
    content = f.read()

# 替换注释的地面为实际的地面
# 找到所有 <!-- <geom pos="0 0 0" size="1 1 1" type="plane" ... --> 并替换
pattern = r'<!-- <geom pos="0 0 0" size="1 1 1" type="plane" rgba="1 0\.83 0\.61 0\.5" /> -->'
replacement = '<geom pos="0 0 -0.001" size="5 5 0.001" type="plane" rgba="0.8 0.8 0.8 1" />'

# 执行替换
new_content = re.sub(pattern, replacement, content)

# 写入文件
with open('chassis.xml', 'w') as f:
    f.write(new_content)

print("地面已添加！")
print("修改后的地面定义: <geom pos=\"0 0 -0.001\" size=\"5 5 0.001\" type=\"plane\" rgba=\"0.8 0.8 0.8 1\" />")