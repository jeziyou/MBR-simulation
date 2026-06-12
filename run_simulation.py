"""
MBR仿真系统运行脚本 - 修正文件路径版本
"""
import os
import sys

# 确保输出目录存在
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 修改app.py中的硬编码路径
import importlib.util
spec = importlib.util.spec_from_file_location("app", "app.py")
app_module = importlib.util.module_from_spec(spec)

# 在执行前替换路径
app_code = open("app.py", "r", encoding="utf-8").read()
app_code = app_code.replace("/workspace/mbr_simulation/", output_dir + "/")

# 执行修改后的代码
exec(app_code, app_module.__dict__)
