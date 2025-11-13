# delayed_run.py
import time
import subprocess

# 等待 2 小时（单位：秒）
time.sleep(2 * 60 * 60)

# 启动 script.py
subprocess.run(["python", "script.py"])
