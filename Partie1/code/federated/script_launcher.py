import subprocess
import time
import sys

PYTHON = sys.executable

subprocess.Popen([PYTHON, "server_flower.py"])
time.sleep(1)

for cid in range(3):
    subprocess.Popen([PYTHON, "client_flower.py", str(cid)])
