import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.join(ROOT, "vercel_deploy")

if os.path.exists(DEPLOY_DIR):
    shutil.rmtree(DEPLOY_DIR)
os.makedirs(DEPLOY_DIR, exist_ok=True)

# 1. Copy web folder
shutil.copytree(os.path.join(ROOT, "web"), os.path.join(DEPLOY_DIR, "web"), ignore=shutil.ignore_patterns("sessions", "__pycache__"))

# 2. Copy vercel.json
shutil.copy2(os.path.join(ROOT, "vercel.json"), os.path.join(DEPLOY_DIR, "vercel.json"))

# 3. Copy config folder (only yaml settings)
os.makedirs(os.path.join(DEPLOY_DIR, "config"), exist_ok=True)
shutil.copy2(os.path.join(ROOT, "config", "settings.yaml"), os.path.join(DEPLOY_DIR, "config", "settings.yaml"))

print("SUCCESS: Clean vercel_deploy folder prepared with only web files!")
