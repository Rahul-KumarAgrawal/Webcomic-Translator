import os
import re
import subprocess
import sys

def prompt_domain():
    print("\n" + "="*50)
    print(" ☁️  CBZ Translator - Cloud Deployment Wizard ")
    print("="*50)
    print("\nTo deploy the UI to Vercel, you need an ngrok static domain.")
    print("If you don't have one, get a free static domain from: https://dashboard.ngrok.com/cloud-edge/domains")
    print("\nEnter your exact ngrok domain (e.g., cod-concave-glucose.ngrok-free.dev).")
    print("Do NOT include 'https://' or trailing slashes.")
    
    domain = input("\nNgrok Domain: ").strip()
    
    # Clean up user input just in case they added https://
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]
    
    if not domain or "ngrok" not in domain:
        print("\n[!] Warning: This doesn't look like a standard ngrok domain.")
        confirm = input("Are you sure you want to use '%s'? (y/n): " % domain)
        if confirm.lower() != 'y':
            sys.exit(1)
            
    return domain

def update_file(filepath, pattern, replacement, success_msg):
    if not os.path.exists(filepath):
        print(f"[!] Error: Could not find {filepath}")
        return False
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    new_content, count = re.subn(pattern, replacement, content)
    
    if count > 0:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"[OK] {success_msg}")
        return True
    else:
        print(f"[!] Warning: Pattern not found in {filepath}. It might already be updated.")
        return False

def main():
    domain = prompt_domain()
    backend_url = f"https://{domain}"
    
    print("\n[1/3] Injecting your ngrok domain into configuration files...")
    
    # 1. Update run_web.bat
    update_file(
        "run_web.bat",
        r'npx ngrok http 5000 --domain=[^\s"]+',
        f'npx ngrok http 5000 --domain={domain}',
        "Updated local backend ngrok tunnel (run_web.bat)"
    )
    
    # 2. Update main.js (both local and vercel_deploy copies)
    js_pattern = r'const BACKEND_API_BASE\s*=\s*"[^"]+";'
    js_replacement = f'const BACKEND_API_BASE = "{backend_url}";'
    
    update_file("web/static/main.js", js_pattern, js_replacement, "Updated local UI JavaScript (web/static/main.js)")
    update_file("vercel_deploy/web/static/main.js", js_pattern, js_replacement, "Updated Vercel UI JavaScript (vercel_deploy/web/static/main.js)")
    
    print("\n[2/3] Authenticating with Vercel...")
    print("If you are not logged in, Vercel will open a browser window to authenticate.")
    try:
        subprocess.run(["npx", "vercel", "login"], check=True, cwd="vercel_deploy", shell=True)
    except subprocess.CalledProcessError:
        print("\n[!] Vercel login failed or was cancelled. Exiting.")
        sys.exit(1)
        
    print("\n[3/3] Deploying to Vercel Production...")
    print("This will take about 30-60 seconds. Please wait...")
    try:
        subprocess.run(["npx", "vercel", "--prod", "--yes"], check=True, cwd="vercel_deploy", shell=True)
        print("\n" + "="*50)
        print(" 🎉 Deployment Complete!")
        print("="*50)
        print("\nYour website is live! You can now access it from any device.")
        print("IMPORTANT: Leave 'run_web.bat' running on your PC so the website can communicate with your GPU.")
        print("\nPress any key to exit.")
    except subprocess.CalledProcessError:
        print("\n[!] Deployment failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
