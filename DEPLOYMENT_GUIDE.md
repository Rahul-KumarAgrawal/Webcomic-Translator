# ☁️ Cloud Deployment Guide

This project features a **Decoupled Architecture**. This means you can host the website interface on the cloud (Vercel) so you can access it from your phone or laptop anywhere in the world, while all the heavy AI processing is securely tunneled back to your local PC's graphics card.

This guide explains how a **new user** can set up their own personal Cloud Deployment from scratch.

---

## 📋 Prerequisites
Before you begin, ensure you have:
1. **Node.js** installed on your PC (Download from [nodejs.org](https://nodejs.org/)).
2. A free [Vercel](https://vercel.com/) account.
3. A free [ngrok](https://ngrok.com/) account.

---

## Step 1: Claim a Static Ngrok Domain
We use **ngrok** to create a secure tunnel from the public internet directly to your local Flask backend.

1. Sign up/log in at the [ngrok dashboard](https://dashboard.ngrok.com/).
2. Go to **Cloud Edge -> Domains** in the left sidebar.
3. Click **"Create Domain"** or claim your free static domain (e.g., `your-name.ngrok-free.dev`).
4. On your local PC, open a command prompt and authenticate ngrok using your token (found in your dashboard):
   ```bash
   npx ngrok config add-authtoken <YOUR_AUTH_TOKEN>
   ```
5. Open the `run_web.bat` file in a text editor.
6. Find the line that starts the ngrok tunnel (around line 32) and replace the domain with your new domain:
   ```bat
   start "ngrok Tunnel" cmd /k "npx ngrok http 5000 --domain=your-name.ngrok-free.dev"
   ```

---

## Step 2: Connect the Frontend to Your Tunnel
The Vercel UI needs to know the address of your specific ngrok tunnel so it can send translation requests to your PC.

1. Open the file located at: `vercel_deploy\web\static\main.js`
2. At the very top of the file, find the `BACKEND_API_BASE` variable.
3. Change it to match your exact ngrok domain:
   ```javascript
   const BACKEND_API_BASE = "https://your-name.ngrok-free.dev";
   ```
*(Do not put a trailing slash `/` at the end of the URL).*

---

## Step 3: Deploy to Vercel
Now you will upload the frontend UI code to Vercel.

1. Open a terminal or command prompt inside the `vercel_deploy` folder.
2. Log into Vercel via the CLI:
   ```bash
   npx vercel login
   ```
3. Deploy the application to production:
   ```bash
   npx vercel --prod
   ```
4. Follow the on-screen prompts (accept the defaults by pressing Enter).
5. Vercel will give you a live Production URL (e.g., `https://my-comic-translator.vercel.app`).

---

## Step 4: Run and Access!
You are fully set up! Here is your daily workflow:

1. **Start the Engine:** Double-click `run_web.bat` on your PC. Leave the black command prompt windows running in the background.
2. **Access Anywhere:** Open your Vercel URL on any device (phone, tablet, work laptop).
3. **Translate:** Upload a file. The UI will instantly beam the file securely to your PC at home, process it using your GPU, stream the live progress bar to your phone, and let you download the translated `.cbz` file directly!

---

## 💡 Alternative: Cloudflare Tunnels
If you run into ngrok free-tier limits, you can use **Cloudflare Tunnels**. 

*Note: Free Cloudflare Quick Tunnels change their URL every time you restart them. To use Cloudflare permanently, you must own a custom domain and set up a named tunnel.*

1. Install Cloudflare: `npm install -g cloudflared`
2. Authenticate: `cloudflared tunnel login`
3. Create a tunnel: `cloudflared tunnel create manga-translator`
4. Route it to your local port 5000 and attach it to your domain.
5. Update `main.js` with your new Cloudflare URL and redeploy to Vercel!
