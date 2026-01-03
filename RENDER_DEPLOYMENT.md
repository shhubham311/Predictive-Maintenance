# Render Deployment Guide + Cron Pinger Setup

## Step 1: Push to GitHub

```bash
cd /Users/shubh/Work/predictive-maintenance

# Initialize git (if not done)
git init
git add .
git commit -m "Add deployment config for Render"

# Create repo on GitHub, then:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/predictive-maintenance.git
git push -u origin main
```

## Step 2: Deploy on Render

1. Go to https://render.com (sign up with GitHub)
2. Click **"New +"** → **"Web Service"**
3. Select your GitHub repo
4. Fill in:
   - **Name:** `predictive-maintenance`
   - **Environment:** `Docker`
   - **Plan:** `Free`
5. Click **"Create Web Service"**
6. Wait 5-10 minutes for deployment

Your app will be at: `https://predictive-maintenance.onrender.com`

---

## Step 3: Set Up Cron Pinger (Keep App Alive)

Render's free tier sleeps your app after 15 min of inactivity. Use a cron pinger to send requests every 10 min.

### Option A: UptimeRobot (Easiest) ⭐

1. Go to https://uptimerobot.com
2. Sign up (free)
3. Click **"Add New Monitor"**
4. Fill in:
   - **Monitor Type:** `HTTP(s)`
   - **Friendly Name:** `Predictive Maintenance`
   - **URL:** `https://predictive-maintenance.onrender.com/docs`
   - **Monitoring Interval:** `5 minutes`
5. Click **"Create Monitor"**

✅ Done! It will ping your app every 5 minutes automatically.

### Option B: cron-job.org (Free Alternative)

1. Go to https://cron-job.org
2. Sign up (free)
3. Click **"Create Cronjob"**
4. Fill in:
   - **Title:** `Keep Predictive Maintenance Alive`
   - **URL:** `https://predictive-maintenance.onrender.com/docs`
   - **Execution Time:** `*/10 * * * *` (every 10 min)
5. Click **"Create"**

✅ Done!

---

## Access Your App

- **Streamlit UI:** `https://predictive-maintenance.onrender.com:8501`
- **FastAPI Docs:** `https://predictive-maintenance.onrender.com:8000/docs`

Wait, Render doesn't support multiple ports directly. You need to pick ONE port.

---

## ⚠️ IMPORTANT: Single Port Setup

Render's free tier doesn't support multiple ports. You have two options:

### Option 1: Run Only Streamlit (Recommended)
Modify your Dockerfile to run ONLY Streamlit, and embed FastAPI calls.

### Option 2: Use Gunicorn + Proxy
Run FastAPI on port 8000 as the main app, access Streamlit separately.

**I recommend creating a simple start script that runs both on the same port using Nginx, OR:**

Just modify your startup to run **only Streamlit** if you don't need the raw API endpoints.

---

## Summary

✅ Deploy on Render (free)
✅ Set up UptimeRobot cron pinger (free)
✅ App stays alive forever
✅ No cost, no credit card after 3-4 months

---

Let me know if you need help with:
- GitHub setup
- Dockerfile modifications for single port
- Cron pinger troubleshooting
