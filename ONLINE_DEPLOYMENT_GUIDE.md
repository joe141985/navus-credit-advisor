# 🌐 NAVUS ONLINE DEPLOYMENT GUIDE

## Complete Step-by-Step Instructions for Public Hosting

---

## 🚀 **OPTION A: VERCEL + RENDER (RECOMMENDED)**

**Cost: FREE for 30 days, then $5-10/month**  
**Setup Time: 15 minutes**  
**Best for: Investor demos, small-scale production**

### **Step 1: Deploy Backend to Render**

1. **Sign up for Render**: Go to [render.com](https://render.com) and create account

2. **Connect GitHub**: 
   - First, push your code to GitHub (instructions below)
   - In Render dashboard, click "New +" → "Web Service"
   - Connect your GitHub account
   - Select your NAVUS repository

3. **Configure Web Service**:
   ```
   Name: navus-api
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python backend_production.py
   ```

4. **Set Environment Variables** (if needed):
   ```
   PORT=8000
   PYTHON_VERSION=3.9
   ```

5. **Deploy**: Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - You'll get a URL like: `https://navus-api.onrender.com`

### **Step 2: Deploy Frontend to Vercel**

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   # Follow prompts to authenticate
   ```

3. **Deploy Frontend**:
   ```bash
   cd /Users/joebanerjee/NAVUS/WebApp/frontend
   
   # Update API URL to your Render backend
   # Edit src/App.js and change API_BASE_URL to your Render URL
   
   # Deploy to Vercel
   vercel --prod
   ```

4. **Follow Prompts**:
   ```
   Set up and deploy? [Y/n] Y
   Which scope? → Your username
   Link to existing project? [y/N] N  
   What's your project's name? navus-credit-advisor
   In which directory is your code located? ./
   ```

5. **Get Your Live URL**:
   - Vercel will give you a URL like: `https://navus-credit-advisor.vercel.app`
   - Your app is now LIVE! 🎉

---

## 🚀 **OPTION B: NETLIFY + RAILWAY**

**Cost: FREE tier available**  
**Setup Time: 10 minutes**  
**Best for: Quick demos**

### **Backend on Railway:**

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your NAVUS repo
5. Choose "WebApp" folder
6. Railway auto-deploys your backend
7. Get URL like: `https://navus-api-production.up.railway.app`

### **Frontend on Netlify:**

1. Go to [netlify.com](https://netlify.com)
2. Sign up and connect GitHub
3. Click "New site from Git"
4. Choose your NAVUS repo
5. Set:
   ```
   Base directory: WebApp/frontend
   Build command: npm run build
   Publish directory: build
   ```
6. Deploy! Get URL like: `https://navus-advisor.netlify.app`

---

## 🚀 **OPTION C: FULL VPS DEPLOYMENT**

**Cost: $5-20/month (DigitalOcean, Linode, AWS)**  
**Setup Time: 30-45 minutes**  
**Best for: Full production control**

### **Step 1: Get a VPS**

1. **DigitalOcean** (recommended for beginners):
   - Sign up at [digitalocean.com](https://digitalocean.com)
   - Create a "Droplet" (Ubuntu 20.04, $5/month basic)
   - Note your IP address

2. **Connect via SSH**:
   ```bash
   ssh root@your-server-ip
   ```

### **Step 2: Setup Server**

```bash
# Update system
apt update && apt upgrade -y

# Install Node.js 18
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
apt-get install -y nodejs

# Install Python 3 and pip
apt install -y python3 python3-pip python3-venv

# Install PM2 for process management
npm install -g pm2

# Install nginx for reverse proxy
apt install -y nginx

# Install certbot for SSL
apt install -y certbot python3-certbot-nginx
```

### **Step 3: Deploy Your Code**

```bash
# Clone your repository
git clone https://github.com/yourusername/navus.git
cd navus/WebApp

# Setup Python backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd frontend
npm install
npm run build
cd ..

# Start services with PM2
pm2 start backend_production.py --name navus-api --interpreter python3
pm2 serve frontend/build 3000 --name navus-frontend

# Save PM2 configuration
pm2 save
pm2 startup
```

### **Step 4: Configure Nginx**

```bash
# Create nginx config
nano /etc/nginx/sites-available/navus

# Add this configuration:
```

```nginx
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    # Frontend
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Enable the site
ln -s /etc/nginx/sites-available/navus /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl reload nginx

# Get SSL certificate (optional)
certbot --nginx -d your-domain.com
```

---

## 📋 **BEFORE YOU DEPLOY: PUSH TO GITHUB**

### **Step 1: Create GitHub Repository**

1. Go to [github.com](https://github.com) and create account
2. Click "New repository"
3. Name: `navus-credit-advisor`
4. Make it Public (for free hosting)
5. Don't initialize with README

### **Step 2: Push Your Code**

```bash
cd /Users/joebanerjee/NAVUS

# Initialize git repository
git init
git add .
git commit -m "Initial NAVUS credit card advisor"

# Add GitHub remote
git remote add origin https://github.com/yourusername/navus-credit-advisor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## 🎯 **RECOMMENDED DEPLOYMENT WORKFLOW**

### **For Investor Demo (Next 2 Hours):**

1. **Quick Setup**: Use Option A (Vercel + Render)
2. **Steps**:
   ```bash
   # 1. Push to GitHub (5 min)
   # 2. Deploy backend to Render (10 min)
   # 3. Deploy frontend to Vercel (5 min)
   # 4. Test and share URL with investors
   ```

### **For Serious Production (This Week):**

1. **Professional Setup**: Use Option C (VPS)
2. **Get Custom Domain**: Like `navus.ai` or `navusadvisor.com`
3. **Add SSL Certificate**: For https://
4. **Setup Monitoring**: Add error tracking and analytics

---

## 🔧 **DEPLOYMENT CHECKLIST**

### **Before Going Live:**

- [ ] Test locally with `./run_app_enhanced.sh`
- [ ] Push all code to GitHub
- [ ] Update API URLs in frontend
- [ ] Test all major features work
- [ ] Prepare demo script for investors

### **After Deployment:**

- [ ] Test live URL works
- [ ] Try different user personas
- [ ] Test on mobile devices
- [ ] Check API response times
- [ ] Share URL with team/investors

### **Demo URLs to Share:**

```
🌐 Live App: https://navus-credit-advisor.vercel.app
🔧 API Docs: https://navus-api.onrender.com/docs
📱 Mobile: Same URL works on mobile
```

---

## 🎉 **CONGRATULATIONS!**

Once deployed, you'll have:

✅ **Professional URL** to share with investors  
✅ **Mobile-responsive** web app  
✅ **Live API** with documentation  
✅ **24/7 availability** for demos  
✅ **Scalable architecture** ready for growth  

**Your NAVUS credit card advisor is now accessible worldwide!** 🌍🇨🇦💳

---

## 🆘 **NEED HELP?**

**Common Issues:**

- **Build fails**: Check Node.js version (need 16+)
- **API connection error**: Verify backend URL in frontend
- **Slow responses**: Normal on free tiers, upgrade for production
- **SSL issues**: Use http:// for testing, https:// for production

**Quick Test Commands:**
```bash
# Test backend
curl https://your-backend-url.com/health

# Test frontend
curl https://your-frontend-url.com

# Test API
curl -X POST https://your-backend-url.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the best travel card?"}'
```

**Your NAVUS system is now ready to impress investors with a professional online presence!** 🚀