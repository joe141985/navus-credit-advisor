#!/bin/bash
# NAVUS Online Deployment Script
# Automated deployment to Vercel + Render

echo "🚀 NAVUS Online Deployment Assistant"
echo "===================================="

# Check if we're in the right directory
if [ ! -d "WebApp" ]; then
    echo "❌ Error: Run this script from the NAVUS root directory"
    exit 1
fi

# Check for required tools
command -v git >/dev/null 2>&1 || { echo "❌ Git is required but not installed. Install git first."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ Node.js/npm is required but not installed. Install Node.js first."; exit 1; }

echo "✅ Prerequisites checked"

# Step 1: Prepare for deployment
echo ""
echo "📋 Step 1: Preparing deployment files..."
cd WebApp
python prepare_for_deployment.py

# Step 2: GitHub setup
echo ""
echo "📋 Step 2: GitHub Repository Setup"
echo "================================="

read -p "Have you created a GitHub repository yet? (y/n): " github_ready

if [[ $github_ready == "n" || $github_ready == "N" ]]; then
    echo ""
    echo "📝 Please follow these steps:"
    echo "1. Go to https://github.com and create a new repository"
    echo "2. Name it: navus-credit-advisor" 
    echo "3. Make it Public (required for free hosting)"
    echo "4. Don't initialize with README"
    echo "5. Copy the repository URL (e.g., https://github.com/username/navus-credit-advisor.git)"
    echo ""
    read -p "Enter your GitHub repository URL: " github_url
    
    # Initialize and push to GitHub
    cd ..
    git init
    git add .
    git commit -m "Initial NAVUS credit card advisor deployment"
    git branch -M main
    git remote add origin "$github_url"
    
    echo "📤 Pushing to GitHub..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed to GitHub!"
    else
        echo "❌ Failed to push to GitHub. Please check your repository URL and try again."
        exit 1
    fi
    
    cd WebApp
fi

# Step 3: Backend deployment (Render)
echo ""
echo "📋 Step 3: Backend Deployment (Render)"
echo "======================================"

echo "🔗 Please follow these steps to deploy your backend:"
echo ""
echo "1. Go to https://render.com and sign up/login"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub account"
echo "4. Select your 'navus-credit-advisor' repository"
echo "5. Configure the service:"
echo "   - Name: navus-api"
echo "   - Runtime: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: python backend_production.py"
echo "6. Click 'Create Web Service'"
echo "7. Wait for deployment (5-10 minutes)"
echo ""

read -p "Once deployed, enter your Render backend URL (e.g., https://navus-api.onrender.com): " backend_url

if [ -z "$backend_url" ]; then
    echo "❌ Backend URL is required for frontend deployment"
    exit 1
fi

echo "✅ Backend URL recorded: $backend_url"

# Step 4: Update frontend with backend URL
echo ""
echo "📋 Step 4: Updating frontend configuration..."

# Update the API URL in the frontend
cd frontend

# Create environment file for production
cat > .env.production << EOF
REACT_APP_API_URL=$backend_url
EOF

echo "✅ Frontend configured with backend URL"

# Step 5: Frontend deployment (Vercel)
echo ""
echo "📋 Step 5: Frontend Deployment (Vercel)"
echo "======================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "🔐 Please login to Vercel..."
vercel login

if [ $? -ne 0 ]; then
    echo "❌ Failed to login to Vercel. Please try again."
    exit 1
fi

echo "🚀 Deploying frontend to Vercel..."
vercel --prod

if [ $? -eq 0 ]; then
    echo "✅ Frontend deployed successfully!"
    echo ""
    echo "🎉 DEPLOYMENT COMPLETE!"
    echo "======================"
    echo ""
    echo "Your NAVUS Credit Card Advisor is now LIVE online!"
    echo ""
    echo "📱 Frontend URL: Check the Vercel output above"
    echo "🔧 Backend URL: $backend_url"
    echo "📋 API Docs: $backend_url/docs"
    echo ""
    echo "✨ Features Available:"
    echo "   • AI-powered credit card recommendations"
    echo "   • User persona profiles"  
    echo "   • Smart follow-up suggestions"
    echo "   • Mobile-responsive design"
    echo "   • Real-time performance tracking"
    echo ""
    echo "🎯 Ready for investor demos and public use!"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Test your live app thoroughly"
    echo "   2. Share URL with investors/users"
    echo "   3. Monitor usage and performance"
    echo "   4. Consider custom domain for production"
    
else
    echo "❌ Frontend deployment failed. Please check the errors above."
    echo "💡 You can try manual deployment:"
    echo "   cd WebApp/frontend"
    echo "   vercel --prod"
    exit 1
fi

echo ""
echo "🎊 Congratulations! Your NAVUS app is now accessible worldwide!"