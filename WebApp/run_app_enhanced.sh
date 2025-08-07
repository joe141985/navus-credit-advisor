#!/bin/bash
# NAVUS Enhanced Web App Launcher v2.0

echo "🚀 Starting NAVUS Credit Card Advisor Web App v2.0"
echo "================================================="

# Check if we're in the right directory
if [ ! -f "backend_optimized.py" ]; then
    echo "❌ Error: backend_optimized.py not found. Please run from the WebApp directory."
    exit 1
fi

# Check for fine-tuned model
if [ -d "navus_mistral_finetuned" ]; then
    echo "✅ Fine-tuned model found!"
    MODEL_STATUS="🧠 Using Fine-tuned NAVUS Model"
else
    echo "⚠️  Fine-tuned model not found, will use base model"
    MODEL_STATUS="⚠️  Using Base Model (Fine-tuned model recommended)"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Copy optimized files over original ones
echo "🔄 Setting up enhanced versions..."
cp backend_optimized.py backend.py
cd frontend/src
cp App_enhanced.js App.js
cp App_enhanced.css App.css
cd ../..

# Start backend in background
echo "🔥 Starting enhanced backend API server..."
echo "$MODEL_STATUS"
python backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Check if backend is running
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "✅ Backend started successfully"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Install and start frontend
echo "🎨 Starting frontend development server..."
cd frontend

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Node dependencies"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
fi

# Start frontend
echo "✅ Starting React development server..."
npm start &
FRONTEND_PID=$!

# Wait a moment and check if frontend started
sleep 3
if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "✅ Frontend started successfully"
else
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Go back to WebApp directory
cd ..

# Success message
echo ""
echo "🎉 NAVUS Enhanced Web App is now running!"
echo "========================================="
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📋 API Documentation: http://localhost:8000/docs"
echo "$MODEL_STATUS"
echo ""
echo "✨ New Features:"
echo "   • 👤 User profiles and personas"
echo "   • 💡 Smart suggested questions"
echo "   • 🏷️ Categorized preset questions"
echo "   • 🏆 Featured cards display"
echo "   • ⏱️ Response time tracking"
echo "   • 🧠 Enhanced AI responses"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    
    # Kill any remaining node processes
    pkill -f "react-scripts start" 2>/dev/null
    
    echo "👋 NAVUS stopped. Thanks for using the enhanced credit card advisor!"
    exit 0
}

# Handle Ctrl+C gracefully
trap cleanup INT

# Monitor processes and restart if they fail
monitor_processes() {
    while true do
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            echo "⚠️  Backend process died, attempting restart..."
            python backend.py &
            BACKEND_PID=$!
            sleep 2
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            echo "⚠️  Frontend process died, attempting restart..."
            cd frontend
            npm start &
            FRONTEND_PID=$!
            cd ..
            sleep 2
        fi
        
        sleep 10
    done
}

# Start process monitoring in background
monitor_processes &
MONITOR_PID=$!

# Wait indefinitely
wait