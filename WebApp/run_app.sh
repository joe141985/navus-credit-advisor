#!/bin/bash
# NAVUS Web App Launcher

echo "🚀 Starting NAVUS Credit Card Advisor Web App"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "backend.py" ]; then
    echo "❌ Error: backend.py not found. Please run from the WebApp directory."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start backend in background
echo "🔥 Starting backend API server..."
python backend.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found"
    kill $BACKEND_PID
    exit 1
fi

# Install and start frontend
echo "🎨 Starting frontend development server..."
cd frontend

# Install Node dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Start frontend
echo "✅ Starting React development server..."
npm start &
FRONTEND_PID=$!

# Wait for user to stop
echo ""
echo "🎉 NAVUS is now running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"

# Function to cleanup on exit
cleanup() {
    echo "\n🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT

# Wait indefinitely
wait