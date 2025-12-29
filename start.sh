#!/bin/bash

echo "=========================================="
echo "  🚀 Starting Adaptive Ad System"
echo "=========================================="
echo ""

# Get the directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Kill any existing processes on port 5000
echo "🧹 Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || true

# Start backend server
echo "🔧 Starting Backend Server..."
cd "$SCRIPT_DIR/backend"
python3 server.py > ../backend.log 2>&1 &
BACKEND_PID=$!

echo "   Backend PID: $BACKEND_PID"
echo "   Backend URL: http://localhost:5000"
echo ""

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend failed to start. Check backend.log for errors."
    exit 1
fi

echo ""
echo "=========================================="
echo "  ✨ System Started Successfully!"
echo "=========================================="
echo ""
echo "📋 Quick Start Guide:"
echo ""
echo "1. Open frontend/index.html in your browser"
echo "   File: $SCRIPT_DIR/frontend/index.html"
echo ""
echo "2. Upload product images using the upload button"
echo ""
echo "3. Click and hover on the ad canvas to interact"
echo ""
echo "4. Watch the AI adapt the layout in real-time!"
echo ""
echo "=========================================="
echo "  📊 System Information"
echo "=========================================="
echo ""
echo "Backend URL:  http://localhost:5000"
echo "Backend PID:  $BACKEND_PID"
echo "Frontend:     $SCRIPT_DIR/frontend/index.html"
echo "Logs:         $SCRIPT_DIR/backend.log"
echo ""
echo "To stop the system:"
echo "  kill $BACKEND_PID"
echo ""
echo "To view logs:"
echo "  tail -f $SCRIPT_DIR/backend.log"
echo ""
echo "=========================================="
echo ""

# Save PID to file for easy stopping
echo $BACKEND_PID > "$SCRIPT_DIR/backend.pid"

echo "💡 TIP: Keep this terminal open or the backend will stop!"
echo ""
echo "Press Ctrl+C to stop the system"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping system...'; kill $BACKEND_PID 2>/dev/null; rm -f '$SCRIPT_DIR/backend.pid'; echo '✅ System stopped.'; exit 0" INT

# Keep script running
wait $BACKEND_PID
