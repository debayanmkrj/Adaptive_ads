#!/bin/bash

echo "🛑 Stopping Adaptive Ad System..."

# Get the directory where script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PID_FILE="$SCRIPT_DIR/backend.pid"

# Kill backend process
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "✅ Stopped backend (PID: $PID)"
    else
        echo "⚠️  Backend process not found"
    fi
    rm -f "$PID_FILE"
else
    echo "⚠️  PID file not found"
fi

# Kill any process on port 5000
lsof -ti:5000 | xargs kill -9 2>/dev/null && echo "✅ Killed processes on port 5000" || echo "ℹ️  No processes on port 5000"

echo "✅ System stopped"
