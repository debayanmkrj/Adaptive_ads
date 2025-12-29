#!/bin/bash

echo "=========================================="
echo "  Adaptive Ad System - Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Navigate to backend directory
cd backend

echo "📦 Installing Python dependencies..."
pip install --break-system-packages -r requirements.txt

echo ""
echo "🔧 Attempting to install SAM 3..."
echo "   (This requires HuggingFace authentication)"
echo ""

# Try to install SAM 3 if git repo is available
if [ -d "../../sam3" ]; then
    echo "   SAM 3 repo found, installing..."
    cd ../../sam3
    pip install --break-system-packages -e .
    cd ../adaptive-ad-system/backend
    echo "✅ SAM 3 installed!"
else
    echo "⚠️  SAM 3 repo not found in parent directory."
    echo "   To use SAM 3 segmentation:"
    echo "   1. Clone: git clone https://github.com/facebookresearch/sam3.git"
    echo "   2. Authenticate: huggingface-cli login"
    echo "   3. Install: cd sam3 && pip install -e ."
    echo ""
    echo "   The system will work with fallback segmentation for now."
fi

echo ""
echo "=========================================="
echo "  ✨ Setup Complete!"
echo "=========================================="
echo ""
echo "To start the system:"
echo "  1. Backend:  cd backend && python3 server.py"
echo "  2. Frontend: Open frontend/index.html in browser"
echo ""
echo "Backend will run on: http://localhost:5000"
echo "=========================================="
