#!/bin/bash

echo "=========================================="
echo "  🧪 System Test - Adaptive Ad System"
echo "=========================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PASSED=0
FAILED=0

# Test 1: Check Python
echo "[TEST 1] Checking Python installation..."
if command -v python3 &> /dev/null; then
    VERSION=$(python3 --version)
    echo "  ✅ PASS - $VERSION found"
    ((PASSED++))
else
    echo "  ❌ FAIL - Python 3 not found"
    ((FAILED++))
fi

# Test 2: Check pip packages
echo ""
echo "[TEST 2] Checking required Python packages..."
cd "$SCRIPT_DIR/backend"
MISSING_PACKAGES=0
for package in flask flask-cors Pillow numpy torch; do
    if python3 -c "import $(echo $package | tr '-' '_')" 2>/dev/null; then
        echo "  ✅ $package installed"
    else
        echo "  ❌ $package missing"
        ((MISSING_PACKAGES++))
    fi
done

if [ $MISSING_PACKAGES -eq 0 ]; then
    echo "  ✅ PASS - All packages installed"
    ((PASSED++))
else
    echo "  ❌ FAIL - $MISSING_PACKAGES packages missing"
    ((FAILED++))
fi

# Test 3: Check file structure
echo ""
echo "[TEST 3] Checking file structure..."
ALL_FILES_EXIST=true
FILES=(
    "backend/server.py"
    "backend/requirements.txt"
    "frontend/index.html"
    "setup.sh"
    "start.sh"
    "stop.sh"
)

for file in "${FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$file" ]; then
        echo "  ✅ $file exists"
    else
        echo "  ❌ $file missing"
        ALL_FILES_EXIST=false
    fi
done

if $ALL_FILES_EXIST; then
    echo "  ✅ PASS - All files present"
    ((PASSED++))
else
    echo "  ❌ FAIL - Some files missing"
    ((FAILED++))
fi

# Test 4: Check SAM 3 availability
echo ""
echo "[TEST 4] Checking SAM 3 installation..."
if python3 -c "from sam3.model_builder import build_sam3_image_model" 2>/dev/null; then
    echo "  ✅ PASS - SAM 3 installed and importable"
    ((PASSED++))
else
    echo "  ⚠️  WARN - SAM 3 not installed (will use fallback)"
    echo "           This is OK - system will work without SAM 3"
    ((PASSED++))
fi

# Test 5: Check port availability
echo ""
echo "[TEST 5] Checking if port 5000 is available..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "  ⚠️  WARN - Port 5000 already in use"
    echo "           Run ./stop.sh to free the port"
else
    echo "  ✅ PASS - Port 5000 is available"
    ((PASSED++))
fi

# Test 6: Test backend import
echo ""
echo "[TEST 6] Testing backend server import..."
cd "$SCRIPT_DIR/backend"
if python3 -c "import server" 2>/dev/null; then
    echo "  ✅ PASS - Backend imports successfully"
    ((PASSED++))
else
    echo "  ❌ FAIL - Backend import error"
    echo "           Check requirements installation"
    ((FAILED++))
fi

# Test 7: Check frontend HTML validity
echo ""
echo "[TEST 7] Checking frontend HTML..."
if grep -q "Adaptive Ad System" "$SCRIPT_DIR/frontend/index.html"; then
    if grep -q "React.useState" "$SCRIPT_DIR/frontend/index.html"; then
        echo "  ✅ PASS - Frontend HTML looks valid"
        ((PASSED++))
    else
        echo "  ❌ FAIL - React code missing from HTML"
        ((FAILED++))
    fi
else
    echo "  ❌ FAIL - Frontend HTML appears corrupted"
    ((FAILED++))
fi

# Test 8: Check script permissions
echo ""
echo "[TEST 8] Checking script permissions..."
ALL_EXECUTABLE=true
SCRIPTS=("setup.sh" "start.sh" "stop.sh" "download-demo-images.sh")

for script in "${SCRIPTS[@]}"; do
    if [ -x "$SCRIPT_DIR/$script" ]; then
        echo "  ✅ $script is executable"
    else
        echo "  ❌ $script is not executable"
        ALL_EXECUTABLE=false
    fi
done

if $ALL_EXECUTABLE; then
    echo "  ✅ PASS - All scripts are executable"
    ((PASSED++))
else
    echo "  ⚠️  WARN - Some scripts not executable"
    echo "           Run: chmod +x *.sh"
    ((PASSED++))
fi

# Summary
echo ""
echo "=========================================="
echo "  📊 Test Summary"
echo "=========================================="
echo ""
echo "  ✅ Passed: $PASSED"
echo "  ❌ Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "  🎉 ALL TESTS PASSED!"
    echo ""
    echo "  System is ready to use. Run:"
    echo "    ./start.sh"
    echo ""
    EXIT_CODE=0
else
    echo "  ⚠️  SOME TESTS FAILED"
    echo ""
    echo "  Please fix the issues above before running."
    echo "  Try running ./setup.sh first."
    echo ""
    EXIT_CODE=1
fi

echo "=========================================="
exit $EXIT_CODE
