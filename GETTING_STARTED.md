# Getting Started with SAM3 Adaptive Ad System

## Quick Setup Guide

### Prerequisites Checklist

Before you begin, ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] GPU with CUDA support (recommended, but not required)
- [ ] 8GB+ RAM
- [ ] 10GB+ free disk space for model checkpoints
- [ ] HuggingFace account (free) for SAM 3 access

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sam3-adaptive-ads.git
cd sam3-adaptive-ads
```

#### 2. Make Scripts Executable

```bash
chmod +x *.sh
```

#### 3. Run Setup Script

```bash
./setup.sh
```

This will:
- Check Python installation
- Install all required Python packages from `requirements.txt`
- Create necessary directories
- Check for SAM 3 (optional but recommended)

#### 4. Install SAM 3 (Recommended)

SAM 3 provides the best product segmentation quality. Without it, the system will use fallback segmentation.

```bash
# Navigate to parent directory
cd ..

# Clone SAM 3 repository
git clone https://github.com/facebookresearch/sam3.git

# Authenticate with HuggingFace
pip install huggingface_hub
huggingface-cli login
# Enter your HuggingFace token when prompted
# Get token from: https://huggingface.co/settings/tokens

# Install SAM 3
cd sam3
pip install -e .

# Return to project
cd ../sam3-adaptive-ads
```

#### 5. Download Stable Diffusion 1.5

You'll need the SD 1.5 checkpoint file:

**Option A: Download from HuggingFace**
```bash
# Install huggingface-cli if not already
pip install huggingface_hub

# Download the model
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  v1-5-pruned-emaonly-fp16.safetensors \
  --local-dir ./models
```

**Option B: Use existing checkpoint**

If you already have SD 1.5 downloaded, update the path in `backend/server.py`:

```python
# Line 22-25 in backend/server.py
SD15_PATH = "/path/to/your/v1-5-pruned-emaonly-fp16.safetensors"
```

#### 6. Configure Environment (Optional)

Create a `.env` file for custom configuration:

```bash
# .env file
SD15_PATH=/path/to/v1-5-pruned-emaonly-fp16.safetensors
CLIP_CACHE_DIR=~/.cache/clip
CLIP_NAME=ViT-B/32
```

Or set environment variables:

```bash
export SD15_PATH="/path/to/v1-5-pruned-emaonly-fp16.safetensors"
export CLIP_CACHE_DIR="~/.cache/clip"
```

#### 7. Start the System

```bash
./start.sh
```

This will:
- Kill any existing processes on port 5000
- Start the Flask backend server
- Display system information and logs
- Save the backend PID for easy stopping

You should see output like:
```
========================================
  🚀 Starting Adaptive Ad System
========================================

🔧 Starting Backend Server...
   Backend PID: 12345
   Backend URL: http://localhost:5000

✅ Backend is running!

========================================
  ✨ System Started Successfully!
========================================

📋 Quick Start Guide:

1. Open frontend/index.html in your browser
2. Upload product images using the upload button
3. Click and hover on the ad canvas to interact
4. Watch the AI adapt the layout in real-time!
```

#### 8. Open the Frontend

Open `frontend/index.html` in your web browser:

```bash
# Linux/Mac
open frontend/index.html

# Windows
start frontend/index.html

# Or manually navigate to:
file:///path/to/sam3-adaptive-ads/frontend/index.html
```

#### 9. Verify Installation

Run the test suite:

```bash
./test.sh
```

This will check:
- Python installation
- Required packages
- File structure
- SAM 3 availability
- Port availability
- Backend imports
- Frontend HTML validity
- Script permissions

### First Steps After Installation

#### Upload Your First Product

1. Click **"Choose Product Image"** in the sidebar
2. Select a product image (e.g., a shoe, jacket, or any product)
3. Enter a **text prompt** describing the product:
   - Examples: "red sneaker", "leather jacket", "coffee mug", "smartphone"
4. Click **"Segment & Upload"**
5. Wait 1-3 seconds for processing
6. Product appears on the canvas!

#### Interact with the Canvas

- **Click** on products to show interest
- **Hover** over products to browse
- **Click on empty space** to trigger variations
- **Zoom** using the zoom controls (+/-)
- **Pan** by dragging on empty areas

#### Watch the AI Adapt

After **45 seconds of active interaction**, the system will:
1. Analyze your engagement patterns
2. Use CLIP to understand product context
3. Generate 3 unique AI backgrounds with SD 1.5
4. Composite products onto backgrounds
5. Position products based on your attention hotspots

#### View Analytics

Check the sidebar for:
- **Total Clicks**: Number of product interactions
- **Total Hovers**: Browsing behavior count
- **Products Loaded**: Number of products in library
- **Session ID**: Your unique session identifier
- **AI Analysis**: Reasoning for layout changes

### Troubleshooting

#### Backend Won't Start

**Problem:** Port 5000 already in use

**Solution:**
```bash
# Stop any existing process
./stop.sh

# Or manually kill
lsof -ti:5000 | xargs kill -9

# Then restart
./start.sh
```

**Problem:** Python packages missing

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

#### SAM 3 Not Working

**Problem:** SAM 3 import fails

**Check HuggingFace authentication:**
```bash
huggingface-cli whoami
```

**Re-authenticate if needed:**
```bash
huggingface-cli login
```

**Verify SAM 3 installation:**
```bash
cd ../sam3
pip install -e .
cd ../sam3-adaptive-ads
```

**Note:** System will work without SAM 3 using fallback segmentation.

#### CUDA Out of Memory

**Problem:** GPU runs out of memory

**Solution 1:** Use float32 instead of float16
```python
# Edit backend/server.py line 84
dtype = torch.float32  # More compatible, uses more VRAM
```

**Solution 2:** Run on CPU
```python
# Edit backend/server.py line 19
DEVICE = "cpu"  # Slower but no GPU needed
```

#### Frontend Not Connecting

**Problem:** Frontend can't reach backend

**Check backend is running:**
```bash
curl http://localhost:5000
# Should return JSON with status
```

**Check browser console:**
- Press F12 in browser
- Look for errors in Console tab
- Check Network tab for failed requests

**Verify API endpoint in frontend:**
```javascript
// frontend/index.html line 59
const API_BASE = 'http://localhost:5000/api';
```

#### Images Not Uploading

**Problem:** Upload fails or hangs

**Check folder permissions:**
```bash
ls -la backend/uploads
ls -la backend/processed
```

**Verify file size:**
- Max recommended: 5MB per image
- Supported formats: JPG, PNG

**Check backend logs:**
```bash
tail -f backend.log
```

### Stopping the System

**Option 1:** Use the stop script
```bash
./stop.sh
```

**Option 2:** Press Ctrl+C in the terminal running `start.sh`

**Option 3:** Manual kill
```bash
kill $(cat backend.pid)
```

### Next Steps

#### Learn the API

Read the [API documentation](README.md#api-endpoints) to integrate with your applications.

#### Customize the System

- **Modify analysis frequency**: Edit `frontend/index.html` line 86
- **Change background size**: Edit `backend/server.py` lines 35-36
- **Adjust AI prompts**: Edit `backend/server.py` lines 180-190
- **Customize UI colors**: Edit `frontend/index.html` CSS section

#### Deploy to Production

See [README.md Deployment section](README.md#deployment) for:
- Docker deployment
- Cloud platform guides (AWS, GCP, Azure)
- Production optimization tips

#### Contribute

Found a bug or want to add a feature?
- Open an issue on GitHub
- Submit a pull request
- Check the [Contributing Guide](README.md#contributing)

### Useful Commands

```bash
# Start system
./start.sh

# Stop system
./stop.sh

# Run tests
./test.sh

# View logs
tail -f backend.log

# Check if backend is running
curl http://localhost:5000

# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Reinstall dependencies
cd backend && pip install -r requirements.txt

# Clear processed images
rm -rf backend/processed/*
rm -rf backend/uploads/*
```

### Resources

- **README.md**: Comprehensive documentation
- **PRESENTATION.html**: Interactive slides (open in browser)
- **PRESENTATION.md**: Markdown slides (convert with Pandoc/Marp)
- **Backend logs**: `backend.log`
- **GitHub Issues**: Report bugs and get help

### Support

Need help?

1. **Check the logs**: `tail -f backend.log`
2. **Run diagnostics**: `./test.sh`
3. **Read the docs**: [README.md](README.md)
4. **Open an issue**: GitHub Issues page

---

**You're all set!** Start creating intelligent, adaptive product ads with AI.

Happy advertising! 🚀
