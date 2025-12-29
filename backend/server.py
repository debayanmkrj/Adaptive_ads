import os
import io
import json
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch
from pathlib import Path
import hashlib
import time
from typing import Dict, Any, List, Tuple
import clip

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SD15 checkpoint path
SD15_PATH = os.environ.get(
    "SD15_PATH",
    "/mnt/c/Users/dbmkr/Documents/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors",
)

# CLIP cache dir
CLIP_DOWNLOAD_ROOT = os.environ.get(
    "CLIP_CACHE_DIR",
    "/mnt/c/Users/dbmkr/.cache/clip"
)
CLIP_NAME = os.environ.get("CLIP_NAME", "ViT-B/32")

# Image sizes
BG_WIDTH, BG_HEIGHT = 512, 512
MAX_PRODUCT_LONG = 400

# -----------------------------
# APP INIT
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = Path('uploads')
PROCESSED_FOLDER = Path('processed')
COMPOSITIONS_FOLDER = PROCESSED_FOLDER / "compositions"
for p in (UPLOAD_FOLDER, PROCESSED_FOLDER, COMPOSITIONS_FOLDER):
    p.mkdir(exist_ok=True)

# In-memory stores
engagement_data: Dict[str, Any] = {}
product_library: Dict[str, Any] = {}
composition_library: Dict[str, List[Dict[str, Any]]] = {}

# -----------------------------
# MODELS: SAM3 + CLIP + SD15
# -----------------------------
sam3_model = None
sam3_processor = None
clip_model = None
clip_preprocess = None
sd15_pipe = None

def init_sam3():
    """Initialize SAM 3 model."""
    global sam3_model, sam3_processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    sam3_model = build_sam3_image_model()
    sam3_processor = Sam3Processor(sam3_model)
    print(f"[SAM3] Loaded successfully on {DEVICE}")

def init_clip():
    """Load CLIP ViT-B/32."""
    global clip_model, clip_preprocess
    clip_model, clip_preprocess = clip.load(CLIP_NAME, device=DEVICE, download_root=CLIP_DOWNLOAD_ROOT)
    print(f"[CLIP] Loaded {CLIP_NAME}")

def init_sd15():
    """Load SD1.5 from safetensors."""
    global sd15_pipe
    from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    sd15_pipe = StableDiffusionPipeline.from_single_file(
        SD15_PATH,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    # Use EulerAncestralDiscreteScheduler to avoid PNDM scheduler bugs
    sd15_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd15_pipe.scheduler.config)
    sd15_pipe.to(DEVICE)
    if DEVICE == "cuda":
        try:
            sd15_pipe.enable_xformers_memory_efficient_attention()
        except:
            pass
    print(f"[SD1.5] Loaded from {SD15_PATH}")

def normalize_mask(mask):
    """Convert mask to 2D numpy array (H, W) uint8 {0,1}."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if isinstance(mask, (list, tuple)):
        mask = mask[0]
    
    mask = np.asarray(mask)
    
    if mask.ndim == 4:
        mask = mask[0]
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            mask = mask[0]
    
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    
    if mask.dtype != np.uint8 and mask.dtype != np.bool_:
        mask = (mask > 0.5)
    
    return (mask.astype(np.uint8) * 1)

# -----------------------------
# UTILS
# -----------------------------
def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    fmt = "PNG" if img.mode == "RGBA" else "JPEG"
    if fmt == "JPEG":
        img.save(buf, format=fmt, quality=92)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def save_image(img: Image.Image, path: Path):
    if img.mode == 'RGBA' and path.suffix.lower() == ".jpg":
        rgb = Image.new('RGB', img.size, (255, 255, 255))
        rgb.paste(img, mask=img.split()[3])
        rgb.save(path, 'JPEG', quality=92)
    else:
        img.save(path)

def mask_to_rgba(full_img: Image.Image, mask):
    """Convert full image + mask to RGBA."""
    mask_arr = normalize_mask(mask)
    mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
    if mask_img.size != full_img.size:
        mask_img = mask_img.resize(full_img.size, Image.NEAREST)
    
    rgba = full_img.convert("RGBA")
    rgba.putalpha(mask_img)
    return rgba

def crop_to_box(img: Image.Image, box):
    """Crop image to bounding box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    return img.crop((x1, y1, x2, y2))

def compute_avg_rgb(img: Image.Image) -> List[int]:
    """Compute average RGB color."""
    arr = np.array(img.convert("RGB"))
    return arr.mean(axis=(0,1)).astype(int).tolist()

def clip_image_features(img: Image.Image):
    """Get CLIP image features."""
    img_t = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = clip_model.encode_image(img_t)
    return features / features.norm(dim=-1, keepdim=True)

def clip_best_background_prompts(original_img: Image.Image, product_img: Image.Image, avg_rgb: Tuple[int,int,int], top_k: int = 3) -> List[str]:
    """Generate ABSTRACT background prompts based on CLIP color/mood analysis."""
    # Abstract scene categories - NO literal objects
    scene_categories = [
        "smooth gradient from dark to light",
        "soft bokeh circles and blur",
        "abstract geometric shapes and lines", 
        "flowing liquid paint texture",
        "smooth color wash background",
        "soft focus dreamy atmosphere",
        "minimal gradient backdrop",
        "abstract color blend"
    ]
    
    orig_feat = clip_image_features(original_img)
    
    text_tokens = torch.cat([clip.tokenize(c).to(DEVICE) for c in scene_categories])
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarities = (orig_feat @ text_features.T).squeeze(0)
    top_indices = similarities.topk(top_k).indices.cpu().tolist()
    
    # Get dominant color from original
    r, g, b = avg_rgb
    if r > g and r > b:
        color_desc = "warm red orange tones"
    elif b > r and b > g:
        color_desc = "cool blue tones"
    elif g > r and g > b:
        color_desc = "fresh green tones"
    else:
        color_desc = "neutral balanced tones"
    
    prompts = []
    for idx in top_indices:
        style = scene_categories[idx]
        prompts.append(f"abstract background, {style}, {color_desc}, smooth, clean, minimal")
    
    return prompts

def sd15_generate_background(prompt: str, seed: int = 42):
    """Generate background with SD1.5."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    result = sd15_pipe(
        prompt=prompt,
        negative_prompt="low quality, blurry, text, watermark, ugly",
        num_inference_steps=30,
        guidance_scale=7.5,
        width=BG_WIDTH,
        height=BG_HEIGHT,
        generator=generator
    )
    return result.images[0]

def compose_product_on_bg(product_rgba: Image.Image, bg: Image.Image, hotspot_xy=None):
    """Composite product onto background with improved blending."""
    bg = bg.resize((BG_WIDTH, BG_HEIGHT), Image.LANCZOS)
    
    pw, ph = product_rgba.size
    # Make product 80% of canvas for dominant presence
    target_size = int(min(BG_WIDTH, BG_HEIGHT) * 0.8)
    scale = min(target_size / max(pw, ph), 2.0)  # Allow upscaling up to 2x
    new_w, new_h = int(pw * scale), int(ph * scale)
    prod_scaled = product_rgba.resize((new_w, new_h), Image.LANCZOS)
    
    # Create soft shadow for depth
    shadow = Image.new('RGBA', (new_w + 20, new_h + 20), (0, 0, 0, 0))
    shadow_mask = Image.new('L', (new_w, new_h), 0)
    if prod_scaled.mode == 'RGBA':
        shadow_mask = prod_scaled.split()[3]
    shadow.paste((0, 0, 0, 80), (10, 10), shadow_mask)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
    
    if hotspot_xy:
        x_frac, y_frac = hotspot_xy
        paste_x = int(x_frac * BG_WIDTH - new_w / 2)
        paste_y = int(y_frac * BG_HEIGHT - new_h / 2)
    else:
        paste_x = (BG_WIDTH - new_w) // 2
        paste_y = (BG_HEIGHT - new_h) // 2
    
    paste_x = max(0, min(BG_WIDTH - new_w, paste_x))
    paste_y = max(0, min(BG_HEIGHT - new_h, paste_y))
    
    comp = bg.convert("RGBA")
    
    # Paste shadow first
    shadow_x = max(0, paste_x - 10)
    shadow_y = max(0, paste_y - 10)
    comp.paste(shadow, (shadow_x, shadow_y), shadow)
    
    # Then paste product
    comp.paste(prod_scaled, (paste_x, paste_y), prod_scaled)
    
    return comp.convert("RGB")

def heatmap_centroid(sess: Dict[str, Any]) -> Tuple[float, float]:
    """Calculate heatmap centroid from engagement data."""
    xs, ys = [], []
    for c in sess.get('clicks', []):
        xs.append(c.get('x', 0))
        ys.append(c.get('y', 0))
    for h in sess.get('hovers', []):
        xs.append(h.get('x', 0))
        ys.append(h.get('y', 0))
    
    if not xs:
        return (0.5, 0.5)
    
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    return (cx / 1000, cy / 600)

# -----------------------------
# SAM3 TEXT-BASED SEGMENTATION
# -----------------------------
def segment_with_sam3(image: Image.Image, text_prompt: str):
    """
    Segment product using SAM3 with text prompt.
    Returns: dict with masks, boxes, scores
    """
    inference_state = sam3_processor.set_image(image)
    output = sam3_processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    
    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    
    if len(masks) == 0:
        raise ValueError(f"No objects found for prompt: '{text_prompt}'")
    
    best_idx = int(scores.argmax())
    best_mask = masks[best_idx]
    best_box = boxes[best_idx]
    best_score = float(scores[best_idx])
    
    return {
        'mask': best_mask,
        'box': best_box.cpu().numpy().tolist(),
        'score': best_score
    }

# -----------------------------
# API ROUTES
# -----------------------------
@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_product():
    if request.method == 'OPTIONS':
        return '', 200
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    product_name = request.form.get('product_name', 'product')
    
    if not product_name or product_name.strip() == '':
        return jsonify({'error': 'Product name (text prompt) is required'}), 400
    
    raw_bytes = file.read()
    file_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
    
    filepath = UPLOAD_FOLDER / f"{file_hash}_{file.filename}"
    filepath.write_bytes(raw_bytes)
    
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    
    print(f"[UPLOAD] Segmenting with text: '{product_name}'")
    seg = segment_with_sam3(image, product_name)
    
    box = seg['box']
    mask = seg['mask']
    norm_mask = normalize_mask(mask)
    
    rgba_full = mask_to_rgba(image, norm_mask)
    cropped_rgb = crop_to_box(image, box)
    cropped_rgba = crop_to_box(rgba_full, box)
    
    (PROCESSED_FOLDER / f"{file_hash}_original.jpg").write_bytes(raw_bytes)
    save_image(cropped_rgb, PROCESSED_FOLDER / f"{file_hash}_cropped.jpg")
    save_image(cropped_rgba, PROCESSED_FOLDER / f"{file_hash}_cropped_rgba.png")
    
    prod_for_clip = cropped_rgba.convert("RGB")
    _ = clip_image_features(prod_for_clip)
    avg_rgb = compute_avg_rgb(prod_for_clip)
    
    views = {
        'original': {'url': f'/api/image/{file_hash}_original.jpg', 'base64': pil_to_b64(image)},
        'cropped':  {'url': f'/api/image/{file_hash}_cropped.jpg',  'base64': pil_to_b64(cropped_rgb)}
    }
    
    product_library[file_hash] = {
        'id': file_hash,
        'name': product_name,
        'views': views,
        'segmentation': {'box': box, 'score': seg['score']},
        'assets': {
            'original_path': str(PROCESSED_FOLDER / f"{file_hash}_original.jpg"),
            'cropped_rgb': str(PROCESSED_FOLDER / f"{file_hash}_cropped.jpg"),
            'cropped_rgba': str(PROCESSED_FOLDER / f"{file_hash}_cropped_rgba.png"),
            'avg_rgb': avg_rgb
        }
    }
    
    print(f"[UPLOAD] Success: {product_name} (score: {seg['score']:.3f})")
    return jsonify({'product_id': file_hash, 'product': product_library[file_hash]})

@app.route('/api/image/<filename>')
def serve_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify({'products': list(product_library.values())})

@app.route('/api/track', methods=['POST'])
def track_interaction():
    data = request.json or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'session_id required'}), 400
    
    sess = engagement_data.setdefault(session_id, {
        'clicks': [],
        'hovers': [],
        'active_ms': 0,
        'start_time': time.time(),
        'last_composition_ts': 0.0
    })
    
    etype = data.get('type')
    edata = data.get('data', {})
    
    if etype == 'click':
        sess['clicks'].append(edata)
    elif etype == 'hover':
        sess['hovers'].append(edata)
        sess['active_ms'] += max(0, int(edata.get('duration', 0)))
    elif etype == 'hover_canvas':
        sess['active_ms'] += max(0, int(edata.get('duration', 0)))
    
    return jsonify({'status': 'tracked'})

def pick_top_product_by_engagement(sess: Dict[str, Any]) -> str:
    counts: Dict[str,int] = {}
    for c in sess.get('clicks', []):
        pid = c.get('product_id')
        if pid:
            counts[pid] = counts.get(pid, 0) + 1
    if counts:
        return max(counts, key=counts.get)
    return next(iter(product_library.keys()), None)

# Generation lock to prevent simultaneous generations
generation_lock = {}

def generate_compositions_for_session(session_id: str, layout: List[Dict[str,Any]], num_variations:int=3) -> List[Dict[str,Any]]:
    # Prevent simultaneous generations for same session
    if session_id in generation_lock and generation_lock[session_id]:
        print(f"[COMPOSE] Already generating for session {session_id}, skipping")
        return []
    
    generation_lock[session_id] = True
    
    try:
        sess = engagement_data.get(session_id, {})
        pid = pick_top_product_by_engagement(sess)
        if not pid or pid not in product_library:
            generation_lock[session_id] = False
            return []
        
        prod = product_library[pid]
        prod_rgba = Image.open(prod['assets']['cropped_rgba']).convert("RGBA")
        prod_rgb  = Image.open(prod['assets']['cropped_rgb']).convert("RGB")
        orig_img = Image.open(prod['assets']['original_path']).convert("RGB")
        
        avg_rgb = tuple(prod['assets']['avg_rgb'])
        # Pass ORIGINAL image to CLIP to understand scene style
        prompts = clip_best_background_prompts(orig_img, prod_rgb, avg_rgb, top_k=num_variations)
        
        hotspot = heatmap_centroid(sess)
        
        out_items = []
        ts = int(time.time())
        for i, ptxt in enumerate(prompts):
            print(f"[COMPOSE] Generating for product {pid} - background {i+1}/{num_variations}: {ptxt}")
            bg = sd15_generate_background(ptxt, seed=ts + i)
            comp = compose_product_on_bg(prod_rgba, bg, hotspot_xy=hotspot)
            fname = f"comp_{session_id}_{pid}_{ts}_{i}.png"
            fpath = COMPOSITIONS_FOLDER / fname
            save_image(comp, fpath)
            out_items.append({
                'url': f'/api/image/compositions/{fname}',
                'base64': pil_to_b64(comp),
                'prompt': ptxt,
                'product_id': pid
            })
        
        composition_library.setdefault(session_id, []).extend(out_items)
        sess['last_composition_ts'] = time.time()
        return out_items
    finally:
        generation_lock[session_id] = False

@app.route('/api/image/compositions/<filename>')
def serve_composition(filename):
    return send_from_directory(COMPOSITIONS_FOLDER, filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_engagement():
    data = request.json or {}
    session_id = data.get('session_id')
    layout = data.get('layout', [])
    
    if not session_id:
        return jsonify({'action': 'continue', 'reasoning': ['session_id missing']})
    
    sess = engagement_data.setdefault(session_id, {
        'clicks': [], 'hovers': [], 'active_ms': 0,
        'start_time': time.time(), 'last_composition_ts': 0.0
    })
    
    reasoning = []
    reasoning.append(f"Active time: {sess['active_ms']} ms")
    
    should_generate = sess['active_ms'] >= 45000 and (time.time() - sess['last_composition_ts'] > 60.0)
    if should_generate:
        print(f"[ANALYZE] Triggering composition generation for session {session_id}")
        comps = generate_compositions_for_session(session_id, layout, num_variations=3)
        if comps:
            reasoning.append("Generated new SD1.5 backgrounds based on CLIP analysis + heatmap centroid")
            return jsonify({'action': 'new_compositions', 'compositions': comps, 'reasoning': reasoning})
    
    return jsonify({'action': 'continue', 'reasoning': reasoning})

@app.route('/api/heatmap', methods=['POST'])
def get_heatmap():
    data = request.json or {}
    session_id = data.get('session_id')
    sess = engagement_data.get(session_id)
    if not sess:
        return jsonify({'heatmap': []})
    
    heatmap_points = []
    for click in sess['clicks']:
        heatmap_points.append({'x': click.get('x',0),'y':click.get('y',0),'intensity':1.0,'type':'click'})
    for hover in sess['hovers']:
        heatmap_points.append({
            'x': hover.get('x',0),'y':hover.get('y',0),
            'intensity': min(hover.get('duration',0)/1000, 1.0),
            'type':'hover'
        })
    return jsonify({'heatmap': heatmap_points})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    total_sessions = len(engagement_data)
    total_clicks = sum(len(s['clicks']) for s in engagement_data.values())
    total_hovers = sum(len(s['hovers']) for s in engagement_data.values())
    total_active_ms = sum(int(s.get('active_ms',0)) for s in engagement_data.values())
    return jsonify({
        'total_sessions': total_sessions,
        'total_clicks': total_clicks,
        'total_hovers': total_hovers,
        'total_active_ms': total_active_ms,
        'total_products': len(product_library)
    })

@app.route('/')
def index():
    return jsonify({
        'status': 'Adaptive Ad System API',
        'sam3_loaded': sam3_model is not None,
        'clip_loaded': clip_model is not None,
        'sd15_loaded': sd15_pipe is not None,
        'endpoints': [
            '/api/upload','/api/products','/api/track','/api/analyze','/api/heatmap','/api/stats'
        ]
    })

if __name__ == '__main__':
    init_sam3()
    init_clip()
    init_sd15()
    
    print("\n" + "="*72)
    print("🚀 Adaptive Ad System Backend (SAM3 + CLIP + SD1.5)")
    print("="*72)
    print(f"Device:         {DEVICE}")
    print(f"SAM3:           ✅ loaded")
    print(f"CLIP:           ✅ {CLIP_NAME}")
    print(f"SD1.5:          ✅ {SD15_PATH}")
    print(f"Upload folder:  {UPLOAD_FOLDER.absolute()}")
    print(f"Output folder:  {PROCESSED_FOLDER.absolute()}")
    print("="*72 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)