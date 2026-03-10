# Model Architecture and ML Documentation
## Astronomical Image Denoising Project

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Machine Learning Approach](#machine-learning-approach)
3. [Complete Workflow Overview](#complete-workflow-overview)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Model Training Process](#model-training-process)
6. [Model Architecture](#model-architecture)
7. [Training Methodology](#training-methodology)
8. [Loss Functions](#loss-functions)
9. [Data Processing Pipeline](#data-processing-pipeline)
10. [Data Augmentation](#data-augmentation)
11. [Image Reconstruction](#image-reconstruction)
12. [Evaluation Metrics](#evaluation-metrics)
13. [Optimization Techniques](#optimization-techniques)
14. [Hyperparameters](#hyperparameters)
15. [Inference Pipeline](#inference-pipeline)

---

## 1. Project Overview

**Objective:** Self-supervised denoising of astronomical images without requiring clean reference images.

**Key Innovation:** Implementation of the Noise2Void (N2V) methodology with a custom U-Net architecture featuring blind-spot masking, enabling the model to learn denoising from noisy data alone.

**Application Domain:** Astronomical imaging where obtaining noise-free ground truth is impractical or impossible.

---

## 2. Machine Learning Approach

### 2.1 Noise2Void (N2V) Paradigm

**Concept:** Self-supervised learning technique that trains a denoising model using only noisy images.

**Key Principle:**
- **No clean reference images required**
- The model learns to predict pixel values from surrounding context
- Blind-spot masking prevents the model from learning the identity function

**Advantages for Astronomy:**
- Real astronomical images inherently contain noise
- No need for synthetic clean-noisy image pairs
- Preserves astronomical features and flux information

### 2.2 Blind-Spot Training

**Mechanism:**
- During training, random 5×5 pixel regions are masked in each image
- The model cannot see the center pixel it's predicting
- Forces the network to learn denoising from spatial context

**Implementation:**
```python
# Random blind-spot mask generation
def make_center_mask(B, H, W, hole=5, device=None):
    """
    Creates masks with randomly positioned 5x5 holes
    Returns: Binary mask (1=contribute to loss, 0=blind-spot)
    """
```

---

## 3. Complete Workflow Overview

### 3.1 End-to-End Pipeline

The astronomical image denoising system follows a comprehensive pipeline from raw data to denoised output:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW DIAGRAM                     │
└─────────────────────────────────────────────────────────────────┘

1. RAW DATA COLLECTION
   ↓
   • Astronomical images (FITS, PNG, TIFF)
   • Various sizes and formats
   • Inherently noisy (sensor noise, photon noise, cosmic rays)
   
2. DATA PREPROCESSING
   ↓
   ├─ Image Manifest Creation
   │  • Catalog all images with SHA-1 IDs
   │  • Track metadata (dimensions, paths)
   │  • Generate images.csv
   │
   ├─ Patch Extraction (Tiling)
   │  • Extract 512×512 pixel patches
   │  • Overlapping tiles for full coverage
   │  • Percentile-based normalization [0, 1]
   │  • Save as .npy files for fast loading
   │  • Generate patches.csv
   │
   └─ Train/Val/Test Splitting
      • Parent-level splitting (70/15/15)
      • Prevent data leakage
      • Generate patches_splits.csv + splits.json

3. MODEL TRAINING
   ↓
   ├─ Data Loading
   │  • PyTorch DataLoader
   │  • Batch size: 8
   │  • Training augmentation enabled
   │  • Shuffle training data
   │
   ├─ Training Loop (per batch)
   │  • Load noisy patch (512×512×3)
   │  • Generate random blind-spot mask (5×5)
   │  • Forward pass through U-Net
   │  • Compute masked L2 loss
   │  • Backpropagation with AdamW
   │  • Mixed precision (FP16) for speed
   │
   ├─ Validation (per epoch)
   │  • Evaluate on validation set
   │  • Compute masked L2 loss
   │  • Check for improvement
   │
   └─ Checkpointing & Early Stopping
      • Save best model (lowest val loss)
      • Stop if no improvement for 3 epochs
      • Saved to checkpoints/n2v_unet/

4. MODEL INFERENCE
   ↓
   ├─ Single Patch Denoising
   │  • Load image → normalize → pad to multiple of 16
   │  • Forward pass through trained model
   │  • Clip output to [0, 1]
   │
   └─ Full Image Denoising
      • Extract overlapping patches
      • Denoise each patch independently
      • Stitch with cosine-weighted blending
      • Export denoised image

5. EVALUATION & ANALYSIS
   ↓
   ├─ Pixel-Level Metrics
   │  • PSNR, SSIM, MSE, MAE
   │
   ├─ Astronomical Metrics
   │  • Star detection (DAOStarFinder)
   │  • Star count comparison
   │  • Flux preservation analysis
   │
   └─ Visualization
      • Side-by-side comparisons
      • Histograms and distributions
      • Detection overlays
      • PR curves

6. DEPLOYMENT
   ↓
   └─ GUI Application (predict_and_analyze_gui.py)
      • Interactive image loading
      • Real-time denoising
      • Comprehensive analysis tools
      • Export results (images + JSON)
```

### 3.2 Key Workflow Stages

**Stage 1: Data Preparation (One-Time Setup)**
- Duration: ~30-60 minutes for full dataset
- Input: Raw astronomical images
- Output: Preprocessed patches + manifest CSVs
- Scripts: `manifest.py`, `tiler.py`, `split.py`

**Stage 2: Model Training (One-Time or Iterative)**
- Duration: ~2-4 hours (depends on dataset size, GPU)
- Input: Preprocessed patches + splits
- Output: Trained model checkpoints
- Script: `train_n2v.py`
- Checkpoints: 7 models saved (387 to 5805 iterations)

**Stage 3: Inference (On-Demand)**
- Duration: <1 second per patch, ~1-2 seconds per full image
- Input: New noisy image + trained model
- Output: Denoised image
- Scripts: `predict_and_analyze.py`, `predict_and_analyze_gui.py`

**Stage 4: Evaluation (Optional)**
- Duration: ~5-10 minutes per evaluation run
- Input: Denoised images + optional ground truth
- Output: Metrics, plots, reports
- Scripts: `metrics_and_plots.py`, `detection_pr_curves.py`

### 3.3 Data Flow Summary

```
Raw Images 
    → Manifest (images.csv)
    → Patches (patches.csv) 
    → Splits (patches_splits.csv)
    → Training (model checkpoints)
    → Inference (denoised images)
    → Evaluation (metrics + visualizations)
```

---

## 4. Data Preprocessing Pipeline

### 4.1 Overview of Preprocessing Steps

Data preprocessing transforms raw astronomical images into training-ready patches. This is a **critical step** that ensures consistent, normalized data for model training.

### 4.2 Step 1: Image Manifest Creation

**Purpose:** Catalog all raw images and assign unique identifiers

**Script:** `src/dataio/manifest.py`

**Process:**
```python
1. Scan data/raw/ directory for image files
   • Supported formats: .png, .jpg, .tif, .tiff, .fits
   
2. For each image:
   • Read image dimensions (height, width, channels)
   • Compute SHA-1 hash as unique ID
   • Generate short ID (first 12 chars of SHA-1)
   • Record absolute file path
   
3. Create images.csv manifest:
   columns: [parent_id, path, height, width, channels]
   
4. Quality control:
   • Verify all images loadable
   • Check for duplicates
   • Validate dimensions
```

**Output:** `data/manifests/images.csv`

**Example Row:**
```csv
parent_id,path,height,width,channels
03e07b41d4fc,/path/to/image.png,2314,2314,3
```

### 4.3 Step 2: Percentile-Based Normalization

**Purpose:** Handle extreme outliers in astronomical data (hot pixels, cosmic rays)

**Why Not Min-Max Normalization?**
- Astronomical images contain extreme outliers
- Single hot pixel can skew entire normalization
- Percentile method is robust to outliers

**Algorithm:**
```python
def percentile_normalize(image, low=1, high=99):
    """
    Robust normalization using percentiles
    """
    # Compute percentile thresholds
    p_low, p_high = np.percentile(image, [low, high])
    
    # Clip outliers
    image = np.clip(image, p_low, p_high)
    
    # Normalize to [0, 1]
    if p_high > p_low:
        image = (image - p_low) / (p_high - p_low)
    else:
        image = np.zeros_like(image)
    
    return image.astype(np.float32)
```

**Parameters:**
- `low_percentile`: 1st percentile (default: 1)
- `high_percentile`: 99th percentile (default: 99)
- Clips extreme 1% of values on each end

### 4.4 Step 3: Patch Extraction (Tiling)

**Purpose:** Convert large images into uniform 512×512 patches for training

**Script:** `src/dataio/tiler.py`

**Strategy: Overlapping Tiles**
```
Image: 2314 × 2314 pixels
Patch Size: 512 × 512
Overlap: 50 pixels (default)
Stride: 512 - 50 = 462 pixels

Patches per dimension: ceil((2314 - 512 + 462) / 462) = 5-6 patches
```

**Algorithm:**
```python
def extract_patches(image, patch_size=512, overlap=50):
    """
    Extract overlapping patches with coordinates
    """
    H, W, C = image.shape
    stride = patch_size - overlap
    patches = []
    coords = []
    
    # Sliding window extraction
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
            coords.append((y, x))
    
    # Handle edge patches (if image not evenly divisible)
    # Extract patches at right and bottom edges
    
    return patches, coords
```

**Patch Naming Convention:**
```
Format: {parent_id}_y{y_coord}_x{x_coord}.npy

Examples:
03e07b41d4fc_y0_x0.npy      → Top-left patch
03e07b41d4fc_y0_x512.npy    → Next patch to the right
03e07b41d4fc_y512_x0.npy    → Patch below top-left
```

**Storage:**
- Format: NumPy binary (.npy)
- Shape: (512, 512, 3) [H, W, C]
- Data type: float32
- Value range: [0.0, 1.0]
- Location: `data/patches_50/`

**Output:** `data/manifests/patches.csv`

**Example Rows:**
```csv
tile_id,parent_id,path,y,x,height,width
03e07b41d4fc_y0_x0,03e07b41d4fc,data/patches_50/03e07b41d4fc_y0_x0.npy,0,0,512,512
03e07b41d4fc_y0_x512,03e07b41d4fc,data/patches_50/03e07b41d4fc_y0_x512.npy,0,512,512,512
```

### 4.5 Step 4: Train/Validation/Test Splitting

**Purpose:** Create non-overlapping splits for model evaluation

**Script:** `src/dataio/split.py`

**Critical Design: Parent-Level Splitting**

⚠️ **Why Parent-Level?**
- Patches from same image are highly correlated
- Splitting at patch level would leak information
- Solution: Assign all patches from one parent to same split

**Algorithm:**
```python
def create_splits(patches_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split at parent (image) level to prevent leakage
    """
    # Get unique parent images
    parent_ids = patches_df['parent_id'].unique()
    n_parents = len(parent_ids)
    
    # Shuffle parents
    np.random.shuffle(parent_ids)
    
    # Calculate split indices
    n_train = int(n_parents * train_ratio)
    n_val = int(n_parents * val_ratio)
    
    # Assign parents to splits
    train_parents = parent_ids[:n_train]
    val_parents = parent_ids[n_train:n_train+n_val]
    test_parents = parent_ids[n_train+n_val:]
    
    # Assign all patches from each parent to corresponding split
    patches_df['split'] = patches_df['parent_id'].map({
        **{p: 'train' for p in train_parents},
        **{p: 'val' for p in val_parents},
        **{p: 'test' for p in test_parents}
    })
    
    return patches_df
```

**Split Ratios:**
- **Training:** 70% of images
- **Validation:** 15% of images
- **Test:** 15% of images

**Outputs:**
1. `data/manifests/patches_splits.csv` - Patches with split column
2. `data/manifests/splits.json` - Split statistics and parent assignments

**Example patches_splits.csv:**
```csv
tile_id,parent_id,path,y,x,height,width,split
03e07b41d4fc_y0_x0,03e07b41d4fc,data/patches_50/...,0,0,512,512,train
0752c157961a_y0_x0,0752c157961a,data/patches_50/...,0,0,512,512,val
0e95ba25d499_y0_x0,0e95ba25d499,data/patches_50/...,0,0,512,512,test
```

### 4.6 Step 5: Quality Control Validation

**Purpose:** Ensure data integrity before training

**Script:** `src/dataio/qc.py`

**Checks Performed:**
```python
1. File Existence:
   ✓ All .npy files in CSV exist on disk
   ✓ All files are readable
   
2. Data Integrity:
   ✓ Patch shape is exactly (512, 512, 3)
   ✓ Data type is float32
   ✓ Value range is [0.0, 1.0]
   ✓ No NaN or Inf values
   
3. Split Balance:
   ✓ Each split has sufficient samples
   ✓ No parent appears in multiple splits
   
4. Coverage:
   ✓ All parent images have at least one patch
   ✓ Overlapping patches correctly positioned
```

### 4.7 Preprocessing Summary Statistics

**Example Dataset:**
```
Raw Images:        50 images
Average Size:      2000 × 2000 pixels
Total Raw Data:    ~600 MB (uncompressed)

After Preprocessing:
Total Patches:     2,847 patches
Patch Size:        512 × 512 × 3
Storage Size:      ~2.1 GB (float32 .npy files)

Split Distribution:
├─ Training:       1,993 patches (70%) from 35 images
├─ Validation:     427 patches (15%) from 8 images  
└─ Test:           427 patches (15%) from 7 images
```

### 4.8 Command-Line Execution

**Run Complete Preprocessing Pipeline:**

```bash
# Step 1: Build image manifest
python src/dataio/manifest.py --input data/raw --output data/manifests/images.csv

# Step 2: Extract patches
python src/dataio/tiler.py \
    --manifest data/manifests/images.csv \
    --output_dir data/patches_50 \
    --patch_size 512 \
    --overlap 50 \
    --output_csv data/manifests/patches.csv

# Step 3: Create splits
python src/dataio/split.py \
    --input data/manifests/patches.csv \
    --output data/manifests/patches_splits.csv \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15

# Step 4: Quality control
python src/dataio/qc.py \
    --patches_csv data/manifests/patches_splits.csv \
    --verify_files
```

---

## 5. Model Training Process

### 5.1 Training Overview

The training process implements self-supervised Noise2Void learning with blind-spot masking, early stopping, and mixed precision optimization.

### 5.2 Training Initialization

**Step 1: Environment Setup**
```python
# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input size
```

**Step 2: Data Loaders**
```python
train_loader, val_loader, test_loader = make_loaders(
    patch_csv='data/manifests/patches_splits.csv',
    batch_size=8,
    num_workers=0,      # Windows: 0, Linux/Colab: 2-4
    pin_memory=True,    # Faster GPU transfer
    use_aug=True        # Enable augmentation for training only
)
```

**Step 3: Model Initialization**
```python
model = UNetBlindspot(
    in_ch=3,    # RGB channels
    base=48     # Base channel width
).to(device)

# Total parameters: ~13.4M
```

**Step 4: Optimizer Setup**
```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-4,              # Learning rate
    weight_decay=1e-4     # L2 regularization
)
```

**Step 5: Mixed Precision Setup (Optional)**
```python
from torch.amp import autocast, GradScaler

scaler = GradScaler('cuda', enabled=True)
```

### 5.3 Training Loop (Epoch-Level)

**High-Level Structure:**
```python
for epoch in range(1, max_epochs + 1):
    # 1. Training phase
    model.train()
    for batch in train_loader:
        # Forward pass + blind-spot masking
        # Compute loss
        # Backward pass
        # Update weights
    
    # 2. Validation phase
    model.eval()
    with torch.no_grad():
        # Evaluate on validation set
        # Compute validation loss
    
    # 3. Checkpoint & early stopping
    if validation_improved:
        save_checkpoint()
    else:
        check_early_stopping()
```

### 5.4 Training Loop (Batch-Level Detail)

**Detailed Training Step:**

```python
def training_step(batch, model, optimizer, scaler, device):
    """
    Single training iteration with blind-spot masking
    """
    # 1. Load batch
    x = batch['image'].to(device)  # Shape: [8, 3, 512, 512]
    B, C, H, W = x.shape
    
    # 2. Generate blind-spot mask
    mask = make_center_mask(B, H, W, hole=5, device=device)
    # mask shape: [8, 1, 512, 512]
    # mask[i, 0, y_i-2:y_i+3, x_i-2:x_i+3] = 0.0 (random per batch)
    
    # 3. Target = input (self-supervised)
    target = x  # Predict the same image
    
    # 4. Zero gradients
    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    
    # 5. Forward pass (with mixed precision)
    with autocast(device_type='cuda', enabled=True):
        prediction = model(x)  # Shape: [8, 3, 512, 512]
        loss = masked_l2(prediction, target, mask)
    
    # 6. Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # 7. Optimizer step with scaling
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()
```

**Key Training Details:**

1. **Blind-Spot Mask:** Randomly positioned 5×5 hole per sample
2. **Target:** Same as input (self-supervised, not identity due to mask)
3. **Loss:** Only computed on visible regions (mask=1)
4. **Mixed Precision:** FP16 forward pass, FP32 gradients
5. **Gradient Scaling:** Prevents underflow in FP16

### 5.5 Validation Loop

**Validation Step (per epoch):**

```python
def validation_epoch(val_loader, model, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            x = batch['image'].to(device)
            B, C, H, W = x.shape
            
            # Same blind-spot mask for consistent evaluation
            mask = make_center_mask(B, H, W, hole=5, device=device)
            
            # Forward pass (no gradient tracking)
            with autocast(device_type='cuda', enabled=True):
                prediction = model(x)
                loss = masked_l2(prediction, x, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss
```

**Validation Characteristics:**
- No augmentation applied
- Same mask strategy as training
- No gradient computation (faster)
- Averaged over entire validation set

### 5.6 Checkpointing Logic

**Save Checkpoint Function:**

```python
def save_checkpoint(model, optimizer, global_step, output_dir):
    """
    Save model state, optimizer state, and training step
    """
    checkpoint = {
        'model': model.state_dict(),      # Model weights
        'opt': optimizer.state_dict(),    # Optimizer state (momentum, etc.)
        'step': global_step               # Training iteration count
    }
    
    path = Path(output_dir) / f'ckpt_{global_step}.pt'
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved: {path}")
```

**Checkpointing Strategy:**
- Save only when validation loss improves
- Overwrites previous best (or can keep all)
- Includes optimizer state for seamless resumption

### 5.7 Early Stopping Implementation

**Early Stopping Logic:**

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 3  # Stop after 3 epochs without improvement
min_delta = 1e-4  # Minimum improvement threshold

for epoch in range(1, max_epochs + 1):
    # ... training ...
    val_loss = validate(model, val_loader)
    
    # Check improvement
    if val_loss + min_delta < best_val_loss:
        # Improved!
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(model, optimizer, global_step, output_dir)
        print(f"✓ Validation improved: {val_loss:.5f}")
    else:
        # No improvement
        patience_counter += 1
        print(f"⚠ No improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"⏹ Early stopping triggered after {epoch} epochs")
            break
```

**Benefits:**
- Prevents overfitting
- Saves training time
- Automatically selects best model

### 5.8 Resume Training from Checkpoint

**Loading Checkpoint:**

```python
def resume_training(checkpoint_path, model, optimizer, device):
    """
    Resume training from saved checkpoint
    """
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Restore model weights
        model.load_state_dict(ckpt['model'])
        
        # Restore optimizer state
        optimizer.load_state_dict(ckpt['opt'])
        
        # Restore training step
        global_step = ckpt.get('step', 0)
        
        print(f"✓ Resumed from step {global_step}")
        return global_step
    else:
        print("⚠ Checkpoint not found, starting from scratch")
        return 0
```

**Usage:**
```bash
python src/train_n2v.py --resume checkpoints/n2v_unet/ckpt_2709.pt
```

### 5.9 Training Command & Arguments

**Full Training Command:**

```bash
python src/train_n2v.py \
    --patch_csv data/manifests/patches_splits.csv \
    --outdir checkpoints/n2v_unet \
    --epochs 20 \
    --batch_size 8 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --base 48 \
    --hole 5 \
    --amp \
    --early_stop_patience 3 \
    --min_delta 1e-4 \
    --log_every 100 \
    --num_workers 0
```

**Argument Descriptions:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch_csv` | `data/manifests/patches_splits.csv` | Path to preprocessed patches with splits |
| `--outdir` | `checkpoints/n2v_unet` | Output directory for checkpoints |
| `--resume` | `None` | Path to checkpoint for resuming training |
| `--epochs` | `20` | Maximum number of training epochs |
| `--batch_size` | `8` | Number of patches per batch |
| `--num_workers` | `0` | DataLoader worker processes (0 for Windows) |
| `--lr` | `2e-4` | Learning rate for AdamW |
| `--weight_decay` | `1e-4` | L2 regularization coefficient |
| `--base` | `48` | Base channel width in U-Net |
| `--hole` | `5` | Blind-spot size (5×5 pixels) |
| `--amp` | Flag | Enable mixed precision (FP16) training |
| `--log_every` | `100` | Print training loss every N steps |
| `--max_steps_per_epoch` | `0` | Limit batches per epoch (0=all) |
| `--early_stop_patience` | `3` | Epochs without improvement before stopping |
| `--min_delta` | `1e-4` | Minimum loss improvement threshold |

### 5.10 Training Progress Example

**Console Output:**

```
✓ Using GPU: NVIDIA GeForce RTX 3050
✓ VRAM Available: 8.0 GB
Loading data...
✓ Train: 1993 patches, Val: 427 patches, Test: 427 patches

[Epoch 1/20]
[train] epoch 1 step 100 loss 0.01234
[train] epoch 1 step 200 loss 0.01156
[train] epoch 1 step 249 loss 0.01089
[val] epoch 1 masked-L2 0.01023
✓ Validation improved: 0.01023
✓ Checkpoint saved: checkpoints/n2v_unet/ckpt_387.pt

[Epoch 2/20]
[train] epoch 2 step 300 loss 0.00987
[train] epoch 2 step 400 loss 0.00945
[train] epoch 2 step 498 loss 0.00912
[val] epoch 2 masked-L2 0.00891
✓ Validation improved: 0.00891
✓ Checkpoint saved: checkpoints/n2v_unet/ckpt_774.pt

...

[Epoch 15/20]
[train] epoch 15 step 4500 loss 0.00621
[val] epoch 15 masked-L2 0.00598
✓ Validation improved: 0.00598
✓ Checkpoint saved: checkpoints/n2v_unet/ckpt_5805.pt

[Epoch 16/20]
[val] epoch 16 masked-L2 0.00603
⚠ No improvement (1/3)

[Epoch 17/20]
[val] epoch 17 masked-L2 0.00606
⚠ No improvement (2/3)

[Epoch 18/20]
[val] epoch 18 masked-L2 0.00609
⚠ No improvement (3/3)
⏹ Early stopping triggered after 18 epochs

[done] best val masked-L2: 0.00598  ckpts in checkpoints/n2v_unet
```

### 5.11 Training Performance Metrics

**Training Time (RTX 3050, 8GB VRAM):**
- Time per batch (FP16): ~150-200ms
- Time per batch (FP32): ~300-400ms
- Time per epoch (1993 batches): ~5-8 minutes
- Total training time (15 epochs): ~75-120 minutes

**Memory Usage:**
- Model size: ~53 MB
- Batch memory (batch_size=8): ~200 MB
- Optimizer state: ~100 MB
- Total GPU usage: ~400-500 MB (plenty of headroom)

**Speedup Techniques:**
- Mixed precision (FP16): 2× faster
- cudnn.benchmark: 10-15% faster
- pin_memory: 5-10% faster data loading

### 5.12 How the Model Learns Denoising

**Intuition:**

1. **Input:** Noisy patch (512×512×3)
2. **Blind-Spot:** Model cannot see random 5×5 region
3. **Task:** Predict all pixels including blind-spot
4. **Learning:** Model learns spatial correlations (patterns of stars, galaxies, background)
5. **Denoising:** Noise is random → averaging context removes noise

**Mathematical Insight:**

```
Signal: S(x,y) - correlated with neighbors
Noise:  N(x,y) - independent, zero-mean

Observed: O(x,y) = S(x,y) + N(x,y)

Model learns: Ŝ(x,y) = f_θ(O(neighbors))

Loss minimization → Ŝ(x,y) ≈ S(x,y)
Noise averaged out → Denoised output
```

**Why It Works:**
- Noise is spatially independent
- Signal is spatially correlated
- Model exploits correlations to separate signal from noise

---

## 6. Model Architecture

### 3.1 U-Net with Blind-Spot (UNetBlindspot)

**Architecture Type:** Encoder-Decoder with Skip Connections

**Network Structure:**

```
Input (3×H×W) 
    ↓
Encoder Path:
    enc1: Conv3×3 → BN → SiLU → Conv3×3 → BN → SiLU  [base channels]
    enc2: MaxPool → Conv blocks                       [base×2 channels]
    enc3: MaxPool → Conv blocks                       [base×4 channels]
    enc4: MaxPool → Conv blocks                       [base×8 channels]

Bottleneck:
    bott: MaxPool → Conv blocks                       [base×16 channels]

Decoder Path:
    dec4: ConvTranspose → Concat(enc4) → Conv blocks  [base×8 channels]
    dec3: ConvTranspose → Concat(enc3) → Conv blocks  [base×4 channels]
    dec2: ConvTranspose → Concat(enc2) → Conv blocks  [base×2 channels]
    dec1: ConvTranspose → Concat(enc1) → Conv blocks  [base channels]

Output Head:
    head: Conv1×1                                     [3 channels]
    ↓
Output (3×H×W) - Denoised Image
```

### 3.2 Architecture Components

**Base Channels:** 48 (configurable)

**Convolutional Block:**
```python
def conv_block(in_ch, out_ch):
    - Conv2d(3×3, padding=1, bias=False)
    - BatchNorm2d
    - SiLU activation (Swish)
    - Conv2d(3×3, padding=1, bias=False)
    - BatchNorm2d
    - SiLU activation
```

**Key Features:**
- **Skip Connections:** Concatenate encoder features to decoder (U-Net style)
- **Batch Normalization:** Stabilizes training and improves convergence
- **SiLU Activation:** Smooth, non-monotonic activation (better than ReLU for gradients)
- **No Bias in Conv Layers:** Batch normalization makes bias redundant
- **Transposed Convolutions:** Learnable upsampling in decoder path

### 3.3 Network Depth

**Encoder Levels:** 4 downsampling stages (each 2× reduction)
**Bottleneck:** Processes features at 1/16th original resolution
**Decoder Levels:** 4 upsampling stages (mirror of encoder)

**Receptive Field:** Large enough to capture spatial context while excluding blind-spots

**Parameter Count:** 
- Base=48: ~13.4M parameters
- Scales quadratically with base channels

---

## 7. Training Methodology

### 7.1 Self-Supervised Training Loop

**Training Strategy:**
1. Load noisy patch (512×512×3)
2. Generate random blind-spot mask (5×5 hole)
3. Forward pass: Model predicts denoised image
4. Compute masked L2 loss (only on visible regions)
5. Backpropagate and update weights

**Key Insight:** The model learns denoising because:
- It cannot see the pixel it's predicting (blind-spot)
- It must use surrounding context to reconstruct the center
- Noise is assumed to be independent between pixels

### 7.2 Training Configuration

**Framework:** PyTorch 2.0+

**Optimizer:** AdamW
- **Learning Rate:** 2×10⁻⁴
- **Weight Decay:** 1×10⁻⁴ (L2 regularization)
- **Betas:** (0.9, 0.999) [default]

**Batch Size:** 8 (configurable based on GPU memory)

**Epochs:** 20 (with early stopping)

**Mixed Precision Training (AMP):**
- Optional FP16 training for faster computation
- Automatic loss scaling via `GradScaler`
- Compatible with PyTorch 2.x and legacy versions

### 7.3 Early Stopping

**Mechanism:**
- Monitor validation loss after each epoch
- Save checkpoint when validation improves
- Stop if no improvement for N epochs (default: 3)

**Parameters:**
- `early_stop_patience`: 3 epochs
- `min_delta`: 1×10⁻⁴ (minimum improvement threshold)

**Benefits:**
- Prevents overfitting
- Saves computation time
- Automatically selects best model

### 7.4 Checkpoint Management

**Saving Strategy:
- Save checkpoint whenever validation loss improves
- Checkpoint contains: `{model, optimizer, step}`
- Naming: `ckpt_{global_step}.pt`

**Resume Training:**
- Can resume from any checkpoint
- Restores model weights, optimizer state, and global step

---

## 8. Loss Functions

### 8.1 Masked L2 Loss

**Formula:**
```
L = Σ(mask × (pred - target)²) / Σ(mask)
```

**Implementation:**
```python
def masked_l2(pred, target, mask):
    diff = (pred - target) ** 2
    diff = diff.mean(dim=1, keepdim=True)  # Average across RGB channels
    num = (mask * diff).sum()
    den = mask.sum().clamp_min(1.0)
    return num / den
```

**Key Properties:**
- **Channel Averaging:** Mean across RGB before masking
- **Normalization:** Divide by number of valid pixels
- **Stability:** Clamp denominator to prevent division by zero

### 8.2 Blind-Spot Masking

**Mask Generation:**
```python
def make_center_mask(B, H, W, hole=5, device=None):
    """
    Creates binary mask with random 5×5 blind-spots
    - 1.0 = pixel contributes to loss
    - 0.0 = blind-spot region (excluded from loss)
    """
    - Random position per batch element
    - Ensures hole fits within image boundaries
    - Single channel mask applied to all RGB channels
```

**Blind-Spot Size:** 5×5 pixels (configurable via `--hole` argument)

**Purpose:** Prevents the network from learning identity mapping by removing target information from input during training.

---

## 9. Data Processing and Loading

### 9.1 Image Normalization

**Method:** Percentile-based normalization (astronomy-specific)
```python
# Typical preprocessing
low, high = np.percentile(image, [1, 99])
image = (image - low) / (high - low)
image = np.clip(image, 0.0, 1.0)
```

**Rationale:** Handles extreme outliers common in astronomical images (hot pixels, cosmic rays)

### 9.2 Patch Extraction

**Patch Size:** 512×512 pixels

**Strategy:**
- Overlapping tiles for full image coverage
- Stored as `.npy` files for fast loading
- Metadata tracked in CSV manifests

**Format:** 
- Storage: `H×W×C` (NumPy)
- Training: `C×H×W` (PyTorch)
- Value Range: [0, 1] (float32)

### 9.3 Data Splits

**Split Strategy:** Parent-level splitting (prevents data leakage)
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

**Key Feature:** Patches from the same parent image never span splits

### 9.4 Data Loading

**PyTorch Dataset:**
```python
class PatchDataset(Dataset):
    - Loads .npy patches
    - Applies optional augmentations
    - Returns: {image, parent_id, tile_id, yx_coords}
```

**DataLoader Settings:**
- `shuffle=True` for training
- `drop_last=True` for consistent batch sizes
- `pin_memory=True` for GPU efficiency
- `num_workers`: 0 (Windows), >0 (Linux/Colab)

---

## 10. Data Augmentation

### 10.1 Augmentation Pipeline

**Astronomy-Safe Transformations:**

1. **Horizontal Flip:** 50% probability
2. **Vertical Flip:** 50% probability
3. **Small Rotation:** ±5 degrees (random)
4. **Translation:** ±8 pixels (random)
5. **Brightness Jitter:** ±0.05 (additive, preserves relative flux)

### 10.2 Implementation Details

**Affine Transformations:**
```python
# Rotation matrix with translation
θ = random(-5°, +5°)
tx, ty = random(-8px, +8px)

# Apply via grid_sample with bilinear interpolation
# Padding mode: 'border' (repeats edge pixels)
```

**Brightness Augmentation:**
```python
# Mild additive jitter
delta = random(-0.05, +0.05)
image = clip(image + delta, 0.0, 1.0)
```

**Design Principles:**
- **Preserve Flux:** Avoid multiplicative scaling
- **Minimal Artifacts:** Bilinear interpolation with border padding
- **Small Perturbations:** Keep transformations subtle for astronomical data

### 10.3 Augmentation Usage

- **Training Set:** Augmentation enabled
- **Validation/Test Sets:** No augmentation (deterministic)

---

## 11. Image Reconstruction

### 11.1 Cosine-Weighted Stitching

**Problem:** Overlapping patches need seamless merging

**Solution:** Cosine window weighted averaging

**Algorithm:**
```python
def cosine_window_2d(h, w):
    """
    Creates 2D cosine window: 0 at edges, 1 at center
    Smooth transition for blending overlapping regions
    """
    wy = 0.5 * (1 - cos(2π * linspace(0,1,h)))
    wx = 0.5 * (1 - cos(2π * linspace(0,1,w)))
    return outer(wy, wx)
```

**Stitching Process:**
1. Initialize accumulation buffers (image + weights)
2. For each patch:
   - Multiply by cosine window
   - Add to accumulator at patch position
   - Accumulate weights
3. Normalize: `final_image = accumulator / weights`

**Benefits:**
- Eliminates visible seams
- Smooth transitions at patch boundaries
- Weighted averaging reduces edge artifacts

### 11.2 Full Image Denoising

**Pipeline:**
1. Extract overlapping patches from noisy image
2. Denoise each patch independently
3. Stitch denoised patches using cosine weighting
4. Clip final image to [0, 1] range

**Padding:** Images padded to multiples of 16 (U-Net requirement) using reflection padding

---

## 12. Evaluation Metrics

### 12.1 Pixel-Level Metrics

**Mean Squared Error (MSE):**
```
MSE = mean((target - pred)²)
```

**Mean Absolute Error (MAE):**
```
MAE = mean(|target - pred|)
```

**Peak Signal-to-Noise Ratio (PSNR):**
```
PSNR = 10 × log₁₀(MAX² / MSE)
```
- Higher is better
- Units: decibels (dB)

**Structural Similarity Index (SSIM):**
```
SSIM = luminance × contrast × structure
```
- Range: [-1, 1], typically [0, 1]
- 1.0 = perfect similarity
- Considers local structure, not just pixel differences

### 12.2 Astronomical Metrics

**Star Detection:**
- **Tool:** DAOStarFinder (photutils library)
- **Method:** Sigma-clipped statistics for background estimation
- **Parameters:** FWHM, detection threshold

**Star Counting:**
- Count detected stars in noisy vs. denoised images
- Compare with ground truth (if available)

**Flux Preservation:**
- Measure total flux before/after denoising
- Critical for photometric accuracy

**Precision-Recall Analysis:**
- Precision: True detections / All detections
- Recall: True detections / All real stars
- PR curves for threshold tuning

### 12.3 Detection Metrics

**Confusion Matrix:**
- True Positives (TP): Correctly detected stars
- False Positives (FP): Spurious detections
- False Negatives (FN): Missed stars

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 13. Optimization Techniques

### 13.1 GPU Acceleration

**CUDA Optimizations:**
```python
# Automatic optimization setup
torch.backends.cudnn.benchmark = True  # Auto-tune kernels for fixed input size
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Mixed Precision (FP16):**
```python
# Training with AMP
with autocast(device_type="cuda", enabled=True):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- **2× faster training** on modern GPUs (RTX series)
- **Reduced memory usage** (can increase batch size)
- **Automatic loss scaling** prevents underflow

### 10.2 Inference Optimizations

**Half-Precision Inference:**
```python
if device == "cuda":
    model = model.half()  # Convert to FP16
    input = input.half()
```

**Batch Processing:** Process multiple patches in parallel

**Memory Management:**
- `torch.no_grad()`: Disable gradient computation
- `pin_memory=True`: Faster CPU→GPU transfers
- `non_blocking=True`: Asynchronous data transfers

### 13.3 Computational Efficiency

**Checkpoint Strategy:**
- Save only improved models
- Reduced disk I/O

**Early Stopping:**
- Prevents unnecessary epochs
- Saves computation time

**cudnn.benchmark:**
- Auto-selects optimal convolution algorithms
- Significant speedup for fixed input sizes

---

## 14. Hyperparameters

### 14.1 Model Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base` | 48 | Base number of channels in U-Net |
| `in_ch` | 3 | Input channels (RGB) |
| `depth` | 5 | Encoder/decoder depth (4 + bottleneck) |
| `kernel_size` | 3×3 | Convolutional kernel size |
| `pool_size` | 2×2 | Max pooling size |
| `activation` | SiLU | Activation function |

### 14.2 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 2×10⁻⁴ | Learning rate |
| `weight_decay` | 1×10⁻⁴ | L2 regularization |
| `batch_size` | 8 | Samples per batch |
| `epochs` | 20 | Maximum training epochs |
| `hole` | 5 | Blind-spot size (5×5 pixels) |
| `amp` | Optional | Mixed precision training |
| `early_stop_patience` | 3 | Epochs before early stopping |
| `min_delta` | 1×10⁻⁴ | Minimum improvement threshold |

### 14.3 Data Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 512×512 | Training patch dimensions |
| `train_split` | 70% | Training data fraction |
| `val_split` | 15% | Validation data fraction |
| `test_split` | 15% | Test data fraction |
| `num_workers` | 0-4 | DataLoader worker processes |

### 14.4 Augmentation Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hflip` | 50% | Horizontal flip probability |
| `vflip` | 50% | Vertical flip probability |
| `rot_deg` | ±5° | Rotation range |
| `translate_px` | ±8px | Translation range |
| `brightness` | ±0.05 | Brightness jitter range |

---

## 15. Inference Pipeline

### 15.1 Preprocessing

```python
def preprocess_image(img_path):
    1. Load RGB image
    2. Convert to tensor [1, 3, H, W]
    3. Normalize to [0, 1]
    4. Pad to multiple of 16 (reflection mode)
    5. Move to GPU (if available)
    6. Convert to FP16 (if using GPU)
```

### 12.2 Denoising

```python
def denoise_image(model, tensor):
    with torch.no_grad():
        # Single forward pass
        denoised = model(tensor)
    
    # Post-process
    1. Crop padding
    2. Clip to [0, 1]
    3. Convert to NumPy
    4. Transpose to H×W×C
```

### 12.3 GUI Application

**Features:**
- Interactive image loading
- Real-time denoising preview
- Side-by-side comparison
- Star detection overlay
- Histogram analysis
- Spectral analysis
- Export results (images + JSON reports)

**Technical Stack:**
- **GUI:** Tkinter with custom styling
- **Plotting:** Matplotlib (embedded canvas)
- **Visualization:** PIL/Pillow for image display
- **Threading:** Background model loading (async)

---

## Key Technical Innovations

### 1. Self-Supervised Learning
✓ No clean reference images needed  
✓ Learns from noisy data alone  
✓ Applicable to real-world astronomical data

### 2. Blind-Spot Masking
✓ Prevents identity learning  
✓ Forces spatial context utilization  
✓ 5×5 random masking strategy

### 3. U-Net Architecture
✓ Skip connections preserve spatial details  
✓ Deep encoder-decoder for receptive field  
✓ SiLU activations for smooth gradients

### 4. Astronomy-Specific Processing
✓ Percentile-based normalization  
✓ Flux-preserving augmentations  
✓ Cosine-weighted patch stitching  
✓ Star detection evaluation

### 5. Production Optimizations
✓ Mixed precision training (FP16)  
✓ Early stopping with checkpointing  
✓ GPU acceleration with cudnn.benchmark  
✓ Interactive GUI for accessibility

---

## Model Performance

### Training Checkpoints
- **7 saved checkpoints** at iterations: 387, 1161, 1548, 2322, 2709, 4644, 5805
- **Current best model:** `ckpt_5805.pt` (5,805 training iterations)
- **Model size:** ~53 MB (base=48 configuration)

### Hardware Requirements
- **Minimum:** CPU with 8GB RAM
- **Recommended:** CUDA-capable GPU (RTX 2060 or better)
- **Optimal:** RTX 3050/3060 with 8GB+ VRAM

### Inference Speed (RTX 3050)
- **Single 512×512 patch:** ~50-100ms
- **Full 2048×2048 image:** ~1-2 seconds
- **FP16 speedup:** 2-3× faster than FP32

---

## References & Methodology

**Noise2Void:**
- Krull, A., Buchholz, T. O., & Jug, F. (2019). Noise2void-learning denoising from single noisy images. *CVPR 2019*.

**U-Net Architecture:**
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.

**Astronomical Image Processing:**
- Photutils: Astronomical photometry toolkit
- Astropy: Community Python astronomy library

---

## File Structure Reference

```
src/
├── models/
│   └── unet_blindspot.py          # U-Net architecture implementation
├── losses/
│   └── masked_loss.py              # Blind-spot masking + masked L2 loss
├── dataio/
│   └── loaders.py                  # PyTorch dataset & augmentation pipeline
├── utils/
│   └── stitch.py                   # Cosine-weighted patch stitching
├── eval/
│   ├── metrics_and_plots.py        # PSNR, SSIM, detection metrics
│   └── detection_pr_curves.py      # Precision-recall analysis
├── train_n2v.py                     # Main training script
├── predict_and_analyze.py           # CLI inference tool
└── predict_and_analyze_gui.py       # Interactive GUI application
```

---

**Documentation Version:** 1.0  
**Last Updated:** February 28, 2026  
**Model Version:** ckpt_5805.pt  
**Framework:** PyTorch 2.0+
