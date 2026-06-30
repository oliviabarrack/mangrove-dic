"""
Generate a sample image that mimics a speckle-painted mangrove branch
on a background, for testing the ROI selector without a real photo.
"""

import numpy as np
import cv2

rng = np.random.default_rng(0)

W, H = 1024, 768
img = np.full((H, W, 3), (30, 35, 28), dtype=np.uint8)  # dark green background

# --- Draw a rough branch shape (filled ellipse + rotated rect) ---
branch_mask = np.zeros((H, W), dtype=np.uint8)
cv2.ellipse(branch_mask, (512, 384), (340, 90), -18, 0, 360, 255, -1)
# taper the ends a bit
cv2.ellipse(branch_mask, (170, 290), (60, 40), -18, 0, 360, 0, -1)
cv2.ellipse(branch_mask, (855, 478), (60, 40), -18, 0, 360, 0, -1)

# Branch base colour
branch_bgr = np.array([110, 90, 70], dtype=np.float32)  # brownish wood tone
branch = img.copy()
branch[branch_mask > 0] = branch_bgr.astype(np.uint8)

# Add wood-grain texture
noise = rng.normal(0, 18, (H, W)).astype(np.float32)
noise = cv2.GaussianBlur(noise, (1, 31), 0)   # elongated along branch axis
for c in range(3):
    channel = branch[:, :, c].astype(np.float32)
    channel[branch_mask > 0] = np.clip(
        channel[branch_mask > 0] + noise[branch_mask > 0], 0, 255)
    branch[:, :, c] = channel.astype(np.uint8)

# --- Spray white speckle paint only on the branch ---
speckle_layer = np.zeros((H, W), dtype=np.float32)
n_speckles = 1200
for _ in range(n_speckles):
    # Random point inside the branch mask
    ys, xs = np.where(branch_mask > 0)
    idx = rng.integers(len(xs))
    cx, cy = int(xs[idx]), int(ys[idx])
    r = rng.uniform(2, 7)
    brightness = rng.uniform(120, 255)
    x0, x1 = max(0, int(cx - 3*r)), min(W, int(cx + 3*r) + 1)
    y0, y1 = max(0, int(cy - 3*r)), min(H, int(cy + 3*r) + 1)
    xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    blob = brightness * np.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * r**2))
    speckle_layer[y0:y1, x0:x1] = np.clip(
        speckle_layer[y0:y1, x0:x1] + blob, 0, 255)

speckle_layer = np.clip(speckle_layer, 0, 255)

# Composite: only apply speckle where branch exists
for c in range(3):
    ch = branch[:, :, c].astype(np.float32)
    alpha = np.where(branch_mask > 0, speckle_layer / 255.0, 0.0)
    speckle_rgb = speckle_layer * ([0.95, 0.95, 1.0][c])  # slight blue tint
    ch = np.clip(ch * (1 - alpha) + speckle_rgb * alpha, 0, 255)
    branch[:, :, c] = ch.astype(np.uint8)

# Add camera noise
cam_noise = rng.normal(0, 4, (H, W, 3)).astype(np.float32)
final = np.clip(branch.astype(np.float32) + cam_noise, 0, 255).astype(np.uint8)

out = "sample_branch.png"
cv2.imwrite(out, final)
print(f"Saved {out}  ({W}×{H} px)")
