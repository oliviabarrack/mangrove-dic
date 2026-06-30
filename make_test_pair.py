"""
Generate a photorealistic reference + deformed image pair for DIC testing.

Simulates a speckle-painted mangrove branch on a leafy background,
applies a small realistic strain, and writes an ROI JSON aligned to the
speckled region — ready to feed straight into run_dic.py.

Outputs:
    ref.png, def.png   — image pair
    pair_roi.json      — ROI polygon around the speckled patch
"""

import json
import cv2
import numpy as np

RNG = np.random.default_rng(42)

W, H   = 1280, 854    # typical mirrorless camera crop
ANGLE  = -22          # branch tilt (degrees)

# Applied strain (realistic for a loaded mangrove branch)
TRUE_EXX =  0.006   # 0.6 % tension along x
TRUE_EYY = -0.002   # slight Poisson contraction in y
TRUE_EXY =  0.001   # small shear
TX, TY   =  1.2, 0.4


# ---------------------------------------------------------------------------
# Background: blurred leafy texture
# ---------------------------------------------------------------------------
def make_background():
    bg = np.zeros((H, W, 3), dtype=np.float32)
    # Random green/brown blobs to simulate out-of-focus foliage
    for _ in range(400):
        cx = RNG.integers(0, W)
        cy = RNG.integers(0, H)
        r  = RNG.integers(20, 90)
        green  = RNG.uniform(40, 120)
        col = np.array([RNG.uniform(0, 40),
                        green,
                        RNG.uniform(0, 30)], dtype=np.float32)
        x0, x1 = max(0, cx-r), min(W, cx+r)
        y0, y1 = max(0, cy-r), min(H, cy+r)
        xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        blob   = np.exp(-((xg-cx)**2+(yg-cy)**2)/(2*(r*0.5)**2))
        for c in range(3):
            bg[y0:y1, x0:x1, c] = np.clip(
                bg[y0:y1, x0:x1, c] + col[c]*blob, 0, 255)
    bg = cv2.GaussianBlur(bg, (61, 61), 0)
    bg += RNG.normal(0, 6, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Branch mask: thick rotated ellipse
# ---------------------------------------------------------------------------
def make_branch_mask():
    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy = W // 2, H // 2
    cv2.ellipse(mask, (cx, cy), (480, 105), ANGLE, 0, 360, 255, -1)
    return mask


# ---------------------------------------------------------------------------
# Branch appearance: bark texture + cylindrical shading
# ---------------------------------------------------------------------------
def make_branch(mask):
    branch = np.zeros((H, W, 3), dtype=np.float32)

    # Base bark colour (medium grey-brown)
    base = np.array([68, 72, 80], dtype=np.float32)

    # Fine bark grain along the branch axis
    grain_x = RNG.normal(0, 1, (H, W)).astype(np.float32)
    grain_x = cv2.GaussianBlur(grain_x, (1, 21), 0) * 18

    branch[:, :, 0] = base[0] + grain_x
    branch[:, :, 1] = base[1] + grain_x * 0.9
    branch[:, :, 2] = base[2] + grain_x * 0.8

    # Cylindrical highlight: bright stripe along the long axis centre
    # Distance from branch centreline (perpendicular to branch axis)
    angle_rad = np.deg2rad(ANGLE)
    ys, xs = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2
    # Perpendicular distance from centre line
    perp = (-(xs - cx) * np.sin(angle_rad) + (ys - cy) * np.cos(angle_rad))
    branch_half_w = 105.0
    highlight = np.exp(-(perp ** 2) / (2 * (branch_half_w * 0.35) ** 2))
    shadow    = 1.0 - 0.55 * np.exp(-(perp ** 2) / (2 * (branch_half_w * 0.8) ** 2))

    for c in range(3):
        branch[:, :, c] = branch[:, :, c] * shadow + 90 * highlight

    branch = np.clip(branch, 0, 255).astype(np.uint8)
    return branch


# ---------------------------------------------------------------------------
# Speckle paint: white Gaussian blobs, only on the branch
# ---------------------------------------------------------------------------
def make_speckle(mask, density=1800):
    layer = np.zeros((H, W), dtype=np.float32)
    ys, xs = np.where(mask > 0)

    for _ in range(density):
        idx = RNG.integers(len(xs))
        cx, cy = int(xs[idx]), int(ys[idx])
        r = RNG.uniform(2.5, 6.5)
        brightness = RNG.uniform(160, 255)
        x0, x1 = max(0, int(cx-3*r)), min(W, int(cx+3*r)+1)
        y0, y1 = max(0, int(cy-3*r)), min(H, int(cy+3*r)+1)
        xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        blob = brightness * np.exp(-((xg-cx)**2+(yg-cy)**2)/(2*r**2))
        layer[y0:y1, x0:x1] = np.clip(layer[y0:y1, x0:x1]+blob, 0, 255)

    return np.clip(layer, 0, 255)


# ---------------------------------------------------------------------------
# Composite: background + branch + speckle
# ---------------------------------------------------------------------------
def composite(bg, branch, speckle, mask):
    out = bg.astype(np.float32).copy()

    # Place branch over background
    m3 = (mask[:, :, None] / 255.0)
    out = out * (1 - m3) + branch.astype(np.float32) * m3

    # Blend speckle as paint (only inside mask)
    alpha = (speckle / 255.0) * (mask / 255.0)
    alpha3 = alpha[:, :, None]
    white = np.full_like(out, 240.0)
    out = out * (1 - alpha3) + white * alpha3

    # Global camera noise
    out += RNG.normal(0, 3.5, out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Apply known deformation (same approach as dic_validation.py)
# ---------------------------------------------------------------------------
def apply_strain(image, exx, eyy, exy, tx, ty):
    h, w = image.shape[:2]
    F = np.array([[1+exx, exy/2], [exy/2, 1+eyy]], dtype=np.float64)
    F_inv = np.linalg.inv(F)
    t = np.array([tx, ty])

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    coords = np.stack([xg.ravel(), yg.ravel()], axis=1)
    ref_coords = (F_inv @ (coords - t).T).T
    map_x = ref_coords[:, 0].reshape(h, w).astype(np.float32)
    map_y = ref_coords[:, 1].reshape(h, w).astype(np.float32)

    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


# ---------------------------------------------------------------------------
# ROI: polygon that tightly wraps the speckled region of the branch
# ---------------------------------------------------------------------------
def make_roi(mask):
    """
    Shrink-wrap the branch mask to a tight 8-point polygon,
    suitable for use with run_dic.py.
    """
    # Erode mask slightly so subsets never touch the bare-branch edge
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    roi_mask = cv2.erode(mask, kernel)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour found in branch mask")

    cnt = max(contours, key=cv2.contourArea)
    # Approximate to ~8 vertices
    eps = 0.04 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True)
    pts = approx[:, 0, :].tolist()

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return {
        "mode":   "polygon",
        "points": pts,
        "bbox":   [min(xs), min(ys), max(xs), max(ys)],
        "image_path":  "ref.png",
        "image_size":  [W, H],
        "display_scale": 1.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Generating background...")
bg     = make_background()

print("Building branch...")
mask   = make_branch_mask()
branch = make_branch(mask)

print("Spraying speckle paint...")
speckle = make_speckle(mask)

print("Compositing reference image...")
ref = composite(bg, branch, speckle, mask)

print("Applying deformation...")
defm = apply_strain(ref, TRUE_EXX, TRUE_EYY, TRUE_EXY, TX, TY)

print("Writing images...")
cv2.imwrite("ref.png",  ref)
cv2.imwrite("def.png",  defm)

print("Computing ROI...")
roi = make_roi(mask)
with open("pair_roi.json", "w") as f:
    json.dump(roi, f, indent=2)

print(f"\nDone.")
print(f"  ref.png / def.png  — {W}×{H} px")
print(f"  pair_roi.json      — {len(roi['points'])} vertices, "
      f"bbox {roi['bbox']}")
print(f"\nApplied strain: ε_xx={TRUE_EXX:+.4f}  "
      f"ε_yy={TRUE_EYY:+.4f}  ε_xy={TRUE_EXY:+.4f}")
print(f"Translation:    tx={TX} px, ty={TY} px")
print(f"\nRun DIC with:")
print(f"  .venv/bin/python run_dic.py ref.png def.png "
      f"--roi pair_roi.json --out results/branch_test")
