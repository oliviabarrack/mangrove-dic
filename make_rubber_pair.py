"""
Generate a synthetic rubber-band tensile test image pair.

Mimics the real setup: grey desk, beige rubber strip, red tape grips,
fine dense black speckle on the gauge section.  A known strain is applied
to produce the deformed image.

Usage:
    python make_rubber_pair.py [--strain FLOAT] [--seed INT] [--out PREFIX]

Outputs:
    <out>_ref.png, <out>_def.png   — image pair (2000x1500 px)
    <out>_roi.json                 — tight ROI around the speckled gauge
    <out>_truth.json               — applied strain / translation ground truth
"""

import argparse
import json
import cv2
import numpy as np

W, H = 2000, 1500


# ---------------------------------------------------------------------------
# Background: grey desk with subtle texture
# ---------------------------------------------------------------------------
def make_background(rng):
    base = np.full((H, W, 3), 118, dtype=np.float32)
    noise = rng.normal(0, 6, (H, W)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (9, 9), 0)
    for c in range(3):
        base[:, :, c] += noise + rng.normal(0, 2, (H, W)).astype(np.float32)
    # Slight vignette
    cx, cy = W / 2, H / 2
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    vign = 1.0 - 0.18 * ((xs - cx) ** 2 + (ys - cy) ** 2) / (cx ** 2 + cy ** 2)
    for c in range(3):
        base[:, :, c] *= vign
    return np.clip(base, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Strip geometry helpers
# ---------------------------------------------------------------------------
def _strip_mask(cx0, cy0, cx1, cy1, half_w, shape):
    """Binary mask for a rectangular strip between two centre points."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    angle = np.arctan2(cy1 - cy0, cx1 - cx0)
    dx = int(-np.sin(angle) * half_w)
    dy = int( np.cos(angle) * half_w)
    pts = np.array([
        [cx0 + dx, cy0 + dy],
        [cx1 + dx, cy1 + dy],
        [cx1 - dx, cy1 - dy],
        [cx0 - dx, cy0 - dy],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


# ---------------------------------------------------------------------------
# Composite elements
# ---------------------------------------------------------------------------
def make_scene(rng, exx, eyy, tx, ty, seed):
    # Strip layout: left anchor → gauge → right grip
    strip_cy  = H // 2 + 30            # slight off-centre
    left_x    = 120                    # string tie-off x
    gauge_x0  = 280                    # start of speckled gauge
    gauge_x1  = 1580                   # end of speckled gauge
    grip_x1   = 1820                   # right edge of right grip
    half_w    = 120                    # half-width of strip in px (~16% of H, matches real photos)

    bg = make_background(rng)
    canvas = bg.astype(np.float32)

    # ---- Bare strip (beige) -----------------------------------------------
    full_mask, _ = _strip_mask(left_x, strip_cy, grip_x1, strip_cy, half_w, canvas.shape)

    beige = np.array([170.0, 185.0, 195.0])   # BGR: warm off-white
    grain = rng.normal(0, 10, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (1, 31), 0)
    for c in range(3):
        strip_layer = np.full((H, W), beige[c]) + grain * 0.6
        alpha = (full_mask / 255.0)
        canvas[:, :, c] = canvas[:, :, c] * (1 - alpha) + strip_layer * alpha

    # Cylindrical highlight
    ys_g, xs_g = np.mgrid[0:H, 0:W]
    perp = (ys_g - strip_cy).astype(np.float32)
    highlight = 35 * np.exp(-(perp ** 2) / (2 * (half_w * 0.3) ** 2))
    shadow    = -20 * np.exp(-(perp ** 2) / (2 * (half_w * 0.75) ** 2))
    for c in range(3):
        canvas[:, :, c] = np.where(
            full_mask > 0,
            np.clip(canvas[:, :, c] + highlight + shadow, 0, 255),
            canvas[:, :, c])

    # ---- Red tape grips (rectangles over grip zones) ----------------------
    grip_zones = [
        (gauge_x1, grip_x1),    # right grip
    ]
    for x0, x1 in grip_zones:
        gm, _ = _strip_mask(x0, strip_cy, x1, strip_cy, half_w + 10, canvas.shape)
        red = np.array([45.0, 60.0, 210.0])    # BGR red
        tape_noise = rng.normal(0, 8, (H, W)).astype(np.float32)
        tape_noise = cv2.GaussianBlur(tape_noise, (1, 5), 0)
        for c in range(3):
            tape_layer = np.full((H, W), red[c]) + tape_noise
            alpha = (gm / 255.0)
            canvas[:, :, c] = canvas[:, :, c] * (1 - alpha) + tape_layer * alpha

    # Small left grip
    lgm, _ = _strip_mask(left_x, strip_cy, gauge_x0 - 20, strip_cy, half_w + 8, canvas.shape)
    for c in range(3):
        tape_layer = np.full((H, W), red[c]) + rng.normal(0, 8, (H, W)).astype(np.float32)
        alpha = (lgm / 255.0)
        canvas[:, :, c] = canvas[:, :, c] * (1 - alpha) + tape_layer * alpha

    # ---- String (thin white line to the left) ----------------------------
    string_y0 = strip_cy - 4
    for i in range(4):
        xi = left_x - rng.integers(20, 80)
        cv2.line(canvas.astype(np.uint8), (left_x, string_y0 + i),
                 (max(0, xi), string_y0 - 60 + i * 15), (220, 220, 220), 1)

    # ---- Fine dense speckle on gauge section only ------------------------
    gauge_mask, _ = _strip_mask(gauge_x0, strip_cy, gauge_x1, strip_cy,
                                half_w - 6, canvas.shape)

    # Fine dense speckle: target ~40% coverage — each dot is a small dark blob
    speckle_layer = np.zeros((H, W), dtype=np.float32)
    gauge_ys, gauge_xs = np.where(gauge_mask > 0)
    n_speckles = 1400

    for _ in range(n_speckles):
        idx = rng.integers(len(gauge_xs))
        cx, cy = int(gauge_xs[idx]), int(gauge_ys[idx])
        r = rng.uniform(2.5, 6.0)              # 2–6 px radius at 2000px width
        x0, x1 = max(0, int(cx - 3*r)), min(W, int(cx + 3*r) + 1)
        y0, y1 = max(0, int(cy - 3*r)), min(H, int(cy + 3*r) + 1)
        xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        blob = np.exp(-((xg - cx)**2 + (yg - cy)**2) / (2 * r**2))
        speckle_layer[y0:y1, x0:x1] = np.clip(
            speckle_layer[y0:y1, x0:x1] + blob, 0, 1.0)

    # Apply as dark paint: speckle_layer=1 → black, 0 → keep beige
    alpha_s = speckle_layer * (gauge_mask / 255.0)
    for c in range(3):
        canvas[:, :, c] = np.clip(
            canvas[:, :, c] * (1 - alpha_s), 0, 255)

    # ---- Camera noise ---------------------------------------------------
    canvas += rng.normal(0, 3, canvas.shape).astype(np.float32)
    ref = np.clip(canvas, 0, 255).astype(np.uint8)

    # ---- Auto ROI: simple rectangle around the eroded gauge mask --------
    margin = 18   # px inset from gauge edges
    x0_roi = gauge_x0 + margin
    x1_roi = gauge_x1 - margin
    y0_roi = strip_cy - (half_w - 6) + margin
    y1_roi = strip_cy + (half_w - 6) - margin
    pts = [[x0_roi, y0_roi], [x1_roi, y0_roi],
           [x1_roi, y1_roi], [x0_roi, y1_roi]]
    roi = {
        "mode": "polygon", "points": pts,
        "bbox": [x0_roi, y0_roi, x1_roi, y1_roi],
        "image_path": "", "image_size": [W, H], "display_scale": 1.0,
    }

    return ref, roi


def apply_strain(image, exx, eyy, exy, tx, ty, cx=0, cy=0):
    """
    Apply deformation centred on (cx, cy) so displacement is zero at the
    gauge midpoint and ±exx*half_gauge_length at the ends.
    """
    h, w = image.shape[:2]
    F = np.array([[1+exx, exy/2], [exy/2, 1+eyy]], dtype=np.float64)
    F_inv = np.linalg.inv(F)
    centre = np.array([cx, cy], dtype=np.float64)
    t      = np.array([tx, ty], dtype=np.float64)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    coords = np.stack([xg.ravel(), yg.ravel()], axis=1)
    # deformed → reference: undo translation, un-strain relative to centre
    rc = (F_inv @ (coords - t - centre).T).T + centre
    map_x = rc[:, 0].reshape(h, w).astype(np.float32)
    map_y = rc[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(image, map_x, map_y,
                     interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strain", type=float, default=0.05,
                    help="Axial (ε_xx) strain to apply (default 0.05 = 5%%)")
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--out",    default="rubber",
                    help="Output filename prefix")
    args = ap.parse_args()

    EXX = args.strain
    EYY = -0.35 * EXX      # approximate Poisson contraction for rubber
    EXY = 0.0
    TX, TY = 0.8, 0.3

    rng = np.random.default_rng(args.seed)

    print(f"Generating scene (seed={args.seed}, ε_xx={EXX:+.4f})...")
    ref, roi = make_scene(rng, EXX, EYY, TX, TY, args.seed)

    # Centre the deformation on the gauge midpoint so max displacement
    # is ±exx*half_gauge_length, keeping the search radius manageable.
    GAUGE_CX = (280 + 1580) / 2   # midpoint x of the gauge section
    GAUGE_CY = W // 2 + 30        # strip centre y (matches make_scene)

    print("Applying deformation...")
    defm = apply_strain(ref, EXX, EYY, EXY, TX, TY, cx=GAUGE_CX, cy=GAUGE_CY)

    ref_path  = f"{args.out}_ref.png"
    def_path  = f"{args.out}_def.png"
    roi_path  = f"{args.out}_roi.json"
    truth_path = f"{args.out}_truth.json"

    cv2.imwrite(ref_path,  ref)
    cv2.imwrite(def_path,  defm)

    roi["image_path"] = ref_path
    with open(roi_path, "w") as f:
        json.dump(roi, f, indent=2)

    truth = {"exx": EXX, "eyy": EYY, "exy": EXY, "tx": TX, "ty": TY}
    with open(truth_path, "w") as f:
        json.dump(truth, f, indent=2)

    print(f"\nSaved:")
    print(f"  {ref_path}  /  {def_path}   ({W}x{H} px)")
    print(f"  {roi_path}  ({len(roi['points'])} vertices)")
    print(f"  {truth_path}")
    print(f"\nApplied: ε_xx={EXX:+.4f}  ε_yy={EYY:+.4f}  ε_xy={EXY:+.4f}")
    print(f"         tx={TX} px  ty={TY} px")
    print(f"\nRun DIC:")
    print(f"  .venv/bin/python run_dic.py {ref_path} {def_path} "
          f"--roi {roi_path} --out results/{args.out}")


if __name__ == "__main__":
    main()
