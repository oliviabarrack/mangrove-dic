"""
DIC pipeline for real images with ROI support.

Usage:
    python run_dic.py reference.jpg deformed.jpg --roi roi.json [options]

Options:
    --roi      Path to ROI JSON from select_roi.py  (required)
    --subset   Subset size in pixels                (default: 31)
    --step     Grid step between subsets in pixels  (default: 16)
    --search   Search radius in pixels              (default: 18)
    --out      Output prefix, e.g. results/run1     (default: dic_result)

Outputs:
    <out>.png   — strain map figure
    <out>.json  — displacement + strain grids as JSON
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from dic_core import (
    load_roi, crop_to_roi, make_roi_mask,
    run_dic, compute_strain,
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: cannot open '{path}'", file=sys.stderr)
        sys.exit(1)
    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _dark_ax(ax, title):
    ax.set_title(title, color="white", fontsize=10)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_facecolor("#0d1117")


def _colorbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color="#aaa")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#ccc", fontsize=7)


def _draw_roi_outline(ax, roi, crop_origin, color="#00ff55", lw=1.2):
    """Overlay the ROI polygon on an axis that shows the cropped image."""
    ox, oy = crop_origin
    pts = np.array([[x - ox, y - oy] for x, y in roi["points"]], dtype=float)
    pts = np.vstack([pts, pts[0]])  # close the loop
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=5)


def visualise(ref_crop, def_crop, roi, crop_origin,
              gx, gy, u_field, v_field, exx, eyy, exy, out_prefix):

    mu  = np.nanmedian(u_field)
    mv  = np.nanmedian(v_field)
    mxx = np.nanmedian(exx)
    myy = np.nanmedian(eyy)
    mxy = np.nanmedian(exy)

    # Masked arrays so NaN regions render as transparent
    um  = np.ma.masked_invalid(u_field)
    vm  = np.ma.masked_invalid(v_field)
    exxm = np.ma.masked_invalid(exx)
    eyym = np.ma.masked_invalid(eyy)
    exym = np.ma.masked_invalid(exy)

    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("DIC Result — Real Image", color="white",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.38,
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.06)

    # Row 0: reference, deformed, u, v
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_ref.imshow(ref_crop, cmap="gray", vmin=0, vmax=255)
    _draw_roi_outline(ax_ref, roi, crop_origin)
    _dark_ax(ax_ref, "Reference (ROI)")

    ax_def = fig.add_subplot(gs[0, 1])
    ax_def.imshow(def_crop, cmap="gray", vmin=0, vmax=255)
    _draw_roi_outline(ax_def, roi, crop_origin)
    _dark_ax(ax_def, "Deformed (ROI)")

    ext = [gx.min(), gx.max(), gy.max(), gy.min()]

    ax_u = fig.add_subplot(gs[0, 2])
    im_u = ax_u.imshow(um, cmap="RdBu_r", extent=ext, aspect="auto")
    _colorbar(fig, im_u, ax_u)
    _dark_ax(ax_u, f"Displacement u  (median {mu:+.3f} px)")

    ax_v = fig.add_subplot(gs[0, 3])
    im_v = ax_v.imshow(vm, cmap="RdBu_r", extent=ext, aspect="auto")
    _colorbar(fig, im_v, ax_v)
    _dark_ax(ax_v, f"Displacement v  (median {mv:+.3f} px)")

    # Row 1: exx, eyy, exy, summary
    strain_kw = dict(cmap="plasma", extent=ext, aspect="auto")

    ax_xx = fig.add_subplot(gs[1, 0])
    im_xx = ax_xx.imshow(exxm, **strain_kw)
    _colorbar(fig, im_xx, ax_xx)
    _dark_ax(ax_xx, f"ε_xx  (median {mxx:+.5f})")

    ax_yy = fig.add_subplot(gs[1, 1])
    im_yy = ax_yy.imshow(eyym, **strain_kw)
    _colorbar(fig, im_yy, ax_yy)
    _dark_ax(ax_yy, f"ε_yy  (median {myy:+.5f})")

    ax_xy = fig.add_subplot(gs[1, 2])
    im_xy = ax_xy.imshow(exym, **strain_kw)
    _colorbar(fig, im_xy, ax_xy)
    _dark_ax(ax_xy, f"ε_xy  (median {mxy:+.5f})")

    # Summary panel
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.set_facecolor("#161b22")
    ax_txt.axis("off")

    active = int(np.sum(~np.isnan(u_field)))
    total  = u_field.size

    lines = [
        ("Measured Strain (median)", "#58a6ff"),
        ("", "#888"),
        (f"  ε_xx  =  {mxx:+.5f}", "#e6edf3"),
        (f"  ε_yy  =  {myy:+.5f}", "#e6edf3"),
        (f"  ε_xy  =  {mxy:+.5f}", "#e6edf3"),
        ("", "#888"),
        ("Displacement (median)", "#58a6ff"),
        (f"  u     =  {mu:+.3f} px", "#e6edf3"),
        (f"  v     =  {mv:+.3f} px", "#e6edf3"),
        ("", "#888"),
        (f"Subsets: {active}/{total} in ROI", "#888"),
    ]
    y = 0.96
    for text, col in lines:
        ax_txt.text(0.05, y, text, transform=ax_txt.transAxes,
                    color=col, fontsize=9, fontfamily="monospace", va="top")
        y -= 0.085

    out_png = f"{out_prefix}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"Saved figure → {out_png}")
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="DIC on real images with ROI.")
    ap.add_argument("reference",  help="Reference image path")
    ap.add_argument("deformed",   help="Deformed image path")
    ap.add_argument("--roi",      required=True, help="ROI JSON from select_roi.py")
    ap.add_argument("--subset",   type=int, default=31,  help="Subset size (px)")
    ap.add_argument("--step",     type=int, default=16,  help="Grid step (px)")
    ap.add_argument("--search",   type=int, default=18,  help="Search radius (px)")
    ap.add_argument("--out",      default="dic_result",  help="Output file prefix")
    args = ap.parse_args()

    print(f"\nReference : {args.reference}")
    print(f"Deformed  : {args.deformed}")
    print(f"ROI       : {args.roi}")
    print(f"Subset {args.subset} px | Step {args.step} px | Search ±{args.search} px\n")

    # Load images
    ref_full = load_gray(args.reference)
    def_full = load_gray(args.deformed)

    if ref_full.shape != def_full.shape:
        print("Warning: reference and deformed images have different sizes.",
              file=sys.stderr)

    # Load ROI and crop
    roi = load_roi(args.roi)
    ref_crop, origin = crop_to_roi(ref_full, roi)
    # def stays full-frame so search_radius is not constrained by crop width
    print(f"Cropped to ROI bbox {roi['bbox']}  →  "
          f"{ref_crop.shape[1]}×{ref_crop.shape[0]} px")

    # Build polygon mask in cropped-image coordinates
    mask = make_roi_mask(roi, ref_crop.shape, crop_origin=origin)
    n_mask = int(np.sum(mask > 0))
    print(f"ROI mask: {n_mask} px ({n_mask / mask.size * 100:.1f}% of bbox)\n")

    # Run DIC (search in full deformed image via def_offset)
    print("Running DIC...")
    gx, gy, u_field, v_field = run_dic(
        ref_crop, def_full,
        subset_size=args.subset,
        step=args.step,
        search_radius=args.search,
        mask=mask,
        def_offset=origin,
    )

    # Compute strain
    print("Computing strain field...")
    exx, eyy, exy = compute_strain(gx, gy, u_field, v_field)

    # Print summary
    print(f"\n{'─'*44}")
    print(f"{'Field':<12} {'Median':>10} {'Std':>10}")
    print(f"{'─'*44}")
    for name, data in [("u (px)", u_field), ("v (px)", v_field),
                        ("ε_xx",   exx),     ("ε_yy",   eyy),
                        ("ε_xy",   exy)]:
        print(f"{name:<12} {np.nanmedian(data):>10.5f} {np.nanstd(data):>10.5f}")
    print(f"{'─'*44}")

    # Save numeric results
    out_json = f"{args.out}.json"
    Path(out_json).write_text(json.dumps({
        "roi":          roi,
        "crop_origin":  list(origin),
        "subset_size":  args.subset,
        "step":         args.step,
        "search_radius": args.search,
        "grid_x":       gx.tolist(),
        "grid_y":       gy.tolist(),
        "u":            [[None if np.isnan(v) else v for v in row]
                         for row in u_field],
        "v":            [[None if np.isnan(v) else v for v in row]
                         for row in v_field],
        "exx":          [[None if np.isnan(v) else v for v in row]
                         for row in exx],
        "eyy":          [[None if np.isnan(v) else v for v in row]
                         for row in eyy],
        "exy":          [[None if np.isnan(v) else v for v in row]
                         for row in exy],
    }, indent=2))
    print(f"Saved data  → {out_json}")

    # Visualise
    print("Generating figure...")
    def_crop, _ = crop_to_roi(def_full, roi)
    visualise(ref_crop, def_crop, roi, origin,
              gx, gy, u_field, v_field, exx, eyy, exy, args.out)


if __name__ == "__main__":
    main()
