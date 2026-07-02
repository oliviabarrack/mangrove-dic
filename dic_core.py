"""
Shared DIC functions: ROI handling, ZNCC correlation, strain computation.
Imported by both dic_validation.py (synthetic) and run_dic.py (real images).
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def load_roi(path):
    """Load roi.json saved by select_roi.py."""
    return json.loads(Path(path).read_text())


def crop_to_roi(image, roi):
    """
    Crop image to the ROI bounding box.

    Returns (cropped_image, (x1, y1)) where (x1, y1) is the top-left
    of the crop in original image coordinates.
    """
    x1, y1, x2, y2 = roi["bbox"]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def make_roi_mask(roi, crop_shape, crop_origin=(0, 0)):
    """
    Rasterise the ROI polygon into a binary mask the same size as the
    cropped image.

    roi          : dict from load_roi()
    crop_shape   : (height, width) of the cropped image
    crop_origin  : (x1, y1) returned by crop_to_roi()
    """
    ox, oy = crop_origin
    pts = np.array([[x - ox, y - oy] for x, y in roi["points"]],
                   dtype=np.int32)
    mask = np.zeros(crop_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ---------------------------------------------------------------------------
# ZNCC subset correlation
# ---------------------------------------------------------------------------

def _zncc_displacement(ref_subset, def_image, cx, cy, search_radius):
    """
    Zero-mean normalised cross-correlation between ref_subset and a
    search window in def_image centred on (cx, cy).

    Returns (u, v) subpixel displacement.
    """
    half = ref_subset.shape[0] // 2
    s    = ref_subset.shape[0]

    sx0 = max(0, cx - half - search_radius)
    sy0 = max(0, cy - half - search_radius)
    sx1 = min(def_image.shape[1], cx + half + search_radius + 1)
    sy1 = min(def_image.shape[0], cy + half + search_radius + 1)

    search_patch = def_image[sy0:sy1, sx0:sx1].astype(np.float64)

    tmpl = ref_subset.astype(np.float64)
    tmpl -= tmpl.mean()
    tmpl_norm = np.linalg.norm(tmpl)
    if tmpl_norm == 0:
        return np.nan, np.nan
    tmpl /= tmpl_norm

    corr = fftconvolve(search_patch, tmpl[::-1, ::-1], mode="valid")

    for rr in range(corr.shape[0]):
        for cc in range(corr.shape[1]):
            local = search_patch[rr:rr + s, cc:cc + s]
            lm = local - local.mean()
            ln = np.linalg.norm(lm)
            if ln > 0:
                corr[rr, cc] /= ln

    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    pr, pc   = peak_idx

    def _sub1d(arr, idx):
        if idx == 0 or idx == len(arr) - 1:
            return float(idx)
        den = arr[idx - 1] - 2 * arr[idx] + arr[idx + 1]
        if den == 0:
            return float(idx)
        return idx + 0.5 * (arr[idx - 1] - arr[idx + 1]) / den

    pc_sub = _sub1d(corr[pr, :], pc)
    pr_sub = _sub1d(corr[:, pc], pr)

    matched_cx = sx0 + pc_sub + half
    matched_cy = sy0 + pr_sub + half
    return matched_cx - cx, matched_cy - cy


# ---------------------------------------------------------------------------
# DIC engine
# ---------------------------------------------------------------------------

def run_dic(ref_image, def_image, subset_size=31, step=16,
            search_radius=15, mask=None, def_offset=(0, 0)):
    """
    Subset-based ZNCC DIC over ref_image / def_image.

    mask       : optional uint8 array (same HxW as ref_image). Subsets whose
                 centres fall outside the mask are skipped (NaN).
    def_offset : (ox, oy) added to subset centres when searching def_image.
                 Use when ref_image is a crop but def_image is the full frame,
                 so the search window is not artificially constrained by the
                 narrow crop width.

    Returns gx, gy, u_field, v_field  (NaN where masked out).
    """
    h, w = ref_image.shape[:2]
    half   = subset_size // 2
    ox, oy = int(def_offset[0]), int(def_offset[1])

    # Margin only needs to keep subsets inside the reference crop.
    # The deformed-image search can roam the full frame via def_offset.
    ref_margin = half + 1
    xs = np.arange(ref_margin, w - ref_margin, step)
    ys = np.arange(ref_margin, h - ref_margin, step)

    u_field = np.full((len(ys), len(xs)), np.nan)
    v_field = np.full((len(ys), len(xs)), np.nan)

    total = len(ys) * len(xs)
    done  = skipped = 0

    for j, cy in enumerate(ys):
        for i, cx in enumerate(xs):
            if mask is not None and mask[cy, cx] == 0:
                skipped += 1
                done += 1
                continue

            ref_sub = ref_image[cy - half:cy + half + 1,
                                cx - half:cx + half + 1]
            u, v = _zncc_displacement(ref_sub, def_image,
                                      cx + ox, cy + oy, search_radius)
            u_field[j, i] = u
            v_field[j, i] = v
            done += 1

            if done % 50 == 0 or done == total:
                active = done - skipped
                print(f"  DIC: {active} computed, {skipped} masked "
                      f"({done}/{total} total)", end="\r")

    print()
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, u_field, v_field


# ---------------------------------------------------------------------------
# Strain
# ---------------------------------------------------------------------------

def compute_strain(gx, gy, u_field, v_field):
    """
    Finite-difference strain on a regular grid. NaN values are masked
    during gradient computation (np.gradient propagates them, which is
    correct — edge/masked subsets produce NaN strain).
    """
    step_x = np.diff(gx[0, :]).mean()
    step_y = np.diff(gy[:, 0]).mean()

    du_dx = np.gradient(u_field, step_x, axis=1)
    du_dy = np.gradient(u_field, step_y, axis=0)
    dv_dx = np.gradient(v_field, step_x, axis=1)
    dv_dy = np.gradient(v_field, step_y, axis=0)

    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    return exx, eyy, exy


# ---------------------------------------------------------------------------
# Stress (isotropic, plane stress)
# ---------------------------------------------------------------------------

def compute_stress(exx, eyy, exy, E, nu):
    """
    Isotropic linear-elastic, plane-stress Hooke's law: converts the strain
    field to stress. Valid for a thin, free surface (no out-of-plane
    constraint) — appropriate for DIC on a branch/stick surface, not for a
    thick/constrained specimen (which would need plane strain instead).

    E   : Young's modulus (same units the caller wants stress in, e.g. MPa)
    nu  : Poisson's ratio (unitless)

    Returns sxx, syy, sxy in the units of E.
    """
    factor = E / (1 - nu ** 2)
    sxx = factor * (exx + nu * eyy)
    syy = factor * (eyy + nu * exx)
    G = E / (2 * (1 + nu))
    sxy = 2 * G * exy  # tau_xy = G * gamma_xy, gamma_xy = 2 * exy
    return sxx, syy, sxy
