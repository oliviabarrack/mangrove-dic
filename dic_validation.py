"""
Digital Image Correlation (DIC) validation script.

Generates a synthetic speckle pattern, applies a known uniform strain,
recovers the displacement field via subset-based cross-correlation with
subpixel peak fitting, and compares true vs. measured strain.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dic_core import run_dic, compute_strain


# ---------------------------------------------------------------------------
# 1. Synthetic speckle pattern
# ---------------------------------------------------------------------------

def generate_speckle(width=512, height=512, n_speckles=800,
                     speckle_radius_range=(3, 8), seed=42):
    """Generate a Gaussian speckle pattern on a grey background."""
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 128.0, dtype=np.float64)

    for _ in range(n_speckles):
        cx = rng.uniform(0, width)
        cy = rng.uniform(0, height)
        r  = rng.uniform(*speckle_radius_range)
        intensity = rng.choice([-1, 1]) * rng.uniform(60, 127)

        x0 = max(0, int(cx - 3 * r))
        x1 = min(width,  int(cx + 3 * r) + 1)
        y0 = max(0, int(cy - 3 * r))
        y1 = min(height, int(cy + 3 * r) + 1)

        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        xg, yg = np.meshgrid(xs, ys)
        blob = intensity * np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2 * r ** 2))
        image[y0:y1, x0:x1] = np.clip(image[y0:y1, x0:x1] + blob, 0, 255)

    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Apply known uniform displacement / strain
# ---------------------------------------------------------------------------

def apply_uniform_strain(image, exx=0.005, eyy=0.003, exy=0.0,
                         tx=2.0, ty=1.5):
    """
    Warp the image with a homogeneous deformation gradient.

    Deformation mapping (from reference to deformed):
        x_def = (1 + exx)*x + exy/2*y + tx
        y_def = exy/2*x + (1 + eyy)*y + ty

    OpenCV remap expects the *inverse* map (from deformed coords back
    to reference), so we invert analytically.
    """
    h, w = image.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    # Deformation gradient F and its inverse
    F = np.array([[1 + exx, exy / 2],
                  [exy / 2, 1 + eyy]], dtype=np.float64)
    t = np.array([tx, ty], dtype=np.float64)
    F_inv = np.linalg.inv(F)

    # Inverse map: reference coords from deformed coords
    coords_def = np.stack([xg.ravel(), yg.ravel()], axis=1)  # (N, 2)
    coords_ref = (F_inv @ (coords_def - t).T).T

    map_x = coords_ref[:, 0].reshape(h, w).astype(np.float32)
    map_y = coords_ref[:, 1].reshape(h, w).astype(np.float32)

    deformed = cv2.remap(image, map_x, map_y,
                         interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT_101)
    return deformed


# ---------------------------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------------------------

def visualise(ref_image, def_image, gx, gy, u_field, v_field,
              exx, eyy, exy, true_strain):
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    fig.suptitle("DIC Validation — Synthetic Speckle Pattern",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.05, right=0.97,
                           top=0.93, bottom=0.06)

    img_kw   = dict(cmap="gray", vmin=0, vmax=255)
    cmap_disp = "RdBu_r"
    cmap_str  = "plasma"

    axes_info = [
        (gs[0, 0], ref_image,  "Reference image",    img_kw),
        (gs[0, 1], def_image,  "Deformed image",     img_kw),
        (gs[0, 2], u_field,    "Displacement u (px)", dict(cmap=cmap_disp)),
        (gs[0, 3], v_field,    "Displacement v (px)", dict(cmap=cmap_disp)),
        (gs[1, 0], exx,        "Strain ε_xx",         dict(cmap=cmap_str)),
        (gs[1, 1], eyy,        "Strain ε_yy",         dict(cmap=cmap_str)),
        (gs[1, 2], exy,        "Shear strain ε_xy",   dict(cmap=cmap_str)),
    ]

    for spec, data, title, kw in axes_info:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#0d1117")

        if data is ref_image or data is def_image:
            im = ax.imshow(data, **kw)
        else:
            # Show on DIC grid
            im = ax.imshow(data, **kw,
                           extent=[gx.min(), gx.max(), gy.max(), gy.min()],
                           aspect="auto")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Summary text panel
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.set_facecolor("#161b22")
    ax_txt.axis("off")

    t_exx, t_eyy, t_exy = true_strain
    m_exx = np.median(exx)
    m_eyy = np.median(eyy)
    m_exy = np.median(exy)

    lines = [
        ("True vs. Measured Strain", None),
        ("", None),
        (f"{'Component':<10} {'True':>10} {'Meas.':>10} {'Err %':>10}", "header"),
        ("─" * 42, None),
        (f"{'ε_xx':<10} {t_exx:>10.4f} {m_exx:>10.4f} "
         f"{abs(m_exx - t_exx) / (abs(t_exx) + 1e-9) * 100:>10.2f}", None),
        (f"{'ε_yy':<10} {t_eyy:>10.4f} {m_eyy:>10.4f} "
         f"{abs(m_eyy - t_eyy) / (abs(t_eyy) + 1e-9) * 100:>10.2f}", None),
        (f"{'ε_xy':<10} {t_exy:>10.4f} {m_exy:>10.4f} "
         f"{abs(m_exy - t_exy) / (abs(t_exy) + 1e-9) * 100:>10.2f}", None),
    ]

    y_pos = 0.95
    for line, style in lines:
        color = "#58a6ff" if style == "header" else "#e6edf3"
        size  = 9 if style != "header" else 8
        ax_txt.text(0.05, y_pos, line,
                    transform=ax_txt.transAxes,
                    color=color, fontsize=size,
                    fontfamily="monospace", va="top")
        y_pos -= 0.12

    plt.savefig("dic_validation.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print("Saved strain map → dic_validation.png")
    try:
        plt.show()
    except Exception:
        pass  # non-interactive environment — PNG already saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Parameters ---
    TRUE_EXX = 0.005   # 0.5 % tensile strain in x
    TRUE_EYY = 0.003   # 0.3 % tensile strain in y
    TRUE_EXY = 0.001   # small shear
    TX       = 2.0     # rigid-body translation x (px)
    TY       = 1.5     # rigid-body translation y (px)

    print("=== DIC Validation ===\n")
    print(f"Applied strain:  ε_xx={TRUE_EXX:.4f}  ε_yy={TRUE_EYY:.4f}  ε_xy={TRUE_EXY:.4f}")
    print(f"Rigid translation: tx={TX} px, ty={TY} px\n")

    # 1. Generate speckle
    print("Generating speckle pattern...")
    ref = generate_speckle(width=512, height=512, n_speckles=900, seed=7)

    # 2. Deform
    print("Applying deformation...")
    defm = apply_uniform_strain(ref,
                                exx=TRUE_EXX, eyy=TRUE_EYY, exy=TRUE_EXY,
                                tx=TX, ty=TY)

    # 3. Run DIC
    print("Running DIC (subset=31 px, step=16 px)...")
    gx, gy, u_field, v_field = run_dic(ref, defm,
                                       subset_size=31, step=16,
                                       search_radius=18)

    # 4. Compute strain
    print("Computing strain field...")
    exx, eyy, exy = compute_strain(gx, gy, u_field, v_field)

    # 5. Report
    m_exx = np.median(exx)
    m_eyy = np.median(eyy)
    m_exy = np.median(exy)

    print("\n─── Results ───────────────────────────────────────────")
    print(f"{'Component':<10} {'True':>10} {'Measured':>12} {'Error %':>10}")
    print("─" * 48)
    for name, true, meas in [("ε_xx", TRUE_EXX, m_exx),
                               ("ε_yy", TRUE_EYY, m_eyy),
                               ("ε_xy", TRUE_EXY, m_exy)]:
        err_pct = abs(meas - true) / (abs(true) + 1e-9) * 100
        print(f"{name:<10} {true:>10.5f} {meas:>12.5f} {err_pct:>9.2f}%")
    print("─" * 48)

    print("\nGenerating visualisation...")
    visualise(ref, defm, gx, gy, u_field, v_field,
              exx, eyy, exy, (TRUE_EXX, TRUE_EYY, TRUE_EXY))


if __name__ == "__main__":
    main()
