"""
DIC GUI — upload reference/deformed images, draw an ROI, run DIC, view
a colorful strain map, and (optionally) check percent error against a
known ground truth.

Usage:
    python dic_gui.py

Everything happens through buttons in a tkinter window. No terminal
interaction required after launch.
"""

import json
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.gridspec as gridspec
import numpy as np
import pillow_heif
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from dic_core import compute_strain, compute_stress, crop_to_roi, make_roi_mask, run_dic

# iPhones save photos as HEIC/HEIF by default; register a Pillow plugin so
# those open the same way as PNG/JPG.
pillow_heif.register_heif_opener()

CANVAS_MAX_W = 480
CANVAS_MAX_H = 480

# Friendly presets so users don't have to understand subset/step/search
# to get a reasonable result. "Advanced settings" still exposes the raw
# numbers for anyone who wants to fine-tune them.
PRESETS = {
    "Fine detail (slower, denser grid)":   dict(subset=21, step=10, search=12),
    "Balanced (recommended)":              dict(subset=31, step=16, search=18),
    "Large stretch (faster, coarser grid)": dict(subset=41, step=24, search=30),
}

TOOLTIP_SUBSET = (
    "Subset size: the size (in pixels) of the little patch of speckle "
    "pattern compared between the two photos. Bigger = smoother, more "
    "reliable matches but less fine detail."
)
TOOLTIP_STEP = (
    "Step: the spacing (in pixels) between the points where strain is "
    "measured. Smaller = a denser grid of results, but takes longer to run."
)
TOOLTIP_SEARCH = (
    "Search radius: how far (in pixels) the tool looks around each point "
    "to find its new position in the deformed photo. Increase this if "
    "your material stretched a lot between photos."
)


def load_gray(path):
    """Load any photo (including iPhone HEIC/HEIF) as a grayscale array."""
    try:
        img = Image.open(path).convert("L")
    except Exception as e:
        raise ValueError(f"Cannot open image: {path}\n({e})")
    return np.array(img, dtype=np.float32)


class ToolTip:
    """Small hover tooltip for any tkinter widget."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _event=None):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left",
                 background="#222831", foreground="#eeeeee",
                 relief="solid", borderwidth=1, font=("Helvetica", 10),
                 wraplength=280, padx=8, pady=6).pack()

    def hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class RoiCanvas(ttk.Frame):
    """Canvas that shows a fitted-to-screen image and lets the user select
    an ROI either by dragging a rectangle or by clicking polygon vertices.
    Coordinates are tracked in original-image space."""

    CLOSE_TOL = 10  # canvas px — click near the first vertex to close

    def __init__(self, parent, interactive=True, **kw):
        super().__init__(parent, **kw)
        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.image_bgr = None
        self.photo = None
        self.scale = 1.0
        self.mode = "rect"  # "rect" or "poly"

        # Rectangle-mode state
        self.rect_id = None
        self.start_xy = None
        self.roi_bbox = None

        # Polygon-mode state
        self.poly_points = []       # canvas-space vertices
        self.poly_item_ids = []     # canvas item ids to clear on reset
        self.poly_closed = False
        self._preview_line_id = None

        if interactive:
            self.canvas.bind("<ButtonPress-1>", self._on_press)
            self.canvas.bind("<B1-Motion>", self._on_drag)
            self.canvas.bind("<ButtonRelease-1>", self._on_release)
            self.canvas.bind("<Button-2>", self._on_right_click)
            self.canvas.bind("<Button-3>", self._on_right_click)
            self.canvas.bind("<Motion>", self._on_motion)

    def set_mode(self, mode):
        self.clear_roi()
        self.mode = mode

    def set_image(self, image_gray):
        self.image_bgr = image_gray
        self.clear_roi()
        h, w = image_gray.shape[:2]
        self.scale = min(CANVAS_MAX_W / w, CANVAS_MAX_H / h, 1.0)
        disp_w, disp_h = max(1, int(w * self.scale)), max(1, int(h * self.scale))

        rgb = cv2.cvtColor(image_gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        disp = cv2.resize(rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        self.photo = ImageTk.PhotoImage(image=Image.fromarray(disp))
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="bg")

    def clear_roi(self):
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.roi_bbox = None
        self.start_xy = None

        for iid in self.poly_item_ids:
            self.canvas.delete(iid)
        self.poly_item_ids = []
        self.poly_points = []
        self.poly_closed = False
        if self._preview_line_id:
            self.canvas.delete(self._preview_line_id)
            self._preview_line_id = None

    # -- Rectangle mode ---------------------------------------------------
    def _on_press(self, event):
        if self.image_bgr is None:
            return
        if self.mode == "rect":
            self.start_xy = (event.x, event.y)
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
        else:
            self._poly_click(event.x, event.y)

    def _on_drag(self, event):
        if self.mode != "rect" or self.start_xy is None:
            return
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x0, y0 = self.start_xy
        self.rect_id = self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="#00ff55", width=2)

    def _on_release(self, event):
        if self.mode != "rect" or self.start_xy is None or self.image_bgr is None:
            return
        x0, y0 = self.start_xy
        x1, y1 = event.x, event.y
        cx1, cx2 = sorted((x0, x1))
        cy1, cy2 = sorted((y0, y1))
        self.start_xy = None

        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            self.roi_bbox = None
            return

        h, w = self.image_bgr.shape[:2]
        ox1 = max(0, min(w, round(cx1 / self.scale)))
        oy1 = max(0, min(h, round(cy1 / self.scale)))
        ox2 = max(0, min(w, round(cx2 / self.scale)))
        oy2 = max(0, min(h, round(cy2 / self.scale)))
        self.roi_bbox = (ox1, oy1, ox2, oy2)

    # -- Polygon mode -------------------------------------------------------
    def _near(self, p1, p2, tol=None):
        tol = tol if tol is not None else self.CLOSE_TOL
        return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol

    def _poly_click(self, x, y):
        if self.image_bgr is None or self.poly_closed:
            return
        if len(self.poly_points) >= 3 and self._near(self.poly_points[0], (x, y)):
            self.close_polygon()
            return
        self._add_poly_point(x, y)

    def _add_poly_point(self, x, y):
        self.poly_points.append((x, y))
        r = 4
        dot = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                       fill="#00c8ff", outline="white")
        self.poly_item_ids.append(dot)
        if len(self.poly_points) > 1:
            x0, y0 = self.poly_points[-2]
            line = self.canvas.create_line(x0, y0, x, y,
                                            fill="#00ff55", width=2)
            self.poly_item_ids.append(line)

    def _on_right_click(self, event):
        if self.mode == "poly" and len(self.poly_points) >= 3 and not self.poly_closed:
            self.close_polygon()

    def _on_motion(self, event):
        if self.mode != "poly" or self.poly_closed or not self.poly_points:
            return
        if self._preview_line_id:
            self.canvas.delete(self._preview_line_id)
        lx, ly = self.poly_points[-1]
        self._preview_line_id = self.canvas.create_line(
            lx, ly, event.x, event.y, fill="#4098ff", width=1, dash=(3, 2))

    def close_polygon(self):
        if len(self.poly_points) < 3 or self.poly_closed:
            return
        self.poly_closed = True
        if self._preview_line_id:
            self.canvas.delete(self._preview_line_id)
            self._preview_line_id = None
        x0, y0 = self.poly_points[0]
        x1, y1 = self.poly_points[-1]
        closing_line = self.canvas.create_line(x1, y1, x0, y0,
                                                fill="#00ff55", width=2)
        self.poly_item_ids.append(closing_line)
        fill_poly = self.canvas.create_polygon(
            [c for pt in self.poly_points for c in pt],
            fill="#00ff55", stipple="gray25", outline="")
        # Keep the translucent fill above the photo but below vertex dots/edges
        self.canvas.tag_lower(fill_poly, self.poly_item_ids[0])
        self.poly_item_ids.append(fill_poly)

    def undo_last_point(self):
        if self.mode != "poly" or self.poly_closed or not self.poly_points:
            return
        pts = self.poly_points[:-1]
        for iid in self.poly_item_ids:
            self.canvas.delete(iid)
        self.poly_item_ids = []
        self.poly_points = []
        for (x, y) in pts:
            self._add_poly_point(x, y)

    # -- Export ---------------------------------------------------------------
    def get_roi(self):
        if self.mode == "rect":
            if self.roi_bbox is None:
                return None
            x1, y1, x2, y2 = self.roi_bbox
            return {
                "mode": "rectangle",
                "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "bbox": [x1, y1, x2, y2],
            }

        if not self.poly_closed or len(self.poly_points) < 3 or self.image_bgr is None:
            return None
        h, w = self.image_bgr.shape[:2]
        pts = []
        for (cx, cy) in self.poly_points:
            ox = max(0, min(w, round(cx / self.scale)))
            oy = max(0, min(h, round(cy / self.scale)))
            pts.append([ox, oy])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return {
            "mode": "polygon",
            "points": pts,
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
        }


class DicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mangrove DIC")
        self.geometry("1350x840")
        self.configure(bg="#0d1117")

        self.ref_path = None
        self.def_path = None
        self.ref_gray = None
        self.def_gray = None
        self.truth = None
        self.truth_name = None
        self._last_result = None

        self._build_layout()

    # ------------------------------------------------------------------
    def _build_layout(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=10, pady=8)

        ttk.Button(top, text="1. Upload Original Photo",
                   command=self.load_reference).pack(side="left", padx=4)
        ttk.Button(top, text="2. Upload Deformed Photo",
                   command=self.load_deformed).pack(side="left", padx=4)
        ttk.Button(top, text="Load Ground Truth (optional)",
                   command=self.load_ground_truth).pack(side="left", padx=(20, 4))

        self.run_btn = ttk.Button(top, text="3. Run DIC ▶",
                                   command=self.run_dic_clicked,
                                   state="disabled")
        self.run_btn.pack(side="right", padx=4)

        # Detail-level row (replaces raw subset/step/search for most users)
        settings = ttk.Frame(self)
        settings.pack(side="top", fill="x", padx=10)

        ttk.Label(settings, text="Detail level:").pack(side="left")
        self.preset_var = tk.StringVar(value="Balanced (recommended)")
        preset_box = ttk.Combobox(settings, textvariable=self.preset_var,
                                   values=list(PRESETS.keys()),
                                   state="readonly", width=32)
        preset_box.pack(side="left", padx=6)
        preset_box.bind("<<ComboboxSelected>>", self._on_preset_change)
        ToolTip(preset_box,
                "Controls how fine-grained the measurement grid is. "
                "'Balanced' works for most photos. Pick 'Large stretch' if "
                "your material moved a lot between photos.")

        self.advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings, text="Show advanced settings",
                        variable=self.advanced_var,
                        command=self._toggle_advanced).pack(side="left", padx=(16, 0))

        self.advanced_frame = ttk.Frame(self)
        self.subset_var = tk.IntVar(value=PRESETS["Balanced (recommended)"]["subset"])
        self.step_var = tk.IntVar(value=PRESETS["Balanced (recommended)"]["step"])
        self.search_var = tk.IntVar(value=PRESETS["Balanced (recommended)"]["search"])
        self._build_advanced_row()

        # Ground truth indicator
        self.truth_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.truth_var,
                  foreground="#7ee787", background="#0d1117"
                  ).pack(side="top", fill="x", padx=12)

        # Manual expected-strain entry (alternative to loading a truth.json)
        manual_row = ttk.Frame(self)
        manual_row.pack(side="top", fill="x", padx=12, pady=(0, 4))

        manual_lbl = ttk.Label(manual_row, text="Expected strain (optional) ❓:")
        manual_lbl.pack(side="left")
        ToolTip(manual_lbl,
                "Type in the strain you expect from this test (e.g. 5 for "
                "5%) to see percent error after running DIC — no file "
                "needed. Leave any field blank to skip it. If you also "
                "loaded a Ground Truth file, values typed here take "
                "priority for whichever fields are filled in.")

        self.manual_exx_var = tk.StringVar()
        self.manual_eyy_var = tk.StringVar()
        self.manual_exy_var = tk.StringVar()

        for label, var in (("ε_xx %", self.manual_exx_var),
                            ("ε_yy %", self.manual_eyy_var),
                            ("ε_xy %", self.manual_exy_var)):
            ttk.Label(manual_row, text=f"  {label}").pack(side="left")
            ttk.Entry(manual_row, textvariable=var, width=6).pack(side="left")

        # Material properties (optional) — if filled in, stress is computed
        # from the measured strain via isotropic plane-stress Hooke's law.
        material_row = ttk.Frame(self)
        material_row.pack(side="top", fill="x", padx=12, pady=(0, 4))

        material_lbl = ttk.Label(material_row, text="Material properties (for stress) ❓:")
        material_lbl.pack(side="left")
        ToolTip(material_lbl,
                "Optional. Fill in both fields to also compute stress from "
                "the measured strain, using isotropic plane-stress Hooke's "
                "law: σxx = E/(1-ν²)·(εxx+ν·εyy), σyy = E/(1-ν²)·(εyy+ν·εxx), "
                "τxy = E/(2(1+ν))·γxy. Plane stress assumes a thin, free "
                "surface (e.g. a branch or stick) — not a thick/constrained "
                "specimen. Leave blank to skip stress and see strain only.")

        self.material_e_var = tk.StringVar()
        self.material_nu_var = tk.StringVar()

        ttk.Label(material_row, text="  Young's modulus E (GPa)").pack(side="left")
        ttk.Entry(material_row, textvariable=self.material_e_var, width=8).pack(side="left")
        ttk.Label(material_row, text="  Poisson's ratio ν").pack(side="left")
        ttk.Entry(material_row, textvariable=self.material_nu_var, width=6).pack(side="left")

        self.status_var = tk.StringVar(value="Upload the original (reference) photo to begin.")
        ttk.Label(self, textvariable=self.status_var,
                  foreground="#8ab4f8", background="#0d1117"
                  ).pack(side="top", fill="x", padx=12, pady=(2, 6))

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=8)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))

        roi_mode_row = ttk.Frame(left)
        roi_mode_row.pack(fill="x")
        ttk.Label(roi_mode_row, text="ROI shape:").pack(side="left")
        self.roi_mode_var = tk.StringVar(value="rect")
        ttk.Radiobutton(roi_mode_row, text="Rectangle", value="rect",
                         variable=self.roi_mode_var,
                         command=self._on_roi_mode_change).pack(side="left", padx=(4, 0))
        ttk.Radiobutton(roi_mode_row, text="Polygon (freeform)", value="poly",
                         variable=self.roi_mode_var,
                         command=self._on_roi_mode_change).pack(side="left", padx=(4, 0))

        self.roi_instructions_var = tk.StringVar(
            value="Original photo — drag to draw the ROI box")
        ttk.Label(left, textvariable=self.roi_instructions_var).pack()
        self.roi_canvas = RoiCanvas(left, interactive=True)
        self.roi_canvas.pack()

        roi_btn_row = ttk.Frame(left)
        roi_btn_row.pack(pady=(4, 0))
        self.undo_pt_btn = ttk.Button(roi_btn_row, text="Undo Last Point",
                                       command=self._undo_roi_point, state="disabled")
        self.undo_pt_btn.pack(side="left", padx=2)
        self.close_poly_btn = ttk.Button(roi_btn_row, text="Close Polygon",
                                          command=self._close_roi_polygon, state="disabled")
        self.close_poly_btn.pack(side="left", padx=2)
        ttk.Button(roi_btn_row, text="Clear ROI",
                   command=self.roi_canvas.clear_roi).pack(side="left", padx=2)

        ttk.Label(left, text="Deformed photo (preview)").pack(pady=(12, 0))
        self.def_canvas = RoiCanvas(left, interactive=False)
        self.def_canvas.pack()

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self.figure = Figure(figsize=(9.5, 7.5), facecolor="#0d1117")
        self.fig_canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_placeholder()

        bottom = ttk.Frame(right)
        bottom.pack(side="bottom", fill="x", pady=(4, 0))
        self.save_btn = ttk.Button(bottom, text="Save Results (PNG + JSON)",
                                    command=self.save_results, state="disabled")
        self.save_btn.pack(side="right")

    def _build_advanced_row(self):
        row = self.advanced_frame
        for w in row.winfo_children():
            w.destroy()

        def labeled_entry(text, var, tooltip):
            f = ttk.Frame(row)
            f.pack(side="left", padx=8)
            lbl = ttk.Label(f, text=text + " ❓")
            lbl.pack(side="left")
            ToolTip(lbl, tooltip)
            ttk.Entry(f, textvariable=var, width=6).pack(side="left", padx=4)

        labeled_entry("Subset (px)", self.subset_var, TOOLTIP_SUBSET)
        labeled_entry("Step (px)", self.step_var, TOOLTIP_STEP)
        labeled_entry("Search ± (px)", self.search_var, TOOLTIP_SEARCH)

    def _toggle_advanced(self):
        if self.advanced_var.get():
            self.advanced_frame.pack(side="top", fill="x", padx=10, pady=(4, 0))
        else:
            self.advanced_frame.pack_forget()

    def _on_preset_change(self, _event=None):
        p = PRESETS[self.preset_var.get()]
        self.subset_var.set(p["subset"])
        self.step_var.set(p["step"])
        self.search_var.set(p["search"])

    def _draw_placeholder(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#0d1117")
        ax.axis("off")
        ax.text(0.5, 0.5, "Strain map will appear here",
                ha="center", va="center", color="#555", fontsize=13,
                transform=ax.transAxes)
        self.fig_canvas.draw()

    # ------------------------------------------------------------------
    def load_reference(self):
        path = filedialog.askopenfilename(
            title="Select original (reference) photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.heic *.heif"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.ref_gray = load_gray(path)
        except ValueError as e:
            messagebox.showerror("Load error", str(e))
            return
        self.ref_path = path
        self.roi_canvas.set_image(self.ref_gray)
        self.status_var.set(
            f"Original loaded: {Path(path).name}  "
            f"({self.ref_gray.shape[1]}×{self.ref_gray.shape[0]} px). "
            "Now drag a box on the photo to select the ROI.")
        self._update_run_state()

    def load_deformed(self):
        path = filedialog.askopenfilename(
            title="Select deformed photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.heic *.heif"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            self.def_gray = load_gray(path)
        except ValueError as e:
            messagebox.showerror("Load error", str(e))
            return
        self.def_path = path
        self.def_canvas.set_image(self.def_gray)
        self.status_var.set(f"Deformed photo loaded: {Path(path).name}  "
                             f"({self.def_gray.shape[1]}×{self.def_gray.shape[0]} px)")
        self._update_run_state()

    def load_ground_truth(self):
        path = filedialog.askopenfilename(
            title="Select ground truth JSON (exx/eyy/exy/tx/ty)",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            messagebox.showerror("Load error", f"Could not read ground truth:\n{e}")
            return
        self.truth = data
        self.truth_name = Path(path).name
        self.truth_var.set(
            f"Ground truth loaded: {self.truth_name}  "
            f"(will show % error after running DIC)")

    def _update_run_state(self):
        ready = self.ref_gray is not None and self.def_gray is not None
        self.run_btn.config(state="normal" if ready else "disabled")

    def _on_roi_mode_change(self):
        mode = self.roi_mode_var.get()
        self.roi_canvas.set_mode(mode)
        if mode == "rect":
            self.roi_instructions_var.set(
                "Original photo — drag to draw the ROI box")
            self.undo_pt_btn.config(state="disabled")
            self.close_poly_btn.config(state="disabled")
        else:
            self.roi_instructions_var.set(
                "Original photo — click to place points around the branch, "
                "then click near the first point (or Close Polygon) to finish")
            self.undo_pt_btn.config(state="normal")
            self.close_poly_btn.config(state="normal")

    def _undo_roi_point(self):
        self.roi_canvas.undo_last_point()

    def _close_roi_polygon(self):
        self.roi_canvas.close_polygon()

    def _get_truth_dict(self):
        """Merge the loaded ground-truth file (if any) with manually typed
        expected-strain values, which take priority per-field when filled in.
        Returns None if nothing is set at all."""
        truth = dict(self.truth) if self.truth else {}
        manual = (("exx", self.manual_exx_var), ("eyy", self.manual_eyy_var),
                  ("exy", self.manual_exy_var))
        for key, var in manual:
            text = var.get().strip()
            if not text:
                continue
            try:
                truth[key] = float(text) / 100.0
            except ValueError:
                pass
        return truth or None

    def _get_material_props(self):
        """Returns (E_mpa, nu) if both fields are filled with valid numbers,
        None if both are blank (stress skipped). Raises ValueError if only
        one is filled or a value doesn't parse, so the caller can surface
        that as a clear error rather than silently skipping stress."""
        e_text = self.material_e_var.get().strip()
        nu_text = self.material_nu_var.get().strip()
        if not e_text and not nu_text:
            return None
        if not e_text or not nu_text:
            raise ValueError(
                "Fill in both Young's modulus and Poisson's ratio to "
                "compute stress (or leave both blank to skip it).")
        e_gpa = float(e_text)
        nu = float(nu_text)
        return e_gpa * 1000.0, nu  # GPa -> MPa

    # ------------------------------------------------------------------
    def run_dic_clicked(self):
        roi = self.roi_canvas.get_roi()
        if roi is None:
            messagebox.showwarning(
                "No ROI selected",
                "Select an ROI on the original photo before running DIC "
                "(drag a box, or place polygon points and close the shape).")
            return
        if self.ref_gray.shape != self.def_gray.shape:
            if not messagebox.askyesno(
                    "Size mismatch",
                    "The original and deformed photos are different sizes. "
                    "Continue anyway?"):
                return

        try:
            subset = int(self.subset_var.get())
            step = int(self.step_var.get())
            search = int(self.search_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid settings",
                                  "Subset/Step/Search must be integers.")
            return

        try:
            material = self._get_material_props()
        except ValueError as e:
            messagebox.showerror("Invalid material properties", str(e))
            return

        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.status_var.set("Running DIC... this may take a moment.")
        thread = threading.Thread(
            target=self._run_dic_worker,
            args=(roi, subset, step, search, material), daemon=True)
        thread.start()

    def _run_dic_worker(self, roi, subset, step, search, material):
        try:
            ref_crop, origin = crop_to_roi(self.ref_gray, roi)
            mask = make_roi_mask(roi, ref_crop.shape, crop_origin=origin)

            gx, gy, u_field, v_field = run_dic(
                ref_crop, self.def_gray,
                subset_size=subset, step=step,
                search_radius=search, mask=mask, def_offset=origin)

            exx, eyy, exy = compute_strain(gx, gy, u_field, v_field)
            def_crop, _ = crop_to_roi(self.def_gray, roi)

            if material is not None:
                e_mpa, nu = material
                sxx, syy, sxy = compute_stress(exx, eyy, exy, e_mpa, nu)
            else:
                sxx = syy = sxy = None

            result = dict(roi=roi, origin=origin, ref_crop=ref_crop,
                          def_crop=def_crop, gx=gx, gy=gy,
                          u=u_field, v=v_field, exx=exx, eyy=eyy, exy=exy,
                          material=material, sxx=sxx, syy=syy, sxy=sxy)
        except Exception as e:
            self.after(0, lambda: self._on_dic_error(e))
            return
        self.after(0, lambda: self._on_dic_done(result))

    def _on_dic_error(self, error):
        self.run_btn.config(state="normal")
        self.status_var.set("DIC failed — see error dialog.")
        messagebox.showerror("DIC error", str(error))

    def _on_dic_done(self, result):
        self.run_btn.config(state="normal")
        self.save_btn.config(state="normal")
        self.status_var.set(
            "DIC complete. Median strain: "
            f"εxx={np.nanmedian(result['exx']):+.5f}  "
            f"εyy={np.nanmedian(result['eyy']):+.5f}  "
            f"εxy={np.nanmedian(result['exy']):+.5f}")
        self._plot_result(result)

    # ------------------------------------------------------------------
    @staticmethod
    def _pct_error(measured, truth):
        """Percent error vs. ground truth. Returns None if truth is ~0
        (percent error is meaningless there; caller should show abs error)."""
        if truth is None:
            return None
        if abs(truth) < 1e-9:
            return None
        return (measured - truth) / abs(truth) * 100

    def _plot_result(self, r):
        self.figure.clear()
        has_stress = r.get("sxx") is not None
        nrows = 3 if has_stress else 2
        self.figure.set_size_inches(9.5, 10.6 if has_stress else 7.5, forward=True)
        gs = gridspec.GridSpec(nrows, 4, figure=self.figure,
                                hspace=0.5, wspace=0.45,
                                left=0.05, right=0.97, top=0.95, bottom=0.05)

        def dark(ax, title):
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="#888", labelsize=7)
            for s in ax.spines.values():
                s.set_edgecolor("#333")
            ax.set_facecolor("#0d1117")

        def colorbar(im, ax):
            cb = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors="#aaa", labelsize=7)

        mu, mv = np.nanmedian(r["u"]), np.nanmedian(r["v"])
        mxx, myy, mxy = (np.nanmedian(r["exx"]), np.nanmedian(r["eyy"]),
                         np.nanmedian(r["exy"]))

        ax_ref = self.figure.add_subplot(gs[0, 0])
        ax_ref.imshow(r["ref_crop"], cmap="gray", vmin=0, vmax=255)
        dark(ax_ref, "Original (ROI)")

        ax_def = self.figure.add_subplot(gs[0, 1])
        ax_def.imshow(r["def_crop"], cmap="gray", vmin=0, vmax=255)
        dark(ax_def, "Deformed (ROI)")

        ext = [r["gx"].min(), r["gx"].max(), r["gy"].max(), r["gy"].min()]

        um = np.ma.masked_invalid(r["u"])
        vm = np.ma.masked_invalid(r["v"])
        ax_u = self.figure.add_subplot(gs[0, 2])
        im_u = ax_u.imshow(um, cmap="coolwarm", extent=ext, aspect="auto")
        colorbar(im_u, ax_u)
        dark(ax_u, f"X shift, u  (median {mu:+.3f} px)")

        ax_v = self.figure.add_subplot(gs[0, 3])
        im_v = ax_v.imshow(vm, cmap="coolwarm", extent=ext, aspect="auto")
        colorbar(im_v, ax_v)
        dark(ax_v, f"Y shift, v  (median {mv:+.3f} px)")

        strain_kw = dict(cmap="turbo", extent=ext, aspect="auto")
        exxm = np.ma.masked_invalid(r["exx"])
        eyym = np.ma.masked_invalid(r["eyy"])
        exym = np.ma.masked_invalid(r["exy"])

        ax_xx = self.figure.add_subplot(gs[1, 0])
        im_xx = ax_xx.imshow(exxm, **strain_kw)
        colorbar(im_xx, ax_xx)
        dark(ax_xx, f"ε_xx  (median {mxx:+.5f})")

        ax_yy = self.figure.add_subplot(gs[1, 1])
        im_yy = ax_yy.imshow(eyym, **strain_kw)
        colorbar(im_yy, ax_yy)
        dark(ax_yy, f"ε_yy  (median {myy:+.5f})")

        ax_xy = self.figure.add_subplot(gs[1, 2])
        im_xy = ax_xy.imshow(exym, **strain_kw)
        colorbar(im_xy, ax_xy)
        dark(ax_xy, f"ε_xy  (median {mxy:+.5f})")

        # Summary / percent-error panel
        ax_txt = self.figure.add_subplot(gs[1, 3])
        ax_txt.set_facecolor("#161b22")
        ax_txt.axis("off")

        lines = [
            ("Displacement (median)", "#58a6ff"),
            (f"  u  =  {mu:+.3f} px", "#e6edf3"),
            (f"  v  =  {mv:+.3f} px", "#e6edf3"),
            ("", "#888"),
            ("Strain (median)", "#58a6ff"),
            (f"  ε_xx  =  {mxx:+.5f}", "#e6edf3"),
            (f"  ε_yy  =  {myy:+.5f}", "#e6edf3"),
            (f"  ε_xy  =  {mxy:+.5f}", "#e6edf3"),
        ]

        truth = self._get_truth_dict()
        if truth is not None:
            t = truth
            lines.append(("", "#888"))
            lines.append(("% error vs. expected strain", "#f0883e"))

            # Only strain is compared: it's uniform across the ROI so the
            # measured median is directly comparable to the applied truth.
            # Raw displacement (u, v) is NOT comparable to tx/ty unless the
            # ROI happens to be centered on the test rig's gauge center, so
            # we don't compute a (misleading) shift error here.
            for label, measured, truth_key in (
                ("ε_xx", mxx, "exx"), ("ε_yy", myy, "eyy"),
                ("ε_xy", mxy, "exy"),
            ):
                truth_val = t.get(truth_key)
                if truth_val is None:
                    continue
                err = self._pct_error(measured, truth_val)
                if err is None:
                    lines.append((f"  {label}: truth ≈ 0, abs err "
                                   f"{measured - truth_val:+.4f}", "#e6edf3"))
                else:
                    lines.append((f"  {label}: {err:+.1f}%", "#e6edf3"))
            if "tx" in t or "ty" in t:
                lines.append(("  (shift vs tx/ty not shown — depends", "#666"))
                lines.append(("   on ROI position, not comparable)", "#666"))

        y = 0.97
        for text, col in lines:
            ax_txt.text(0.04, y, text, transform=ax_txt.transAxes,
                        color=col, fontsize=8.5, fontfamily="monospace", va="top")
            y -= 0.078

        if has_stress:
            e_mpa, nu = r["material"]
            sxxm = np.ma.masked_invalid(r["sxx"])
            syym = np.ma.masked_invalid(r["syy"])
            sxym = np.ma.masked_invalid(r["sxy"])
            msxx, msyy, msxy = (np.nanmedian(r["sxx"]), np.nanmedian(r["syy"]),
                                 np.nanmedian(r["sxy"]))

            stress_kw = dict(cmap="turbo", extent=ext, aspect="auto")

            ax_sxx = self.figure.add_subplot(gs[2, 0])
            im_sxx = ax_sxx.imshow(sxxm, **stress_kw)
            colorbar(im_sxx, ax_sxx)
            dark(ax_sxx, f"σ_xx  (median {msxx:+.2f} MPa)")

            ax_syy = self.figure.add_subplot(gs[2, 1])
            im_syy = ax_syy.imshow(syym, **stress_kw)
            colorbar(im_syy, ax_syy)
            dark(ax_syy, f"σ_yy  (median {msyy:+.2f} MPa)")

            ax_sxy = self.figure.add_subplot(gs[2, 2])
            im_sxy = ax_sxy.imshow(sxym, **stress_kw)
            colorbar(im_sxy, ax_sxy)
            dark(ax_sxy, f"σ_xy  (median {msxy:+.2f} MPa)")

            ax_stress_txt = self.figure.add_subplot(gs[2, 3])
            ax_stress_txt.set_facecolor("#161b22")
            ax_stress_txt.axis("off")
            stress_lines = [
                ("Stress (median, plane stress)", "#58a6ff"),
                (f"  E = {e_mpa / 1000.0:g} GPa,  ν = {nu:g}", "#888"),
                (f"  σ_xx  =  {msxx:+.2f} MPa", "#e6edf3"),
                (f"  σ_yy  =  {msyy:+.2f} MPa", "#e6edf3"),
                (f"  σ_xy  =  {msxy:+.2f} MPa", "#e6edf3"),
            ]
            y = 0.9
            for text, col in stress_lines:
                ax_stress_txt.text(0.04, y, text, transform=ax_stress_txt.transAxes,
                                    color=col, fontsize=8.5, fontfamily="monospace", va="top")
                y -= 0.16

        self.figure.suptitle("DIC Result", color="white",
                              fontsize=13, fontweight="bold")
        self.fig_canvas.draw()
        self._last_result = r

    # ------------------------------------------------------------------
    def save_results(self):
        r = self._last_result
        if r is None:
            return
        out_path = filedialog.asksaveasfilename(
            title="Save results as...",
            defaultextension=".png",
            initialfile="dic_result.png",
            filetypes=[("PNG image", "*.png")])
        if not out_path:
            return
        prefix = str(Path(out_path).with_suffix(""))

        self.figure.savefig(f"{prefix}.png", dpi=150,
                             bbox_inches="tight", facecolor="#0d1117")

        data = {
            "roi": r["roi"],
            "crop_origin": list(r["origin"]),
            "grid_x": r["gx"].tolist(),
            "grid_y": r["gy"].tolist(),
            "u": [[None if np.isnan(v) else v for v in row] for row in r["u"]],
            "v": [[None if np.isnan(v) else v for v in row] for row in r["v"]],
            "exx": [[None if np.isnan(v) else v for v in row] for row in r["exx"]],
            "eyy": [[None if np.isnan(v) else v for v in row] for row in r["eyy"]],
            "exy": [[None if np.isnan(v) else v for v in row] for row in r["exy"]],
        }
        if self.truth is not None:
            data["ground_truth"] = self.truth
        if r.get("sxx") is not None:
            e_mpa, nu = r["material"]
            data["material"] = {"E_GPa": e_mpa / 1000.0, "nu": nu}
            data["sxx"] = [[None if np.isnan(v) else v for v in row] for row in r["sxx"]]
            data["syy"] = [[None if np.isnan(v) else v for v in row] for row in r["syy"]]
            data["sxy"] = [[None if np.isnan(v) else v for v in row] for row in r["sxy"]]
        Path(f"{prefix}.json").write_text(json.dumps(data, indent=2))
        messagebox.showinfo("Saved",
                             f"Saved:\n{prefix}.png\n{prefix}.json")


if __name__ == "__main__":
    app = DicApp()
    app.mainloop()
