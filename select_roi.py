"""
ROI selection tool for DIC.

Opens an image, lets you draw a rectangle or polygon over the speckled
region, then saves the coordinates to a JSON file for reuse.

Usage:
    python select_roi.py <image_path> [--out roi.json] [--mode rect|poly]

Rectangle mode (default):
    Click and drag to draw the box.
    Release the mouse to confirm.
    Press S or Enter to save, R to redraw, Q/Esc to quit without saving.

Polygon mode (--mode poly):
    Left-click to place vertices.
    Right-click or press Enter to close and confirm the polygon.
    Backspace/Z to remove the last vertex.
    Press S to save, R to reset, Q/Esc to quit without saving.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COL_LINE   = (0, 255, 80)    # green — active / confirmed outline
COL_VERTEX = (0, 200, 255)   # cyan  — polygon vertex dots
COL_DRAG   = (80, 180, 255)  # blue  — rectangle drag preview
COL_TEXT   = (255, 255, 255)
COL_DIM    = (160, 160, 160)


# ---------------------------------------------------------------------------
# Shared overlay helper
# ---------------------------------------------------------------------------

def _overlay_text(canvas, lines, origin=(10, 24), line_h=22):
    """Draw a semi-transparent instruction panel in the top-left corner."""
    x, y = origin
    max_w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
                for l in lines) + 16
    panel_h = len(lines) * line_h + 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 6, y - 18),
                  (x + max_w, y + panel_h - 4), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, line in enumerate(lines):
        col = COL_TEXT if not line.startswith("  ") else COL_DIM
        cv2.putText(canvas, line,
                    (x, y + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Rectangle mode
# ---------------------------------------------------------------------------

class RectSelector:
    def __init__(self, image):
        self.base    = image.copy()
        self.canvas  = image.copy()
        self.start   = None   # (x, y) mouse-down point
        self.end     = None   # (x, y) current drag point
        self.rect    = None   # confirmed (x1, y1, x2, y2) in pixel coords
        self.drawing = False

    # -- Mouse callback -------------------------------------------------------
    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.end     = (x, y)
            self.rect    = None

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.end = (x, y)
            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            x2 = max(self.start[0], self.end[0])
            y2 = max(self.start[1], self.end[1])
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                self.rect = (x1, y1, x2, y2)

    # -- Draw frame -----------------------------------------------------------
    def draw(self):
        self.canvas = self.base.copy()
        if self.rect:
            x1, y1, x2, y2 = self.rect
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), COL_LINE, 2)
            cv2.putText(self.canvas,
                        f"ROI  {x2-x1} x {y2-y1} px",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LINE, 1, cv2.LINE_AA)
            hint = ["S / Enter — save & quit",
                    "R         — redraw",
                    "Q / Esc   — quit without saving"]
        elif self.drawing and self.start and self.end:
            x1 = min(self.start[0], self.end[0])
            y1 = min(self.start[1], self.end[1])
            x2 = max(self.start[0], self.end[0])
            y2 = max(self.start[1], self.end[1])
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), COL_DRAG, 1)
            hint = ["Drag to resize …",
                    "R     — reset",
                    "Q/Esc — quit without saving"]
        else:
            hint = ["Click and drag to draw ROI",
                    "R     — reset",
                    "Q/Esc — quit without saving"]

        _overlay_text(self.canvas, hint)
        return self.canvas

    # -- Key handler ----------------------------------------------------------
    def handle_key(self, key):
        """Return 'save', 'reset', 'quit', or None."""
        if key in (ord('s'), ord('S'), 13):   # S or Enter
            if self.rect:
                return "save"
        elif key in (ord('r'), ord('R')):
            self.rect    = None
            self.drawing = False
            return "reset"
        elif key in (ord('q'), ord('Q'), 27):  # Q or Esc
            return "quit"
        return None

    # -- Export ---------------------------------------------------------------
    def export(self):
        x1, y1, x2, y2 = self.rect
        return {
            "mode":   "rectangle",
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox":   [x1, y1, x2, y2],
        }


# ---------------------------------------------------------------------------
# Polygon mode
# ---------------------------------------------------------------------------

class PolySelector:
    def __init__(self, image):
        self.base      = image.copy()
        self.canvas    = image.copy()
        self.vertices  = []      # confirmed vertices [(x,y), ...]
        self.cursor    = None    # current mouse position
        self.closed    = False   # polygon completed?

    # -- Mouse callback -------------------------------------------------------
    def callback(self, event, x, y, flags, param):
        self.cursor = (x, y)

        if self.closed:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Close polygon on click near first vertex
            if (len(self.vertices) >= 3
                    and _near(self.vertices[0], (x, y), tol=12)):
                self.closed = True
            else:
                self.vertices.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.vertices) >= 3:
                self.closed = True

    # -- Draw frame -----------------------------------------------------------
    def draw(self):
        self.canvas = self.base.copy()
        pts = self.vertices

        if pts:
            # Draw edges
            for i in range(len(pts) - 1):
                cv2.line(self.canvas, pts[i], pts[i + 1], COL_LINE, 2, cv2.LINE_AA)
            if self.closed:
                cv2.line(self.canvas, pts[-1], pts[0], COL_LINE, 2, cv2.LINE_AA)
                # Filled semi-transparent overlay
                mask = np.zeros_like(self.canvas)
                cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], (0, 255, 80))
                cv2.addWeighted(mask, 0.18, self.canvas, 0.82, 0, self.canvas)
            elif self.cursor:
                # Preview edge from last vertex to cursor
                cv2.line(self.canvas, pts[-1], self.cursor, COL_DRAG, 1, cv2.LINE_AA)
                # Highlight snap-to-close when near first vertex
                if (len(pts) >= 3 and _near(pts[0], self.cursor, tol=12)):
                    cv2.circle(self.canvas, pts[0], 12, (0, 255, 255), 1)

            # Vertex dots
            for i, p in enumerate(pts):
                cv2.circle(self.canvas, p, 5, COL_VERTEX, -1, cv2.LINE_AA)
                cv2.circle(self.canvas, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(self.canvas, str(i + 1),
                            (p[0] + 8, p[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_VERTEX, 1, cv2.LINE_AA)

        if self.closed:
            hint = ["S / Enter  — save & quit",
                    "R          — reset",
                    "Q / Esc    — quit without saving"]
        elif not pts:
            hint = ["Left-click to place vertices",
                    "Right-click or Enter when done",
                    "Backspace/Z — undo last vertex",
                    "R     — reset | Q/Esc — quit"]
        else:
            hint = [f"Vertices: {len(pts)}",
                    "Left-click near  [1]  to close",
                    "Right-click or Enter when done",
                    "Backspace/Z — undo | R — reset"]

        _overlay_text(self.canvas, hint)
        return self.canvas

    # -- Key handler ----------------------------------------------------------
    def handle_key(self, key):
        if key in (ord('s'), ord('S'), 13):  # S or Enter
            if len(self.vertices) >= 3 and not self.closed:
                self.closed = True
            if self.closed:
                return "save"
        elif key in (8, ord('z'), ord('Z')):  # Backspace or Z
            if not self.closed and self.vertices:
                self.vertices.pop()
        elif key in (ord('r'), ord('R')):
            self.vertices = []
            self.closed   = False
            return "reset"
        elif key in (ord('q'), ord('Q'), 27):
            return "quit"
        return None

    # -- Export ---------------------------------------------------------------
    def export(self):
        return {
            "mode":   "polygon",
            "points": [list(p) for p in self.vertices],
            "bbox":   _bbox(self.vertices),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _near(p1, p2, tol=12):
    return abs(p1[0] - p2[0]) < tol and abs(p1[1] - p2[1]) < tol


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _fit_to_screen(image, max_w=1400, max_h=900):
    """Scale image down if it's too large for a typical screen."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(image_path: str, out_path: str, mode: str):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: cannot open '{image_path}'", file=sys.stderr)
        sys.exit(1)

    display, scale = _fit_to_screen(img_bgr)

    selector = RectSelector(display) if mode == "rect" else PolySelector(display)

    win = "DIC — ROI Selection  (Q/Esc to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, selector.callback)

    print(f"\nImage : {image_path}  ({img_bgr.shape[1]}×{img_bgr.shape[0]} px)")
    print(f"Mode  : {'rectangle' if mode == 'rect' else 'polygon'}")
    print(f"Output: {out_path}\n")

    action = None
    while True:
        frame = selector.draw()
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        action = selector.handle_key(key)
        if action in ("save", "quit"):
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            action = "quit"
            break

    cv2.destroyAllWindows()

    if action != "save":
        print("Quit without saving.")
        return

    data = selector.export()

    # Scale coordinates back to original image pixels
    if scale < 1.0:
        data["points"] = [[round(x / scale), round(y / scale)]
                          for x, y in data["points"]]
        data["bbox"]   = [round(v / scale) for v in data["bbox"]]

    data["image_path"]  = str(Path(image_path).resolve())
    data["image_size"]  = [img_bgr.shape[1], img_bgr.shape[0]]  # [w, h]
    data["display_scale"] = round(scale, 4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))

    print(f"Saved ROI → {out}")
    print(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI selector for DIC images.")
    parser.add_argument("image",
                        help="Path to the image file (reference or any frame).")
    parser.add_argument("--out", default="roi.json",
                        help="Output JSON file (default: roi.json).")
    parser.add_argument("--mode", choices=["rect", "poly"], default="rect",
                        help="Selection mode: rect (default) or poly.")
    args = parser.parse_args()
    run(args.image, args.out, args.mode)


if __name__ == "__main__":
    main()
