#!/usr/bin/env python3
"""
DT-ablation HDF5 visualiser
===========================

• Detects and displays every image-like dataset saved by your collector:
      rs_side_rgb   (N,3,H,W)   – colour
      zed_rgb       (N,3,H,W)   – colour
      zed_depth     (N,H,W)     – uint16 depth in mm
      dt_left       (N,3,H,W)   – colour
      dt_right      (N,3,H,W)   – colour
• Handles both channel-first and channel-last layouts automatically.
• Shows all streams side-by-side in a tiled grid; advance frames in sync.

Dependencies:  h5py, numpy, opencv-python
"""

import os, sys, math, glob
import cv2, h5py, numpy as np

# ---------- window layout -----------------------------------------------------
WIN_W, WIN_H = 640, 480          # size for every cv2 window

# ---------- helper: image-shape heuristics ------------------------------------
def is_image_like(shape):
    """
    Returns *True* if `shape` could be a single image or a sequence of images
    in any of the following layouts:

        (H,W)                       – gray / depth
        (H,W,C)     C∈{1,3,4}       – HWC
        (C,H,W)     C∈{1,3,4}       – CHW
        (N,H,W)                     – seq gray / depth
        (N,H,W,C)   C∈{1,3,4}       – seq HWC
        (N,C,H,W)   C∈{1,3,4}       – seq CHW
    """
    if len(shape) == 2:
        return True
    if len(shape) == 3:
        return shape[2] in (1, 3, 4) or shape[0] in (1, 3, 4)
    if len(shape) == 4:
        return shape[3] in (1, 3, 4) or shape[1] in (1, 3, 4)
    return False


def to_hwc(img):
    """Convert CHW / gray to H×W×C with C ∈ {1,3} (no alpha)."""
    if img.ndim == 2:                               # gray / depth
        return img[..., None]
    if img.ndim == 3 and img.shape[0] in (1, 3, 4): # CHW
        return img.transpose(1, 2, 0)
    return img                                      # already HWC


def colourise_depth(depth_hwc):
    """
    depth_hwc : H×W×1, uint16 (mm) *or* float32 (m)
    Returns a colour-mapped BGR image ready for display.
    """
    depth = depth_hwc.squeeze()

    # normalisation mask
    if depth.dtype == np.uint16:
        valid = depth > 0
        depth_m = depth.astype(np.float32) / 1000.0
    else:  # float32
        valid = np.isfinite(depth) & (depth > 0)
        depth_m = depth

    if not np.any(valid):
        return np.zeros((WIN_H, WIN_W, 3), np.uint8)

    d_min, d_max = depth_m[valid].min(), depth_m[valid].max()
    if abs(d_max - d_min) < 1e-6:
        d_max = d_min + 1e-3
    norm = np.zeros_like(depth_m, np.float32)
    norm[valid] = (depth_m[valid] - d_min) / (d_max - d_min) * 255
    jet = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    jet[~valid] = 0
    return cv2.resize(jet, (WIN_W, WIN_H))


def to_display(img_any):
    """Return a BGR image (WIN_H×WIN_W×3 uint8) for cv2.imshow()."""
    img = to_hwc(img_any)
    H, W, C = img.shape

    # depth?
    if C == 1 and (img.dtype == np.uint16 or np.issubdtype(img.dtype, np.floating)):
        return colourise_depth(img)

    # grayscale
    if C == 1:
        view = img.astype(np.float32)
        if view.max() <= 1.0:
            view *= 255.0
        view = cv2.cvtColor(view.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(view, (WIN_W, WIN_H))

    # colour (RGB / RGBA)
    if C == 4:
        img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (WIN_W, WIN_H))

# ---------- gather datasets ---------------------------------------------------
def find_streams(h5):
    """
    Returns {name: (dataset, length)} for every image-like dataset.
    """
    out = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and is_image_like(obj.shape):
            shape = obj.shape
            if len(shape) == 4:                       # N,H,W,C  or  N,C,H,W
                length = shape[0]
            elif len(shape) == 3 and shape[0] not in (1, 3, 4):
                length = shape[0]                     # N,H,W
            else:
                length = 1                            # single image
            out[name] = (obj, length)
    h5.visititems(visitor)
    return out

# ---------- window tiling -----------------------------------------------------
def window_positions(n):
    """Return list[(x,y)] – top-left corner for each window so they tile."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pos = []
    for i in range(n):
        r, c = divmod(i, cols)
        pos.append((c * WIN_W, r * WIN_H))
    return pos

# ---------- visualise one HDF5 episode ---------------------------------------
def view_episode(path):
    print(f"\n▶ {os.path.basename(path)}")
    with h5py.File(path, "r") as h5:
        streams = find_streams(h5)
        if not streams:
            print("  (no image-like datasets found)")
            return

        # stable order: alphabetical
        streams = dict(sorted(streams.items()))
        longest = max(length for _, length in streams.values())
        print(f"  {len(streams)} streams | longest = {longest} frames")

        positions = window_positions(len(streams))
        frame_idx = 0

        while True:
            for i, (name, (ds, length)) in enumerate(streams.items()):
                if frame_idx >= length:
                    continue
                # fetch one frame (handles all layouts automatically)
                if len(ds.shape) == 4 and ds.shape[0] == length:
                    img = ds[frame_idx]
                elif len(ds.shape) == 3 and ds.shape[0] == length:
                    img = ds[frame_idx]
                else:
                    img = ds[()]

                view = to_display(img)
                win = f"{name}"
                cv2.imshow(win, view)
                cv2.moveWindow(win, *positions[i])

            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), 27):                    # q or Esc
                break
            if key in (ord('n'), ord(' '), 83, 0x27):    # next / →
                frame_idx = min(frame_idx + 1, longest - 1)
            elif key in (ord('p'), 81, 0x25):            # previous / ←
                frame_idx = max(frame_idx - 1, 0)

        cv2.destroyAllWindows()

# ---------- entry -------------------------------------------------------------
def main(root="demo_data"):
    root = sys.argv[1] if len(sys.argv) > 1 else root
    files = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    if not files:
        print(f"No HDF5 files found in '{root}'.")
        return
    for f in files:
        view_episode(f)


if __name__ == "__main__":
    main()
