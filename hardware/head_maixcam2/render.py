"""Small software renderer for the preview images (no GPU or display needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

GREY, DARK, BOARD, EYE, FLASH, NECK, SHEET = (205, 205, 200), (90, 90, 95), (60, 62, 70), (70, 110, 200), (220, 190, 60), (190, 110, 80), (120, 170, 140)


def _basis(az, el):
    a, e = np.radians(az), np.radians(el)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0, 0, 1.0]) if abs(el) < 89 else np.array([-np.cos(a), -np.sin(a), 0])
    r = np.cross(up, d)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(d, r), d])


def render(parts, az, el, W=1000, H=760, label=None):
    """Orthographic z-buffer render of [(mesh, rgb)], camera at azimuth/elevation (deg)."""
    R = _basis(az, el)
    pv = np.vstack([m.vertices for m, _ in parts]) @ R.T
    lo, hi = pv.min(0), pv.max(0)
    c = (lo + hi) / 2
    sc = 0.9 * min(W / (hi[0] - lo[0]), H / (hi[1] - lo[1]))
    zb = np.full((H, W), -np.inf)
    img = np.ones((H, W, 3))
    light = np.array([0.35, 0.45, 0.82])
    for m, col in parts:
        col = np.array(col) / 255
        V = m.vertices @ R.T
        X, Y, Z = (V[:, 0] - c[0]) * sc + W / 2, H / 2 - (V[:, 1] - c[1]) * sc, V[:, 2]
        sh = 0.3 + 0.7 * np.abs((m.face_normals @ R.T) @ light)
        for fi, (i, j, k) in enumerate(m.faces):
            xs, ys, zs = X[[i, j, k]], Y[[i, j, k]], Z[[i, j, k]]
            x0, x1 = int(max(np.floor(xs.min()), 0)), int(min(np.ceil(xs.max()), W - 1))
            y0, y1 = int(max(np.floor(ys.min()), 0)), int(min(np.ceil(ys.max()), H - 1))
            if x1 < x0 or y1 < y0:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            w2 = 1 - w0 - w1
            ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not ins.any():
                continue
            z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
            sub = zb[y0:y1 + 1, x0:x1 + 1]
            upd = ins & (z > sub)
            sub[upd] = z[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = col * sh[fi]
    # outlines where depth jumps
    zz = np.where(np.isinf(zb), -1e9, zb)
    edge = np.zeros_like(zb, bool)
    for ax in (0, 1):
        edge |= np.abs(np.diff(zz, axis=ax, prepend=zz[:1] if ax == 0 else zz[:, :1])) > 1.5
    img[edge] *= 0.3
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    if label:
        ImageDraw.Draw(im).text((12, 10), label, fill=(0, 0, 0))
    return im


def sheet_of(images, cols, out: Path):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    g = Image.new("RGB", (w * cols, h * rows), "white")
    for k, im in enumerate(images):
        g.paste(im, ((k % cols) * w, (k // cols) * h))
    g.save(out, optimize=True)


def _cut(m: trimesh.Trimesh, keep_y_below: float) -> trimesh.Trimesh:
    from build_head import to_manifold, to_trimesh

    return to_trimesh(to_manifold(m).trim_by_plane([0, -1, 0], -keep_y_below))


def previews(out: Path, head, sheet, board, eyes, flash, neck, pl):
    out.mkdir(parents=True, exist_ok=True)
    print("rendering previews")
    assembled = [(head, GREY), (sheet, SHEET), (board, BOARD)] + [(e, EYE) for e in eyes] + [(f, FLASH) for f in flash]
    sheet_of([
        render(assembled, 25, 18, label="front, 3/4"),
        render(assembled, 0, 4, label="face"),
    ], 2, out / "head_front.png")
    cut = [(_cut(head, 0.0), GREY), (_cut(sheet, 0.0), SHEET), (board, BOARD)] + [(n, NECK) for n in neck] \
        + [(_cut(e, 0.0), EYE) for e in eyes]
    sheet_of([
        render(cut, 90, 8, label="cut at the centre line, seen from the left (y < 0 half)"),
        render(cut, 60, 35, label="same cut, from above"),
    ], 2, out / "head_cutaway.png")
    sheet_of([
        render([(sheet, SHEET)], 30, 40, label="bottom sheet with cradle"),
        render([(sheet, SHEET), (board, BOARD)], 30, 40, label="with the MaixCAM2"),
        render([(sheet, SHEET)], 20, -45, label="underside: tripod screw boss, TF card slot"),
    ], 3, out / "bottom_sheet.png")
