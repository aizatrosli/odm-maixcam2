#!/usr/bin/env python3
"""Open Duck Mini v2 head, modified to carry a Sipeed MaixCAM2.

Starts from the upstream printable STLs (apirrone/Open_Duck_Mini, ``print/``) and
Sipeed's official MaixCAM2 STEP model, and writes printable STLs:

    stl/head_maixcam2.stl           replaces print/head.stl
    stl/head_bot_sheet_maixcam2.stl replaces print/head_bot_sheet.stl

What changes (all numbers are parameters below):

* The head is cut at x = STRETCH_AT, where its cross-section is a plain prism, and
  everything in front of the cut moves forward by EXTEND. The face, eyes, eye mounts
  and the bottom-sheet screw bosses keep their shape; only the walls get longer.
  This is needed because the board (21.2 mm deep) plus room for the header plugs does
  not fit between the stock face and the neck's head_roll_mount.
* The Raspberry Pi camera bosses behind the face are removed, and a tapered window is
  cut in the face in front of the lens, sized for the full 16:9 field of view.
* A slot in the top of the head reaches the board's power switch.
* The bottom sheet gets the same stretch, plus a cradle: a pad the board stands on,
  side and back fences that stop it turning, and a hole for a 1/4"-20 screw into the
  board's tripod nut. A slot under the TF card lets you swap the card from below.

Coordinates are the upstream assembly frame, in mm: x forward, y to the duck's left,
z up. The output STLs stay in that frame, like the upstream print files.

Run:  python build_head.py            (downloads the inputs into .cache/ on first run)
      python build_head.py --extend 20 --plug-gap 12
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"

UPSTREAM_COMMIT = "b23317a485b3cec7d8417f352478778b3475173c"
UPSTREAM_RAW = f"https://raw.githubusercontent.com/apirrone/Open_Duck_Mini/{UPSTREAM_COMMIT}/print/"
UPSTREAM_PARTS = [
    "head", "head_bot_sheet", "head_roll_mount", "head_yaw_to_roll", "head_pitch_to_yaw",
    "left_eye", "right_eye", "bulb", "speaker_stand", "flash_light_module",
    "flash_reflector_interface", "left_antenna_holder", "right_antenna_holder",
]
SIPEED_ZIP = ("https://dl.sipeed.com/fileDownload?verify_code=&file_url="
              "MaixCAM/MaixCAM2/Hardware/assemble_maixcam2/3D_MaixCAM2_2025-11-24.zip")
SIPEED_STEP = "MaixCAM2_3.0.step"

# --- MaixCAM2 geometry, measured from Sipeed's STEP (MaixCAM2_3.0.step, 2025-11-24) ---
# STEP frame: x across the board (+x = power-switch end of the top), y through it
# (-y = lens side, +y = screen side), z up (lens near the top edge).
M2_X = (-33.17, 33.17)          # case, without the small +x side bump (to 34.16 at z 7.9..16.1)
M2_Y = (-7.41, 13.84)           # front cover face .. screen glass
M2_Z = (-24.92, 25.94)          # bottom .. top of power switch
M2_LENS_XZ = (0.055, 14.945)    # optical axis
M2_LENS_FRONT_Y = -13.20        # front of the M12 lens
M2_TRIPOD_XY = (0.0, 4.7)       # 1/4"-20 nut in the bottom face
M2_TRIPOD_DEPTH = 4.56          # usable thread depth of that nut
M2_TF_X, M2_TF_Y = (12.8, 24.5), (0.7, 2.3)       # TF slot in the bottom face
M2_SWITCH_X, M2_SWITCH_Y = (-18.9, -8.7), (-5.3, 1.1)  # power switch on the top face
M2_HEADER_X, M2_HEADER_Z = (14.13, 29.91), (9.85, 17.25)  # both 2x6 headers, +-x
M2_GLASS_Z = (-23.36, 23.36)    # screen glass on the back face
# -x end: Type-C at y -4.6..-1.5 / z -14.8..-6, 6-pin 1.25 mm at y -6..2 / z -5..6

# --- upstream head geometry (assembly frame, mm) ---
SHEET_TOP, SHEET_BOTTOM = 112.71, 109.71
FACE_INNER_STOCK, FACE_OUTER_STOCK = 117.0, 120.0  # checked against the mesh in build()
ROLL_MOUNT_FRONT = 100.0                            # front of head_roll_mount (top of the head)
STOCK_FRONT_PARTS = ["left_eye", "right_eye", "bulb"]  # ride on the face, move with it

# Lens field of view (stock MaixCAM2 lens, 86.7 deg wide at 16:9)
HFOV_HALF_169 = 43.35
VFOV_HALF_169 = np.degrees(np.arctan(np.tan(np.radians(HFOV_HALF_169)) * 9 / 16))


@dataclass
class Params:
    stretch_at: float = 110.0   # x of the cut; head 107..113 and sheet 109..116 are plain prisms
    extend: float = 18.0        # how much longer the head gets
    plug_gap: float = 11.0      # board front cover -> inside of the face (header plugs live here)
    pad: float = 1.5            # board stands this high above the bottom sheet
    boss_below: float = 1.0     # boss under the sheet around the tripod screw
    tripod_hole: float = 6.6    # clearance for 1/4"-20
    fence_h: float = 6.0        # side fence height above the board's bottom
    fence_t: float = 2.0
    clear: float = 0.3          # board-to-fence clearance
    win_start: float = 12.0     # window size at the lens front (square)
    win_h_half: float = 45.0    # window half-angles, a little over the lens FOV
    win_v_half: float = 31.0
    switch_slot: bool = True


# ----------------------------------------------------------------------------- helpers
def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "odm-maixcam2-build"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        dest.write_bytes(data)
    return dest


def upstream(name: str) -> trimesh.Trimesh:
    return trimesh.load(fetch(UPSTREAM_RAW + name + ".stl", CACHE / "upstream" / f"{name}.stl"))


def maixcam2_meshes() -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """(full model, case body hull, lens hull) in the STEP frame. Tessellates once, then caches."""
    full_p = CACHE / "maixcam2_full.stl"
    if not full_p.exists():
        z = fetch(SIPEED_ZIP, CACHE / "sipeed" / "3D_MaixCAM2.zip")
        step = CACHE / "sipeed" / SIPEED_STEP
        if not step.exists():
            with zipfile.ZipFile(z) as zf:
                step.write_bytes(zf.read(SIPEED_STEP))
        import cadquery as cq  # only needed once, to tessellate the STEP

        print("tessellating the MaixCAM2 STEP (about a minute)")
        parts = []
        for s in cq.importers.importStep(str(step)).solids().vals():
            v, f = s.tessellate(0.2, 0.5)
            if f:
                parts.append(trimesh.Trimesh(np.array([(p.x, p.y, p.z) for p in v]), np.array(f)))
        trimesh.util.concatenate(parts).export(full_p)
    full = trimesh.load(full_p)
    v = full.vertices
    body = trimesh.convex.convex_hull(v[v[:, 1] >= M2_Y[0] - 0.05])
    lens = trimesh.convex.convex_hull(v[v[:, 1] < M2_Y[0] - 0.05])
    return full, body, lens


def to_manifold(m: trimesh.Trimesh) -> Manifold:
    out = Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    assert out.status().name == "NoError", out.status()
    return out


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def box(x0, x1, y0, y1, z0, z1) -> Manifold:
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl_z(cx, cy, z0, z1, d, n=64) -> Manifold:
    return Manifold.cylinder(z1 - z0, d / 2, d / 2, n).translate([cx, cy, z0])


def stretch(m: trimesh.Trimesh, at: float, by: float) -> trimesh.Trimesh:
    m = m.copy()
    v = m.vertices.copy()
    v[v[:, 0] > at, 0] += by
    m.vertices = v
    return m


# ------------------------------------------------------------------- board placement
@dataclass
class Placement:
    ox: float
    oy: float
    oz: float

    def pt(self, x, y, z):
        """STEP-frame point -> assembly frame. Lens looks along +x, top stays up."""
        return np.array([-y + self.ox, x + self.oy, z + self.oz])

    def X(self, y):
        return -y + self.ox

    def Y(self, x):
        return x + self.oy

    def Z(self, z):
        return z + self.oz

    def mesh(self, m: trimesh.Trimesh) -> trimesh.Trimesh:
        m = m.copy()
        v = m.vertices
        m.vertices = np.column_stack([-v[:, 1] + self.ox, v[:, 0] + self.oy, v[:, 2] + self.oz])
        return m  # a proper rotation, so the faces keep their winding


def place(p: Params, face_inner: float) -> Placement:
    return Placement(
        ox=face_inner - p.plug_gap + M2_Y[0],   # front cover sits plug_gap behind the face
        oy=-M2_LENS_XZ[0],                       # lens on the centre line
        oz=SHEET_TOP + p.pad - M2_Z[0],          # board bottom on the pad
    )


# ----------------------------------------------------------------------------- parts
def build_head(p: Params, head: trimesh.Trimesh, pl: Placement, face_inner: float, face_outer: float) -> Manifold:
    h = to_manifold(stretch(head, p.stretch_at, p.extend))
    lens_x = pl.X(M2_LENS_FRONT_Y)
    lz = pl.Z(M2_LENS_XZ[1])
    # Pi camera bosses (21 x 12.5 mm pattern behind the old lens hole) and anything else
    # left inside the face between the eyes, above the eye centres
    h = h - box(face_inner - 8, face_inner - 0.02, -17.0, 5.0, 136.0, 165.5)
    # tapered window: from a square at the lens front out through the face
    s = p.win_start / 2
    d = face_outer + 1.0 - lens_x
    hw = s + d * np.tan(np.radians(p.win_h_half))
    vw = s + d * np.tan(np.radians(p.win_v_half))
    pts = [(lens_x, y, z) for y in (-s, s) for z in (lz - s, lz + s)]
    pts += [(face_outer + 1.0, y, z) for y in (-hw, hw) for z in (lz - vw, lz + vw)]
    h = h - Manifold.hull_points(np.array(pts, np.float64))
    if p.switch_slot:
        y0, y1 = sorted((pl.Y(M2_SWITCH_X[0]), pl.Y(M2_SWITCH_X[1])))
        x0, x1 = sorted((pl.X(M2_SWITCH_Y[0]), pl.X(M2_SWITCH_Y[1])))
        h = h - box(x0 - 1.0, x1 + 1.0, y0 - 1.0, y1 + 1.0, pl.Z(M2_Z[1]) + 0.2, 180.0)
    return h


def build_sheet(p: Params, sheet: trimesh.Trimesh, pl: Placement) -> Manifold:
    s = to_manifold(stretch(sheet, p.stretch_at, p.extend))
    xf, xb = pl.X(M2_Y[0]), pl.X(M2_Y[1])        # board front / back in x
    yl, yr = pl.Y(M2_X[1]), pl.Y(M2_X[0])        # board left (+y) / right (-y) side
    zb = pl.Z(M2_Z[0])                           # board bottom
    c, t = p.clear, p.fence_t
    add = [
        box(xb + 1.0, xf - 1.0, yr + 2.0, yl - 2.0, SHEET_TOP - 0.5, zb),  # pad
        # side fences on the rear half (the Type-C / 1.25 mm ports are on the front half of the -y side)
        box(pl.X(13.84), pl.X(3.0), yl + c, yl + c + t, SHEET_TOP - 0.5, zb + p.fence_h),
        box(pl.X(13.84), pl.X(3.0), yr - c - t, yr - c, SHEET_TOP - 0.5, zb + p.fence_h),
        # back stop: stays below the screen glass
        box(xb - c - t, xb - c, -20.0, 20.0, SHEET_TOP - 0.5, zb + (M2_GLASS_Z[0] - M2_Z[0]) - 0.2),
        # front stop, low and between the microphones
        box(xf + c, xf + c + t, -12.0, 12.0, SHEET_TOP - 0.5, zb + 2.0),
    ]
    tx, ty = pl.X(M2_TRIPOD_XY[1]), pl.Y(M2_TRIPOD_XY[0])
    add.append(cyl_z(tx, ty, SHEET_BOTTOM - p.boss_below, SHEET_BOTTOM + 0.5, 12.0))
    for a in add:
        s = s + a
    s = s - cyl_z(tx, ty, SHEET_BOTTOM - p.boss_below - 1, zb + 1, p.tripod_hole)
    # TF card access from below
    y0, y1 = sorted((pl.Y(M2_TF_X[0]), pl.Y(M2_TF_X[1])))
    x0, x1 = sorted((pl.X(M2_TF_Y[0]), pl.X(M2_TF_Y[1])))
    s = s - box(x0 - 2.0, x1 + 2.0, y0 - 1.0, y1 + 1.0, SHEET_BOTTOM - 5, zb + 0.5)
    return s


# ---------------------------------------------------------------------------- checks
def ray_first(m: trimesh.Trimesh, origin, direction) -> float:
    loc, _, _ = m.ray.intersects_location([origin], [direction])
    if len(loc) == 0:
        return np.inf
    return float(np.min((loc - origin) @ np.asarray(direction, float)))


def fov_blocked(meshes: list[trimesh.Trimesh], origin, h_half, v_half, n=41) -> float:
    dirs = []
    for a in np.linspace(-h_half, h_half, n):
        for b in np.linspace(-v_half, v_half, n):
            d = np.array([1.0, np.tan(np.radians(a)), np.tan(np.radians(b))])
            dirs.append(d / np.linalg.norm(d))
    dirs = np.array(dirs)
    hit = np.zeros(len(dirs), bool)
    for m in meshes:
        idx = m.ray.intersects_any(np.repeat([origin], len(dirs), 0), dirs)
        hit |= idx
    return float(hit.mean())


def overlap(a: Manifold, b: Manifold) -> float:
    return (a ^ b).volume()


# ------------------------------------------------------------------------------ main
def build(p: Params, out: Path, check: bool = True, previews: bool = True) -> dict:
    parts = {n: upstream(n) for n in UPSTREAM_PARTS}
    head = parts["head"]
    # the face really is where we think it is
    fi = ray_first(head, [100.0, 0.0, 125.0], [1, 0, 0]) + 100.0
    assert abs(fi - FACE_INNER_STOCK) < 0.3, fi
    face_inner, face_outer = FACE_INNER_STOCK + p.extend, FACE_OUTER_STOCK + p.extend
    pl = place(p, face_inner)

    head_m = build_head(p, head, pl, face_inner, face_outer)
    sheet_m = build_sheet(p, parts["head_bot_sheet"], pl)
    out.mkdir(parents=True, exist_ok=True)
    head_t, sheet_t = to_trimesh(head_m), to_trimesh(sheet_m)
    head_t.export(out / "head_maixcam2.stl")
    sheet_t.export(out / "head_bot_sheet_maixcam2.stl")

    report: dict = {"params": asdict(p)}
    report["board_in_head"] = {
        "front_cover_x": round(pl.X(M2_Y[0]), 2), "screen_x": round(pl.X(M2_Y[1]), 2),
        "bottom_z": round(pl.Z(M2_Z[0]), 2), "top_z": round(pl.Z(M2_Z[1]), 2),
        "lens_front": pl.pt(M2_LENS_XZ[0], M2_LENS_FRONT_Y, M2_LENS_XZ[1]).round(2).tolist(),
        "clearance_to_roll_mount_mm": round(pl.X(M2_Y[1]) - ROLL_MOUNT_FRONT, 2),
        "tripod_screw_engagement_mm": round(9.525 - (SHEET_TOP + p.pad - (SHEET_BOTTOM - p.boss_below)), 2),
        "tripod_nut_depth_mm": M2_TRIPOD_DEPTH,
    }
    report["parts"] = {
        n: {"watertight": bool(t.is_watertight), "volume_cm3": round(t.volume / 1000, 1),
            "stock_volume_cm3": round(parts[s].volume / 1000, 1),
            "pla_g_solid": round(t.volume / 1000 * 1.24, 1)}
        for n, t, s in [("head_maixcam2", head_t, "head"), ("head_bot_sheet_maixcam2", sheet_t, "head_bot_sheet")]
    }
    if not check:
        return report

    full, body, lens = maixcam2_meshes()
    body_w, lens_w, full_w = pl.mesh(body), pl.mesh(lens), pl.mesh(full)
    ceiling = ray_first(head_t, [pl.X(3.0), 0.0, 140.0], [0, 0, 1]) + 140.0
    report["board_in_head"]["ceiling_z"] = round(ceiling, 2)
    report["board_in_head"]["top_clearance_mm"] = round(ceiling - pl.Z(M2_Z[1]), 2)

    others = {"head (new)": head_m, "bottom sheet (new)": sheet_m}
    for n in ["head_roll_mount", "head_yaw_to_roll", "head_pitch_to_yaw", "speaker_stand",
              "flash_light_module", "flash_reflector_interface"]:
        others[n] = to_manifold(parts[n])
    for n in STOCK_FRONT_PARTS:
        others[n + " (moved)"] = to_manifold(parts[n].copy().apply_translation([p.extend, 0, 0]))
    # header plugs: right-angle Dupont, `plug_gap - 3` deep is what they get before the eye mounts
    plugs = None
    for sx in (-1, 1):
        xs = sorted((sx * M2_HEADER_X[0], sx * M2_HEADER_X[1]))
        a, b = pl.pt(xs[0] - 0.5, M2_Y[0] - 0.01, M2_HEADER_Z[0] - 0.5), pl.pt(xs[1] + 0.5, M2_Y[0] - 8.0, M2_HEADER_Z[1] + 0.5)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        bx = box(*[v for pair in zip(lo, hi) for v in pair])
        plugs = bx if plugs is None else plugs + bx
    body_m, lens_m = to_manifold(body_w), to_manifold(lens_w)
    report["overlap_mm3"] = {
        n: {"board": round(overlap(body_m, o), 2), "lens": round(overlap(lens_m, o), 2),
            "header_plugs_8mm": round(overlap(plugs, o), 2)}
        for n, o in others.items()
    }
    eyes = [parts[n].copy().apply_translation([p.extend, 0, 0]) for n in STOCK_FRONT_PARTS]
    origin = pl.pt(M2_LENS_XZ[0], M2_LENS_FRONT_Y + 0.5, M2_LENS_XZ[1])
    report["fov_blocked_fraction"] = {
        "16:9 (86.7 x 55.8 deg)": round(fov_blocked([head_t] + eyes, origin, HFOV_HALF_169, VFOV_HALF_169), 4),
        "4:3 (70 x 55.8 deg)": round(fov_blocked([head_t] + eyes, origin, 35.2, VFOV_HALF_169), 4),
    }
    if previews:
        import render
        flash = [parts["flash_light_module"], parts["flash_reflector_interface"]]
        render.previews(HERE / "images", head_t, sheet_t, full_w, eyes, flash,
                        [parts["head_roll_mount"], parts["head_yaw_to_roll"]], pl)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Params()
    for k, v in asdict(d).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--out", type=Path, default=HERE / "stl")
    ap.add_argument("--no-check", action="store_true", help="skip the MaixCAM2 fit checks (no cadquery needed)")
    ap.add_argument("--no-previews", action="store_true")
    a = ap.parse_args(argv)
    p = Params(**{k: getattr(a, k) for k in asdict(d)})
    rep = build(p, a.out, check=not a.no_check, previews=not a.no_previews)
    txt = json.dumps(rep, indent=2)
    print(txt)
    (a.out / "build_report.json").write_text(txt + "\n")
    bad = [n for n, v in rep.get("overlap_mm3", {}).items() if max(v.values()) > 1.0]
    if bad:
        print("WARNING: overlaps with", bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
