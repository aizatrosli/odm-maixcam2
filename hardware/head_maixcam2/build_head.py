#!/usr/bin/env python3
"""Open Duck Mini v2 head, modified to carry a Sipeed MaixCAM2.

Starts from the upstream printable STLs (apirrone/Open_Duck_Mini, ``print/``) and
Sipeed's official MaixCAM2 STEP model, and writes printable STLs for two layouts:

screen_face (default)
    The MaixCAM2 faces backward, so its screen is the duck's face, seen through a
    window that replaces the LED eyes. The camera module (sensor board + M12 lens) is
    taken out of the case and sits in a pod on top of the head, looking forward, on a
    longer 22-pin flat cable.

        stl/screen_face/head_maixcam2_screen.stl       replaces print/head.stl
        stl/screen_face/head_bot_sheet_maixcam2.stl    replaces print/head_bot_sheet.stl
        stl/screen_face/camera_pod_cap.stl             new, closes the back of the pod

case_closed
    The MaixCAM2 stays closed, lens forward through a window between the stock eyes.
    The screen faces into the head.

        stl/case_closed/head_maixcam2.stl              replaces print/head.stl
        stl/case_closed/head_bot_sheet_maixcam2.stl    replaces print/head_bot_sheet.stl

Both layouts:

* Cut the head at x = stretch_at, where its cross-section is a plain prism, and move
  everything in front of the cut forward by `extend`. Only the walls get longer. The
  board (21.2 mm deep) plus its header plugs do not fit between the stock face and the
  neck's head_roll_mount otherwise.
* Cut a slot in the top of the head over the board's power switch.
* Stretch the bottom sheet the same way and add a cradle: a pad the board stands on,
  fences that stop it turning, a hole for a 1/4"-20 screw into the board's tripod nut,
  and a slot under the TF card so the card can be swapped from below.

Coordinates are the upstream assembly frame, in mm: x forward, y to the duck's left,
z up. The output STLs stay in that frame, like the upstream print files.

Run:  python build_head.py                       (downloads inputs into .cache/ on first run)
      python build_head.py --layout case_closed
      python build_head.py --extend 14 --lip 1.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
M2_LENS_HOLE_D = 15.1           # hole in the front cover around the lens (main PCB hole is ~16)
M2_TRIPOD_XY = (0.0, 4.7)       # 1/4"-20 nut in the bottom face
M2_TRIPOD_DEPTH = 4.56          # usable thread depth of that nut
M2_TF_X, M2_TF_Y = (12.8, 24.5), (0.7, 2.3)       # TF slot in the bottom face
M2_SWITCH_X, M2_SWITCH_Y = (-18.9, -8.7), (-5.3, 1.1)  # power switch on the top face
M2_HEADER_X, M2_HEADER_Z = (14.13, 29.91), (9.85, 17.25)  # both 2x6 headers, +-x, open toward -y
M2_GLASS_Z = (-23.36, 23.36)    # screen cover glass (63.3 x 46.8), centred on x = z = 0
M2_ACTIVE = (48.96, 36.72)      # 2.4" 640x480 active area (assumed centred on the glass)
# -x end: Type-C at y -4.6..-1.5 / z -14.8..-6, 6-pin 1.25 mm at y -6..2 / z -5..6
# Camera module = sensor board + M12 lens holder + lens (solids 1, 2, 3, 23, 24, 95, 341).
# It sits behind the main PCB, lens through a hole in the PCB and the front cover, and
# plugs into the main board's 22-pin FPC5 (x -7.3..7.3, z -16.7..-11.4, on the PCB's back).
CAM_SOLIDS = (1, 2, 3, 23, 24, 95, 341)
CAM_X = (-11.99, 11.97)
CAM_Y = (-12.75, 6.64)          # lens front .. back of its FPC connector
CAM_Z = (6.65, 23.27)
CAM_BOARD_BACK_Y = 4.56         # back face of the sensor board
CAM_CONN_X = (-7.63, 7.67)      # FPC connector on the back of the sensor board

# --- upstream head geometry (assembly frame, mm) ---
SHEET_TOP, SHEET_BOTTOM = 112.71, 109.71
FACE_INNER_STOCK, FACE_OUTER_STOCK = 117.0, 120.0  # checked against the mesh in build()
ROLL_MOUNT_FRONT = 100.0                            # front of head_roll_mount (top of the head)
STOCK_FRONT_PARTS = ["left_eye", "right_eye", "bulb"]  # ride on the face, move with it

# Lens field of view (stock MaixCAM2 lens, 86.7 deg wide at 16:9)
HFOV_HALF_169 = 43.35
VFOV_HALF_169 = float(np.degrees(np.arctan(np.tan(np.radians(HFOV_HALF_169)) * 9 / 16)))

LAYOUTS = ("screen_face", "case_closed")


@dataclass
class Params:
    layout: str = "screen_face"
    stretch_at: float = 110.0   # x of the cut; head 107..113 and sheet 109..116 are plain prisms
    extend: float = -1.0        # how much longer the head gets; -1 = layout default (12 / 18)
    plug_gap: float = 11.0      # case_closed: front cover -> inside of the face (header plugs)
    plug_depth: float = 8.0     # room the header plugs need (right-angle Dupont)
    lip: float = 1.0            # screen_face: face lip in front of the screen glass
    lip_window: tuple = (53.0, 40.0)  # screen_face: visible opening, over the 49 x 37 active area
    pad: float = 1.5            # board stands this high above the bottom sheet
    boss_below: float = 1.0     # boss under the sheet around the tripod screw
    tripod_hole: float = 6.6    # clearance for 1/4"-20
    fence_h: float = 6.0        # side fence height above the board's bottom
    fence_t: float = 2.0
    clear: float = 0.3          # board-to-fence / board-to-pocket clearance
    pod_wall: float = 2.0       # screen_face: camera pod walls
    pod_fpc_room: float = 4.0   # screen_face: space behind the camera's connector for the cable bend
    win_start: float = 12.0     # lens window size at the lens front (square)
    win_h_half: float = 45.0    # lens window half-angles, a little over the lens FOV
    win_v_half: float = 31.0
    switch_slot: bool = True

    def ext(self) -> float:
        if self.extend >= 0:
            return self.extend
        return 12.0 if self.layout == "screen_face" else 18.0


# ----------------------------------------------------------------------------- helpers
def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "odm-maixcam2-build"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except OSError as e:  # dl.sipeed.com sometimes answers 502; it works on a retry
                if attempt == 3:
                    raise SystemExit(f"could not download {url}: {e}. Download it by hand to {dest}")
                time.sleep(2 ** (attempt + 1))
        dest.write_bytes(data)
    return dest


def upstream(name: str) -> trimesh.Trimesh:
    return trimesh.load(fetch(UPSTREAM_RAW + name + ".stl", CACHE / "upstream" / f"{name}.stl"))


def maixcam2_meshes() -> dict:
    """MaixCAM2 meshes in the STEP frame: full, case (no camera module), cam (camera module),
    and convex hulls for the fit checks. Tessellates the STEP once, then uses the cache."""
    paths = {k: CACHE / f"maixcam2_{k}.stl" for k in ("full", "case", "cam")}
    if not all(p.exists() for p in paths.values()):
        z = fetch(SIPEED_ZIP, CACHE / "sipeed" / "3D_MaixCAM2.zip")
        step = CACHE / "sipeed" / SIPEED_STEP
        if not step.exists():
            with zipfile.ZipFile(z) as zf:
                step.write_bytes(zf.read(SIPEED_STEP))
        import cadquery as cq  # only needed once, to tessellate the STEP

        print("tessellating the MaixCAM2 STEP (about a minute)")
        case, cam = [], []
        for i, s in enumerate(cq.importers.importStep(str(step)).solids().vals()):
            v, f = s.tessellate(0.2, 0.5)
            if f:
                m = trimesh.Trimesh(np.array([(p.x, p.y, p.z) for p in v]), np.array(f))
                (cam if i in CAM_SOLIDS else case).append(m)
        trimesh.util.concatenate(case + cam).export(paths["full"])
        trimesh.util.concatenate(case).export(paths["case"])
        trimesh.util.concatenate(cam).export(paths["cam"])
    out = {k: trimesh.load(p) for k, p in paths.items()}
    v = out["full"].vertices
    out["body_hull"] = trimesh.convex.convex_hull(v[v[:, 1] >= M2_Y[0] - 0.05])
    out["lens_hull"] = trimesh.convex.convex_hull(v[v[:, 1] < M2_Y[0] - 0.05])
    out["case_hull"] = trimesh.convex.convex_hull(out["case"].vertices)
    cv = out["cam"].vertices  # two hulls: board + lens holder, and the connector on its back
    out["cam_hull"] = trimesh.util.concatenate([
        trimesh.convex.convex_hull(cv[cv[:, 1] <= CAM_BOARD_BACK_Y + 0.05]),
        trimesh.convex.convex_hull(cv[cv[:, 1] > CAM_BOARD_BACK_Y + 0.05])])
    return out


def to_manifold(m: trimesh.Trimesh) -> Manifold:
    out = Manifold(Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
    assert out.status().name == "NoError", out.status()
    return out


def to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=True)


def box(x0, x1, y0, y1, z0, z1) -> Manifold:
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    z0, z1 = sorted((z0, z1))
    return Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])


def cyl(axis: str, c, a0, a1, d, n=48) -> Manifold:
    """Cylinder along `axis` ('x' or 'z') through point c (2 coords of the other axes), from a0 to a1."""
    m = Manifold.cylinder(a1 - a0, d / 2, d / 2, n)
    if axis == "z":
        return m.translate([c[0], c[1], a0])
    return m.rotate([0, 90, 0]).translate([a0, c[0], c[1]])


def rounded_rect_x(x0, x1, cy, cz, w, h, r) -> Manifold:
    """Box along x with rounded corners in the y-z plane."""
    r = min(r, w / 2 - 0.01, h / 2 - 0.01)
    pts = []
    for sy, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        pts += [(x, cy + sy * (w / 2 - r), cz + sz * (h / 2 - r)) for x in (x0, x1)]
    core = Manifold.hull_points(np.array(pts))
    rods = [cyl("x", (cy + sy * (w / 2 - r), cz + sz * (h / 2 - r)), x0, x1, 2 * r, 32)
            for sy in (1, -1) for sz in (1, -1)]
    return Manifold.batch_hull(rods + [core])


def span(a, b, margin=0.0):
    lo, hi = sorted((a, b))
    return lo - margin, hi + margin


def stretch(m: trimesh.Trimesh, at: float, by: float) -> trimesh.Trimesh:
    m = m.copy()
    v = m.vertices.copy()
    v[v[:, 0] > at, 0] += by
    m.vertices = v
    return m


def window_frustum(x_start, x_end, cy, cz, start, h_half, v_half) -> Manifold:
    s = start / 2
    d = x_end - x_start
    hw = s + d * np.tan(np.radians(h_half))
    vw = s + d * np.tan(np.radians(v_half))
    pts = [(x_start, cy + y, cz + z) for y in (-s, s) for z in (-s, s)]
    pts += [(x_end, cy + y, cz + z) for y in (-hw, hw) for z in (-vw, vw)]
    return Manifold.hull_points(np.array(pts, np.float64))


# ------------------------------------------------------------------- board placement
@dataclass
class Placement:
    """STEP frame -> assembly frame. s = +1: lens side forward. s = -1: screen side forward.
    Either way the top stays up, so neither the screen nor the camera image turns over."""
    ox: float
    oy: float
    oz: float
    s: int = 1

    def pt(self, x, y, z):
        return np.array([self.X(y), self.Y(x), self.Z(z)])

    def X(self, y):
        return -self.s * y + self.ox

    def Y(self, x):
        return self.s * x + self.oy

    def Z(self, z):
        return z + self.oz

    def mesh(self, m: trimesh.Trimesh) -> trimesh.Trimesh:
        m = m.copy()
        v = m.vertices
        m.vertices = np.column_stack([-self.s * v[:, 1] + self.ox, self.s * v[:, 0] + self.oy, v[:, 2] + self.oz])
        return m  # a proper rotation, so the faces keep their winding


def place_board(p: Params, face_inner: float, face_outer: float) -> Placement:
    oz = SHEET_TOP + p.pad - M2_Z[0]          # board bottom on the pad
    if p.layout == "case_closed":
        # front cover sits plug_gap behind the face, lens on the centre line
        return Placement(face_inner - p.plug_gap + M2_Y[0], -M2_LENS_XZ[0], oz, 1)
    # screen glass sits right behind the lip of the face
    return Placement(face_outer - p.lip - M2_Y[1], 0.0, oz, -1)


def place_camera(p: Params, face_outer: float, top_z: float) -> Placement:
    """Camera module in the pod on top of the head, lens 0.4 mm behind the pod's front wall."""
    lens_x = face_outer - p.pod_wall - 0.4
    return Placement(lens_x + CAM_Y[0], -M2_LENS_XZ[0], top_z + 0.3 - CAM_Z[0], 1)


# ----------------------------------------------------------------------------- parts
def head_case_closed(p, h: Manifold, pl: Placement, face_inner, face_outer) -> Manifold:
    # Pi camera bosses (21 x 12.5 mm pattern behind the old lens hole) and anything else
    # left inside the face between the eyes, above the eye centres
    h = h - box(face_inner - 8, face_inner - 0.02, -17.0, 5.0, 136.0, 165.5)
    lens_x = pl.X(M2_LENS_FRONT_Y)
    return h - window_frustum(lens_x, face_outer + 1.0, pl.Y(M2_LENS_XZ[0]), pl.Z(M2_LENS_XZ[1]),
                              p.win_start, p.win_h_half, p.win_v_half)


def head_screen_face(p, h: Manifold, head_t: trimesh.Trimesh, pl: Placement, face_inner, face_outer,
                     ceiling) -> Manifold:
    # Replace the face: drop the eye mounts and Pi camera bosses, then put back a plain
    # plate with the face's own outline (convex hull of the face wall).
    v = head_t.vertices
    face_v = v[(v[:, 0] >= face_inner - 0.01) & (v[:, 0] <= face_outer + 0.01)]
    plate = Manifold.hull_points(face_v.astype(np.float64))
    h = (h - box(face_inner - 4.0, face_outer + 12.0, -59.0, 59.0, SHEET_BOTTOM - 0.1, ceiling - 0.05)) + plate
    # screen: a pocket the case pushes into from behind, and a lip in front of the glass
    cy, cz = pl.Y(0.0), pl.Z(0.0)
    w = M2_X[1] - M2_X[0] + 2 * p.clear
    z0, z1 = pl.Z(M2_Z[0]) - p.clear, pl.Z(M2_Z[1]) + p.clear
    h = h - box(face_inner - 1.0, face_outer - p.lip, cy - w / 2, cy + w / 2, z0, z1)
    h = h - rounded_rect_x(face_outer - p.lip - 0.5, face_outer + 1.0, cy, cz, p.lip_window[0], p.lip_window[1], 2.0)
    return h


def camera_pod(p, cam: Placement, face_outer: float, top_z: float):
    """(pod body to add to the head, cuts to take out of the head, cap part)."""
    t = p.pod_wall
    lens_x = cam.X(CAM_Y[0])
    conn_x = cam.X(CAM_Y[1])
    cx0 = conn_x - p.pod_fpc_room           # back of the cavity, closed by the cap
    hw = (CAM_X[1] - CAM_X[0]) / 2 + p.clear
    cz1 = cam.Z(CAM_Z[1]) + p.clear
    cy = cam.Y(0.0)
    body = box(cx0, face_outer, cy - hw - t, cy + hw + t, top_z - 1.5, cz1 + t)
    ear_y = hw + t + 2.5
    for sy in (1, -1):
        body = body + box(cx0, cx0 + 6.0, cy + sy * (hw + t - 0.5), cy + sy * (hw + t + 5.0), top_z - 1.5, top_z + 8.0)
    cuts = box(cx0 - 1.0, lens_x + 0.4, cy - hw, cy + hw, top_z, cz1)
    cuts = cuts + window_frustum(lens_x, face_outer + 1.0, cy, cam.Z(M2_LENS_XZ[1]), p.win_start, p.win_h_half, p.win_v_half)
    # the flat cable goes down into the head just behind the connector
    cuts = cuts + box(cx0 + 0.3, conn_x - 0.3, cy - 8.0, cy + 8.0, top_z - 8.0, top_z + 0.5)
    for sy in (1, -1):
        cuts = cuts + cyl("x", (cy + sy * ear_y, top_z + 4.0), cx0 - 1.0, cx0 + 5.0, 1.7)  # M2 self-tapping
    # cap: plate over the back, two prongs that hold the sensor board against the pod front
    cap = box(cx0 - 2.0, cx0, cy - hw - t, cy + hw + t, top_z, cz1 + t)
    for sy in (1, -1):
        cap = cap + box(cx0 - 2.0, cx0, cy + sy * (hw + t - 0.5), cy + sy * (hw + t + 5.0), top_z, top_z + 8.0)
    board_back = cam.X(CAM_BOARD_BACK_Y)
    for sy in (1, -1):
        cap = cap + box(cx0 - 0.5, board_back - 0.1, cy + sy * 8.6, cy + sy * 11.4, cam.Z(CAM_Z[0]) + 1.0, cam.Z(CAM_Z[1]) - 1.0)
        cap = cap - cyl("x", (cy + sy * ear_y, top_z + 4.0), cx0 - 3.0, cx0 + 1.0, 2.3)
    return body, cuts, cap


def build_sheet(p: Params, sheet: trimesh.Trimesh, pl: Placement) -> Manifold:
    s = to_manifold(stretch(sheet, p.stretch_at, p.ext()))
    xs = sorted((pl.X(M2_Y[0]), pl.X(M2_Y[1])))           # board extent in x
    ys = sorted((pl.Y(M2_X[0]), pl.Y(M2_X[1])))
    zb = pl.Z(M2_Z[0])                                     # board bottom
    c, t = p.clear, p.fence_t
    add = [box(xs[0] + 1.0, xs[1] - 1.0, ys[0] + 2.0, ys[1] - 2.0, SHEET_TOP - 0.5, zb)]  # pad
    # side fences on the screen half (the Type-C / 1.25 mm ports are on the cover half of one side)
    fx = (pl.X(13.84), pl.X(3.0))
    add += [box(*fx, ys[1] + c, ys[1] + c + t, SHEET_TOP - 0.5, zb + p.fence_h),
            box(*fx, ys[0] - c - t, ys[0] - c, SHEET_TOP - 0.5, zb + p.fence_h)]
    # low stop at the front cover, between the microphones
    xc = pl.X(M2_Y[0])
    add.append(box(xc + pl.s * c, xc + pl.s * (c + t), -12.0, 12.0, SHEET_TOP - 0.5, zb + 2.0))
    if p.layout == "case_closed":
        # stop at the screen side, below the glass (in screen_face the face itself holds it)
        xg = pl.X(M2_Y[1])
        add.append(box(xg - c, xg - c - t, -20.0, 20.0, SHEET_TOP - 0.5, zb + (M2_GLASS_Z[0] - M2_Z[0]) - 0.2))
    tx, ty = pl.X(M2_TRIPOD_XY[1]), pl.Y(M2_TRIPOD_XY[0])
    add.append(cyl("z", (tx, ty), SHEET_BOTTOM - p.boss_below, SHEET_BOTTOM + 0.5, 12.0))
    for a in add:
        s = s + a
    s = s - cyl("z", (tx, ty), SHEET_BOTTOM - p.boss_below - 1, zb + 1, p.tripod_hole)
    # TF card access from below
    s = s - box(*span(pl.X(M2_TF_Y[0]), pl.X(M2_TF_Y[1]), 2.0),
                *span(pl.Y(M2_TF_X[0]), pl.Y(M2_TF_X[1]), 1.0), SHEET_BOTTOM - 5, zb + 0.5)
    return s


# ---------------------------------------------------------------------------- checks
def ray_first(m: trimesh.Trimesh, origin, direction) -> float:
    loc, _, _ = m.ray.intersects_location([origin], [direction])
    if len(loc) == 0:
        return np.inf
    return float(np.min((loc - origin) @ np.asarray(direction, float)))


def fov_blocked(meshes: list, origin, h_half, v_half, n=41) -> float:
    dirs = []
    for a in np.linspace(-h_half, h_half, n):
        for b in np.linspace(-v_half, v_half, n):
            d = np.array([1.0, np.tan(np.radians(a)), np.tan(np.radians(b))])
            dirs.append(d / np.linalg.norm(d))
    dirs = np.array(dirs)
    hit = np.zeros(len(dirs), bool)
    for m in meshes:
        hit |= m.ray.intersects_any(np.repeat([origin], len(dirs), 0), dirs)
    return float(hit.mean())


def overlap(a: Manifold, b: Manifold) -> float:
    return (a ^ b).volume()


def header_plugs(pl: Placement, depth: float) -> Manifold:
    """Space in front of both 2x6 headers that the plugs and their wires take."""
    out = None
    for sx in (-1, 1):
        xs = sorted((sx * M2_HEADER_X[0], sx * M2_HEADER_X[1]))
        a = pl.pt(xs[0] - 0.5, M2_Y[0] - 0.01, M2_HEADER_Z[0] - 0.5)
        b = pl.pt(xs[1] + 0.5, M2_Y[0] - depth, M2_HEADER_Z[1] + 0.5)
        bx = box(a[0], b[0], a[1], b[1], a[2], b[2])
        out = bx if out is None else out + bx
    return out


# ------------------------------------------------------------------------------ main
def build(p: Params, out: Path, check: bool = True, previews: bool = True) -> dict:
    assert p.layout in LAYOUTS, p.layout
    E = p.ext()
    parts = {n: upstream(n) for n in UPSTREAM_PARTS}
    head = parts["head"]
    fi = ray_first(head, [100.0, 0.0, 125.0], [1, 0, 0]) + 100.0
    assert abs(fi - FACE_INNER_STOCK) < 0.3, fi  # the face really is where we think it is
    face_inner, face_outer = FACE_INNER_STOCK + E, FACE_OUTER_STOCK + E
    head_s = stretch(head, p.stretch_at, E)
    top_z = 200.0 - ray_first(head_s, [face_outer - 10.0, 0.0, 200.0], [0, 0, -1])
    ceiling = 140.0 + ray_first(head_s, [face_outer - 10.0, 0.0, 140.0], [0, 0, 1])
    pl = place_board(p, face_inner, face_outer)

    h = to_manifold(head_s)
    cam = cap = None
    if p.layout == "case_closed":
        h = head_case_closed(p, h, pl, face_inner, face_outer)
    else:
        h = head_screen_face(p, h, head_s, pl, face_inner, face_outer, ceiling)
        cam = place_camera(p, face_outer, top_z)
        body, cuts, cap = camera_pod(p, cam, face_outer, top_z)
        h = (h + body) - cuts
    if p.switch_slot:
        h = h - box(*span(pl.X(M2_SWITCH_Y[0]), pl.X(M2_SWITCH_Y[1]), 1.0),
                    *span(pl.Y(M2_SWITCH_X[0]), pl.Y(M2_SWITCH_X[1]), 1.0), pl.Z(M2_Z[1]) + 0.2, 180.0)
    sheet_m = build_sheet(p, parts["head_bot_sheet"], pl)

    out.mkdir(parents=True, exist_ok=True)
    head_name = "head_maixcam2_screen" if p.layout == "screen_face" else "head_maixcam2"
    head_t, sheet_t = to_trimesh(h), to_trimesh(sheet_m)
    files = {head_name: head_t, "head_bot_sheet_maixcam2": sheet_t}
    if cap is not None:
        files["camera_pod_cap"] = to_trimesh(cap)
    for n, t in files.items():
        t.export(out / f"{n}.stl")

    report: dict = {"params": asdict(p) | {"extend": E}}
    bi = {
        "board_x": sorted([round(pl.X(M2_Y[0]), 2), round(pl.X(M2_Y[1]), 2)]),
        "bottom_z": round(pl.Z(M2_Z[0]), 2), "top_z": round(pl.Z(M2_Z[1]), 2),
        "ceiling_z": round(ceiling, 2), "top_clearance_mm": round(ceiling - pl.Z(M2_Z[1]), 2),
        "tripod_screw_engagement_mm": round(9.525 - (SHEET_TOP + p.pad - (SHEET_BOTTOM - p.boss_below)), 2),
        "tripod_nut_depth_mm": M2_TRIPOD_DEPTH,
    }
    if p.layout == "case_closed":
        bi["lens_front"] = pl.pt(M2_LENS_XZ[0], M2_LENS_FRONT_Y, M2_LENS_XZ[1]).round(2).tolist()
        bi["clearance_to_roll_mount_mm"] = round(pl.X(M2_Y[1]) - ROLL_MOUNT_FRONT, 2)
    else:
        bi["lens_front"] = cam.pt(M2_LENS_XZ[0], CAM_Y[0], M2_LENS_XZ[1]).round(2).tolist()
        bi["screen_glass_x"] = round(pl.X(M2_Y[1]), 2)
        bi["screen_centre_yz"] = [round(pl.Y(0.0), 2), round(pl.Z(0.0), 2)]
        bi["visible_window_mm"] = list(p.lip_window)
        bi["active_area_mm"] = list(M2_ACTIVE)
        bi["header_plugs_end_x"] = round(pl.X(M2_Y[0]) - p.plug_depth, 2)
        # cable: FPC5 on the main board -> out the lens hole in the cover -> up -> into the pod
        fpc5 = pl.pt(0.0, 1.5, -14.0)
        hole = pl.pt(M2_LENS_XZ[0], M2_Y[0], M2_LENS_XZ[1])
        conn = cam.pt(0.0, CAM_Y[1], 11.0)
        path = [fpc5, pl.pt(0.0, 1.5, M2_LENS_XZ[1]), hole, [hole[0] - 3, 0, hole[2]], [hole[0] - 3, 0, top_z], [conn[0] - 2, 0, top_z + 2], conn]
        bi["camera_cable_path_mm"] = round(float(sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(path, path[1:]))), 1)
    report["layout"] = p.layout
    report["board_in_head"] = bi
    report["parts"] = {n: {"watertight": bool(t.is_watertight), "volume_cm3": round(t.volume / 1000, 1)} for n, t in files.items()}
    report["parts"][head_name]["stock_volume_cm3"] = round(head.volume / 1000, 1)
    report["parts"]["head_bot_sheet_maixcam2"]["stock_volume_cm3"] = round(parts["head_bot_sheet"].volume / 1000, 1)
    if not check:
        return report

    mx = maixcam2_meshes()
    others = {"head (new)": h, "bottom sheet (new)": sheet_m}
    for n in ["head_roll_mount", "head_yaw_to_roll", "head_pitch_to_yaw", "speaker_stand",
              "flash_light_module", "flash_reflector_interface"]:
        others[n] = to_manifold(parts[n])
    eyes = []
    if p.layout == "case_closed":
        eyes = [parts[n].copy().apply_translation([E, 0, 0]) for n in STOCK_FRONT_PARTS]
        for n, e in zip(STOCK_FRONT_PARTS, eyes):
            others[n + " (moved)"] = to_manifold(e)
        movers = {"board": to_manifold(pl.mesh(mx["body_hull"])), "lens": to_manifold(pl.mesh(mx["lens_hull"]))}
        origin = pl.pt(M2_LENS_XZ[0], M2_LENS_FRONT_Y + 0.5, M2_LENS_XZ[1])
        board_w, cam_w = pl.mesh(mx["full"]), None
    else:
        others["camera pod cap"] = cap
        movers = {"board": to_manifold(pl.mesh(mx["case_hull"])), "camera module": to_manifold(cam.mesh(mx["cam_hull"]))}
        origin = cam.pt(M2_LENS_XZ[0], CAM_Y[0] + 0.5, M2_LENS_XZ[1])
        board_w, cam_w = pl.mesh(mx["case"]), cam.mesh(mx["cam"])
    movers[f"header_plugs_{p.plug_depth:g}mm"] = header_plugs(pl, p.plug_depth)
    report["overlap_mm3"] = {n: {k: round(overlap(m, o), 2) for k, m in movers.items()} for n, o in others.items()}
    report["fov_blocked_fraction"] = {
        "16:9 (86.7 x 55.8 deg)": round(fov_blocked([head_t] + eyes, origin, HFOV_HALF_169, VFOV_HALF_169), 4),
        "4:3 (70 x 55.8 deg)": round(fov_blocked([head_t] + eyes, origin, 35.2, VFOV_HALF_169), 4),
    }
    if p.layout == "screen_face":
        # how much of the active area the lip leaves visible, looking straight on
        ok_w = p.lip_window[0] >= M2_ACTIVE[0] and p.lip_window[1] >= M2_ACTIVE[1]
        report["screen_active_area_fully_visible"] = bool(ok_w)
    if previews:
        import render
        flash = [parts["flash_light_module"], parts["flash_reflector_interface"]]
        render.previews(HERE / "images", p.layout, head_t, sheet_t, board_w, eyes, flash,
                        [parts["head_roll_mount"], parts["head_yaw_to_roll"]], cam_w,
                        files.get("camera_pod_cap"))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Params()
    for k, v in asdict(d).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        elif isinstance(v, tuple):
            ap.add_argument(flag, type=float, nargs=len(v), default=v)
        elif k == "layout":
            ap.add_argument(flag, choices=LAYOUTS, default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    ap.add_argument("--out", type=Path, default=None, help="default: stl/<layout>")
    ap.add_argument("--no-check", action="store_true", help="skip the MaixCAM2 fit checks (no cadquery needed)")
    ap.add_argument("--no-previews", action="store_true")
    a = ap.parse_args(argv)
    p = Params(**{k: (tuple(getattr(a, k)) if isinstance(v, tuple) else getattr(a, k)) for k, v in asdict(d).items()})
    out = a.out or HERE / "stl" / p.layout
    rep = build(p, out, check=not a.no_check, previews=not a.no_previews)
    txt = json.dumps(rep, indent=2)
    print(txt)
    (out / "build_report.json").write_text(txt + "\n")
    bad = [n for n, v in rep.get("overlap_mm3", {}).items() if max(v.values()) > 1.0]
    if bad:
        print("WARNING: overlaps with", bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
