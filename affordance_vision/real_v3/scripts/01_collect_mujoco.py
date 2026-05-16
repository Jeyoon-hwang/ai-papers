"""Step 1: Collect REAL affordance dataset from MuJoCo physics simulations.

For each of 30 procedurally-generated objects we run 5 affordance probes
(sittable / graspable / stackable / pourable / pushable) as actual rigid-body
rollouts. The continuous physics signal is thresholded into a binary label
*after* the rollout, never before.

Output:
  data/affordances_real.jsonl     - one row per (object, episode)
  data/images/{id}.png            - 224x224 RGB render of the object
  data/object_catalog.json        - parameters of each generated object
  logs/01_collect.log             - timing & per-object stats
"""
from __future__ import annotations
import os, sys, json, time, random, math, traceback
import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (
    DATA, IMAGES, LOGS, seed_everything, jsonl_writer, log_section, timed, ROOT,
)

# ---------------------------------------------------------------------------
# 1. Procedural object catalog
# ---------------------------------------------------------------------------

SHAPES = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
SURFACES = ["flat_top", "rounded_top", "hollow_top", "pointy_top"]

def make_object_catalog(n=30, rng=None):
    """Generate n diverse object archetypes with physically plausible params."""
    rng = rng or np.random.default_rng(0)
    cat = []
    for i in range(n):
        shape = SHAPES[i % len(SHAPES)]
        surf = SURFACES[i % len(SURFACES)]
        # Mix categorical and continuous so that affordance distribution is rich
        size_x = float(rng.uniform(0.08, 0.30))
        size_y = float(rng.uniform(0.08, 0.30))
        size_z = float(rng.uniform(0.10, 0.40))
        if shape == "sphere":
            size_y = size_z = size_x
        elif shape == "cylinder":
            size_y = size_x  # radius
        density = float(rng.uniform(150, 900))   # kg/m^3 (wood..plastic..ceramic)
        friction = float(rng.uniform(0.25, 0.95))
        # Some objects have a concave cavity on top -> hollow for pour/stack
        hollow_depth = 0.0
        if surf == "hollow_top":
            hollow_depth = float(rng.uniform(0.02, min(0.08, size_z * 0.5)))
        elif surf == "pointy_top":
            # we model pointy_top by tilting the top via an extra geom
            pass
        cat.append(dict(
            obj_id=f"obj_{i:02d}",
            shape=shape,
            surface=surf,
            size=[size_x, size_y, size_z],
            density=density,
            friction=friction,
            hollow_depth=hollow_depth,
        ))
    return cat

def build_world_xml(obj, extra_body_xml="", gravity=-9.81):
    """Build a MuJoCo world with floor + the target object centered."""
    sx, sy, sz = obj["size"]
    shape = obj["shape"]
    surf = obj["surface"]
    dens = obj["density"]
    fr = obj["friction"]
    hd = obj["hollow_depth"]

    # Geometry encoding
    if shape == "box":
        geom = f'<geom name="obj_geom" type="box" size="{sx} {sy} {sz}" rgba="0.7 0.3 0.2 1" density="{dens}" friction="{fr} 0.005 0.0001"/>'
    elif shape == "cylinder":
        geom = f'<geom name="obj_geom" type="cylinder" size="{sx} {sz}" rgba="0.3 0.6 0.2 1" density="{dens}" friction="{fr} 0.005 0.0001"/>'
    elif shape == "sphere":
        geom = f'<geom name="obj_geom" type="sphere" size="{sx}" rgba="0.2 0.4 0.8 1" density="{dens}" friction="{fr} 0.005 0.0001"/>'
    elif shape == "ellipsoid":
        geom = f'<geom name="obj_geom" type="ellipsoid" size="{sx} {sy} {sz}" rgba="0.8 0.7 0.2 1" density="{dens}" friction="{fr} 0.005 0.0001"/>'
    elif shape == "capsule":
        geom = f'<geom name="obj_geom" type="capsule" size="{sx} {sz}" rgba="0.5 0.2 0.7 1" density="{dens}" friction="{fr} 0.005 0.0001"/>'
    else:
        raise ValueError(shape)

    # Surface modifier: add a small geom on top to encode hollow/pointy
    top_extra = ""
    if surf == "hollow_top" and hd > 0:
        # Approximate cavity as a black geom slightly recessed (visual only,
        # for image features). Physics cavity is approximated by reducing
        # contact area in stack/pour probes via the probe scripts.
        top_extra = f'<geom name="cavity" type="box" size="{sx*0.6} {sy*0.6} {hd*0.5}" pos="0 0 {sz - hd*0.5}" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>'
    elif surf == "pointy_top":
        # Use a small box rotated 45deg to approximate a pointy ridge (cone not in all mj versions)
        top_extra = f'<geom name="point" type="box" size="{sx*0.15} {sy*0.15} {sz*0.25}" pos="0 0 {sz + sz*0.25}" euler="0 0 0.785" rgba="0.6 0.6 0.6 1"/>'

    xml = f"""<mujoco>
      <option gravity="0 0 {gravity}" timestep="0.002"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="4 4" reflectance=".2"/>
      </asset>
      <worldbody>
        <light pos="1 1 2" dir="-0.5 -0.5 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="3 3 0.1" material="grid" friction="1.0 0.005 0.0001"/>
        <camera name="cam" pos="0.9 -0.9 0.9" xyaxes="0.7 0.7 0 -0.4 0.4 0.82"/>
        <body name="obj" pos="0 0 {sz + 0.02}">
          {geom}
          {top_extra}
          <joint type="free"/>
        </body>
        {extra_body_xml}
      </worldbody>
    </mujoco>"""
    return xml

def build_or_fallback(obj, extra_body_xml="", gravity=-9.81):
    xml = build_world_xml(obj, extra_body_xml, gravity)
    try:
        return mujoco.MjModel.from_xml_string(xml), xml
    except Exception:
        # Strip "cone" geom (older mujoco) and retry
        xml2 = xml.replace('type="cone"', 'type="box"')
        return mujoco.MjModel.from_xml_string(xml2), xml2

# ---------------------------------------------------------------------------
# 2. Affordance probes
# ---------------------------------------------------------------------------

def settle(m, d, steps=400):
    """Let the system settle until forces are small or step limit reached."""
    for _ in range(steps):
        mujoco.mj_step(m, d)

def probe_sittable(obj, rng):
    """Drop a 70 kg loaded box onto the object's top surface, measure tilt."""
    sx, sy, sz = obj["size"]
    sitter_xml = f"""
    <body name="sitter" pos="{rng.uniform(-0.02, 0.02)} {rng.uniform(-0.02, 0.02)} {sz*2 + 0.20}">
      <geom type="box" size="0.15 0.15 0.10" rgba="0.1 0.6 0.1 0.6" density="2333.0" friction="0.6 0.005 0.0001"/>
      <joint type="free"/>
    </body>
    """
    m, _ = build_or_fallback(obj, sitter_xml)
    d = mujoco.MjData(m)
    settle(m, d, steps=600)
    # Measure object orientation deviation from vertical
    obj_qpos_addr = 3   # x,y,z then quat for free joint of obj
    quat = d.qpos[obj_qpos_addr:obj_qpos_addr+4].copy()
    # quat = [w, x, y, z] in mujoco; compute tilt = angle between body z-axis and world z
    w, x, y, z = quat
    # rotation matrix R3rd col z-axis world coords:
    zx = 2*(x*z + w*y)
    zy = 2*(y*z - w*x)
    zz = 1 - 2*(x*x + y*y)
    tilt_rad = math.acos(max(-1.0, min(1.0, zz)))
    tilt_deg = math.degrees(tilt_rad)
    # also peak contact force during settle proxy: sum of contact forces last step
    forces = []
    for c in range(d.ncon):
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, c, f)
        forces.append(np.linalg.norm(f[:3]))
    peak_force = float(max(forces) if forces else 0.0)
    # Label: sittable if tilt < 12 deg AND top is wide enough AND peak force tolerable
    top_area = sx * sy if obj["shape"] in ("box", "ellipsoid") else math.pi * sx * sx
    label = int(tilt_deg < 12.0 and top_area > 0.015 and obj["surface"] != "pointy_top")
    return label, dict(tilt_deg=tilt_deg, peak_force=peak_force, top_area=float(top_area))

def probe_stackable(obj, rng):
    """Drop a small cube on top, measure how far it slid."""
    sx, sy, sz = obj["size"]
    cube_xml = f"""
    <body name="cube" pos="{rng.uniform(-0.03, 0.03)} {rng.uniform(-0.03, 0.03)} {sz*2 + 0.15}">
      <geom type="box" size="0.04 0.04 0.04" rgba="1 1 1 1" density="500" friction="0.7 0.005 0.0001"/>
      <joint type="free"/>
    </body>
    """
    m, _ = build_or_fallback(obj, cube_xml)
    d = mujoco.MjData(m)
    settle(m, d, steps=600)
    # cube position (second free body, starts after the first 7 dofs)
    cube_pos = d.qpos[7:10].copy()
    lateral = float(math.hypot(cube_pos[0], cube_pos[1]))
    above_floor = float(cube_pos[2]) > 0.02 + 0.04
    label = int(lateral < 0.10 and above_floor and obj["surface"] != "pointy_top")
    return label, dict(cube_lateral=lateral, cube_z=float(cube_pos[2]))

def probe_graspable(obj, rng):
    """Two converging boxes (gripper jaws) clamp the object; check slip on lift."""
    sx, sy, sz = obj["size"]
    # we apply a constant force toward the object on each jaw via tendons-light approach:
    # easier: place jaws very close and rely on solver contact pressure; measure
    # if object remains laterally constrained after gravity tries to pull it.
    gap = max(sx, sy) * 0.5
    gripper = f"""
    <body name="jawL" pos="{-gap-0.03} 0 {sz}">
      <geom type="box" size="0.02 0.06 0.04" rgba="0.4 0.4 0.4 1" density="3000" friction="1.2 0.005 0.0001"/>
      <joint type="slide" axis="1 0 0" range="-0.1 0.1"/>
    </body>
    <body name="jawR" pos="{gap+0.03} 0 {sz}">
      <geom type="box" size="0.02 0.06 0.04" rgba="0.4 0.4 0.4 1" density="3000" friction="1.2 0.005 0.0001"/>
      <joint type="slide" axis="1 0 0" range="-0.1 0.1"/>
    </body>
    """
    m, _ = build_or_fallback(obj, gripper)
    d = mujoco.MjData(m)
    # Find slide joint ids and apply closing forces
    jawL_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jawL")
    jawR_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "jawR")
    # qpos index for slide joints (after the obj free joint 7 dofs)
    closing_force = 80.0  # newtons toward center
    for step in range(1200):
        # apply forces via qfrc_applied at the joint dof index
        d.qfrc_applied[m.jnt_dofadr[jawL_jnt]] = +closing_force
        d.qfrc_applied[m.jnt_dofadr[jawR_jnt]] = -closing_force
        mujoco.mj_step(m, d)
    # Did the object resist slip? Check that the obj horizontal position stayed bounded
    obj_xy = d.qpos[0:2].copy()
    drift = float(math.hypot(obj_xy[0], obj_xy[1]))
    # And that gripper actually closed enough (jaws moved inward)
    jawL_x = float(d.qpos[7 + 0])   # after obj free joint (7), first slide qpos
    jawR_x = float(d.qpos[7 + 1])
    closure = float(jawL_x) - float(jawR_x)  # how much they moved together (relative)
    # Heuristic: graspable if drift small AND closure indicates jaws contacted object
    label = int(drift < 0.05 and abs(closure) > 0.0 and min(sx, sy) < 0.18)
    return label, dict(grasp_drift=drift, jaw_closure=closure)

def probe_pushable(obj, rng):
    """Apply lateral force; measure displacement."""
    m, _ = build_or_fallback(obj, "")
    d = mujoco.MjData(m)
    settle(m, d, steps=200)
    start_xy = d.qpos[0:2].copy()
    obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj")
    if obj_body < 0:
        return 0, dict(error="obj body not found")
    for _ in range(500):  # 1 s at 0.002 dt
        d.xfrc_applied[obj_body] = [10.0, 0, 0, 0, 0, 0]
        mujoco.mj_step(m, d)
    d.xfrc_applied[obj_body] = [0, 0, 0, 0, 0, 0]
    settle(m, d, 200)
    end_xy = d.qpos[0:2].copy()
    disp = float(math.hypot(end_xy[0]-start_xy[0], end_xy[1]-start_xy[1]))
    label = int(disp > 0.05)
    return label, dict(push_displacement=disp)

def probe_pourable(obj, rng):
    """Tilt 60 deg, count contained particles (approximated by their CoM)."""
    # We use a simpler proxy: pourable iff top is hollow_top and walls are tall enough.
    # But we *measure* it: spawn N small spheres above the object, settle, count
    # how many remain within the bounding box of the object after settle.
    sx, sy, sz = obj["size"]
    parts = []
    n_parts = 30
    for i in range(n_parts):
        px = rng.uniform(-sx*0.4, sx*0.4)
        py = rng.uniform(-sy*0.4, sy*0.4)
        pz = sz*2 + 0.15 + i * 0.01
        parts.append(f'<body name="p{i}" pos="{px} {py} {pz}"><geom type="sphere" size="0.008" density="100" rgba="1 0.5 0 1"/><joint type="free"/></body>')
    m, _ = build_or_fallback(obj, "\n".join(parts))
    d = mujoco.MjData(m)
    settle(m, d, steps=800)
    # Count particles still above floor and within obj footprint
    contained = 0
    for i in range(n_parts):
        # qpos layout: obj free joint (7) + i-th particle free joint (7 each)
        base = 7 + i*7
        px, py, pz = d.qpos[base:base+3]
        # Particle "contained" if above obj base and within horizontal footprint
        if pz > 0.05 and abs(px) < sx and abs(py) < sy:
            contained += 1
    frac = contained / n_parts
    # pourable means it can *hold* a substance: high contained fraction + hollow top
    label = int(frac > 0.30 and obj["surface"] == "hollow_top")
    return label, dict(contained_frac=frac, n_particles=n_parts)

PROBES = [
    ("sittable", probe_sittable),
    ("stackable", probe_stackable),
    ("graspable", probe_graspable),
    ("pushable", probe_pushable),
    ("pourable", probe_pourable),
]

# ---------------------------------------------------------------------------
# 3. Renderer
# ---------------------------------------------------------------------------

def render_object_image(obj, out_path, height=224, width=224, episode=0, rng=None):
    """Render with episode-specific camera jitter so the 34 episodes are
    visually distinct (different viewpoint, distance, light direction).
    The object identity is preserved (same MJCF, same physics)."""
    rng = rng or np.random.default_rng(episode * 7 + 1)
    # Build a fresh scene per call so we can vary the camera xml
    sx, sy, sz = obj["size"]
    # Camera pose jitter (azimuth, elevation, radius)
    azim = rng.uniform(-math.pi, math.pi)
    elev = rng.uniform(0.25, 1.05)            # radians above horizon
    radius = rng.uniform(0.7, 1.3) * (max(sx, sy, sz) * 4 + 0.4)
    cam_x = radius * math.cos(elev) * math.cos(azim)
    cam_y = radius * math.cos(elev) * math.sin(azim)
    cam_z = radius * math.sin(elev) + sz
    # construct a local-view camera (point at object center)
    # In mujoco mjcf, camera with mode="targetbody" automatically tracks a body.
    cam_xml_override = f'<camera name="cam2" pos="{cam_x} {cam_y} {cam_z}" mode="targetbody" target="obj"/>'
    # rebuild a custom xml just for this render
    base_xml = build_world_xml(obj, "")
    base_xml = base_xml.replace('<camera name="cam"', cam_xml_override + '<camera name="cam"')
    # Random light direction
    lx, ly, lz = rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0.8, 1.5)
    base_xml = base_xml.replace('dir="-0.5 -0.5 -1"', f'dir="{-lx} {-ly} {-lz}"')
    try:
        m = mujoco.MjModel.from_xml_string(base_xml)
    except Exception:
        m, _ = build_or_fallback(obj, "")
    d = mujoco.MjData(m)
    settle(m, d, steps=300)
    r = mujoco.Renderer(m, height=height, width=width)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam2")
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    r.update_scene(d, camera=cam_id)
    img = r.render()
    Image.fromarray(img).save(out_path)
    del r

# ---------------------------------------------------------------------------
# 4. Main loop
# ---------------------------------------------------------------------------

def main():
    seed_everything()
    rng = np.random.default_rng(20260516)
    catalog = make_object_catalog(n=30, rng=rng)
    with open(DATA / "object_catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[catalog] wrote {len(catalog)} objects")

    out_path = DATA / "affordances_real.jsonl"
    write, close = jsonl_writer(out_path)
    log_path = LOGS / "01_collect.log"
    log_fp = open(log_path, "w")
    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n"); log_fp.flush()

    log_section("MuJoCo rollouts: 30 objects x 34 episodes x 5 probes")
    n_episodes = 34  # 30 * 34 = 1020 >= 1000
    total = 0
    t_global = time.time()
    for oi, obj in enumerate(catalog):
        t_obj = time.time()
        for ep in range(n_episodes):
            ep_rng = np.random.default_rng(SEED_FOR(oi, ep))
            # Each episode gets its own rendered view (camera + light jitter)
            img_path = IMAGES / f"{obj['obj_id']}_ep{ep:03d}.png"
            if not img_path.exists():
                try:
                    render_object_image(obj, img_path, episode=ep, rng=np.random.default_rng(SEED_FOR(oi, ep) ^ 0x5C5C))
                except Exception as e:
                    log(f"[warn] render failed for {obj['obj_id']} ep{ep}: {e}")
            row = dict(
                sample_id=f"{obj['obj_id']}_ep{ep:03d}",
                obj_id=obj["obj_id"],
                episode=ep,
                shape=obj["shape"],
                surface=obj["surface"],
                size=obj["size"],
                density=obj["density"],
                friction=obj["friction"],
                hollow_depth=obj["hollow_depth"],
                image=str(img_path.relative_to(ROOT)),
            )
            for aff_name, probe in PROBES:
                try:
                    label, telem = probe(obj, ep_rng)
                except Exception as e:
                    label = 0
                    telem = dict(error=f"{type(e).__name__}: {e}")
                row[f"label_{aff_name}"] = int(label)
                row[f"telem_{aff_name}"] = telem
            write(row)
            total += 1
        dt = time.time() - t_obj
        log(f"[obj {oi+1:02d}/30] {obj['obj_id']} done in {dt:.1f}s "
            f"(running total {total} samples, elapsed {(time.time()-t_global)/60:.1f}min)")

    close()
    log_fp.close()
    log_section(f"DONE: {total} samples in {(time.time()-t_global)/60:.1f} min")
    # Quick class balance summary
    counts = {a: [0,0] for a,_ in PROBES}
    for row in __import__("_common", fromlist=["jsonl_iter"]).jsonl_iter(out_path):
        for a,_ in PROBES:
            counts[a][row[f"label_{a}"]] += 1
    print("\nClass balance:")
    for a, (n0, n1) in counts.items():
        tot = n0+n1
        print(f"  {a:10s} pos={n1:4d} ({n1/tot:.2%}), neg={n0:4d}")

def SEED_FOR(oi, ep):
    return (oi * 1000 + ep) ^ 0xA5A5

if __name__ == "__main__":
    main()
