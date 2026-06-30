# Running Closed-Loop Bench2Drive Evaluation (Headless, Remote Cluster)

> 📖 **New to CARLA? Read "Part I — Primer" directly below.** It explains, from zero, what CARLA
> is, how it's installed and run, what Vulkan/scenarios/route-XMLs are, and how Bench2Drive plugs
> into it. The operational **status, setup log, and run roadmap start at "Part II"** further down.

---

# Part I — Primer: CARLA & Bench2Drive from zero

*(This part assumes no prior CARLA knowledge and ignores the driver/version hurdles we hit — those
are documented in Part II. Here we describe how things work in the normal/ideal case.)*

## 1. What is CARLA?

**CARLA** is an open-source **simulator for autonomous driving**. Think of it as a video-game-quality
3D world (cars, roads, pedestrians, traffic lights, weather) that you can drive a virtual car
through — except instead of a human with a controller, *your self-driving model* does the driving.
It exists so you can test a driving policy **safely and repeatably** without a real car.

CARLA is built on top of **Unreal Engine 4.26** (a 3D game engine). That engine is what renders the
photorealistic camera images and simulates physics (how the car accelerates, brakes, collides).

### Client–server architecture (the single most important concept)
CARLA runs as **two separate programs that talk over the network**:

```
   ┌──────────────────────┐         TCP / RPC          ┌──────────────────────────┐
   │   CARLA SERVER        │  <───────────────────────  │   CLIENT (your Python)    │
   │  (CarlaUE4 binary)    │   "spawn a car here",      │  - spawns car & sensors   │
   │  - simulates physics  │   "give me the camera",    │  - reads camera/LiDAR     │
   │  - renders the world  │   "apply steering=0.2"     │  - runs the NN model      │
   │  - runs on the GPU    │  ───────────────────────>  │  - sends throttle/steer   │
   └──────────────────────┘     sensor data, state      └──────────────────────────┘
```

- The **server** (`CarlaUE4`) owns the simulated world. It does the heavy GPU work: rendering camera
  images, running physics. You launch it once; it listens on a TCP port (default **2000**; we use 3000).
- The **client** is a normal Python program using the **`carla` Python package**. It connects to the
  server and issues commands: create a vehicle, attach a camera, read the latest image, send a
  steering/throttle command, advance time by one step. **Your driving model lives in the client.**
- They can be on the same machine (our case) or different machines.

## 2. What is Vulkan, and why does CARLA need it?

**Vulkan** is a low-level **graphics API** — a standard way for software to talk to the GPU to draw
3D graphics (the modern successor to OpenGL). Unreal Engine renders CARLA's world (and every camera
sensor image) through Vulkan, which runs on the NVIDIA GPU.

Key point: **even "headless" CARLA (no monitor) still needs Vulkan + a GPU**, because the camera
sensors your model reads are *rendered images* — they have to be drawn on the GPU. Running CARLA
headless on a server uses the flag **`-RenderOffScreen`**: it renders to GPU memory instead of a
screen. So a working Vulkan + NVIDIA driver stack is a hard prerequisite (this is what Part II's
"NFS/driver" sections are about).

## 3. Installing CARLA — what the files are

CARLA ships as a **precompiled release** (you do *not* build Unreal from source). You download two
tarballs and unpack them:

```bash
# 1) The simulator itself (engine + base maps + Python API)  (~7.9 GB compressed, ~20 GB unpacked)
wget .../CARLA_0.9.15.tar.gz
mkdir CARLA_0.9.15 && tar -xzf CARLA_0.9.15.tar.gz -C CARLA_0.9.15

# 2) Extra high-detail maps used by Bench2Drive (Town06/07/11/12/13/15) (~6.9 GB)
#    Extract on TOP of the same folder.
tar -xzf AdditionalMaps_0.9.15.tar.gz -C CARLA_0.9.15
```

Inside the unpacked `CARLA_0.9.15/` you get:

| Path | What it is |
|------|------------|
| **`CarlaUE4.sh`** | The launcher script — **run this to start the server**. It execs the engine binary. |
| `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping` | The actual compiled Unreal Engine + CARLA server binary. |
| `CarlaUE4/Content/Carla/Maps/` | The **maps/towns** (Town01, Town02, …) as Unreal asset files. |
| **`PythonAPI/`** | The **client side**. `PythonAPI/carla/dist/` holds the `carla` client library as `.egg`/`.whl` files; `PythonAPI/examples/` has demo scripts; `PythonAPI/carla/agents/` has helper navigation code. |
| `HDMaps/` | OpenDRIVE (`.xodr`) road-network files — the logical road graph (lanes, junctions) behind each town. |
| `Import/`, `ImportAssets.sh` | Used to import the AdditionalMaps assets if needed. |
| `CHANGELOG`, `LICENSE`, `Co-Simulation/` | Docs / SUMO co-sim integration (not needed here). |

**Getting the Python client.** Your Python program needs the `carla` module. Two ways:
1. Use the bundled file in `PythonAPI/carla/dist/` (an `.egg`/`.whl` built for a **specific Python
   version** — e.g. cp37 = Python 3.7). You add it to `PYTHONPATH`.
2. Or `pip install carla==0.9.15` (PyPI provides wheels for several Python versions).
   **The client version must match the server version** (0.9.15 ↔ 0.9.15).

## 4. From "installed" to "a car driving" — the normal workflow

Once installed, the typical loop is:

**(a) Start the server** (once):
```bash
./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -graphicsadapter=0
#   -RenderOffScreen : headless (render to GPU, no window)
#   -carla-rpc-port  : the TCP port clients connect to
#   -graphicsadapter : which GPU to render on
```

**(b) Connect a client and build the scene** (Python):
```python
import carla
client = carla.Client('localhost', 2000); client.set_timeout(20.0)
world  = client.load_world('Town03')          # pick a map
bp     = world.get_blueprint_library()
ego    = world.spawn_actor(bp.find('vehicle.lincoln.mkz_2020'),
                           world.get_map().get_spawn_points()[0])   # spawn the car
cam    = world.spawn_actor(bp.find('sensor.camera.rgb'),
                           carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego)
cam.listen(lambda image: handle(image))        # camera pushes frames to your callback
```

**(c) Run the simulation loop.** For reproducible evaluation, CARLA uses **synchronous mode**: the
simulation only advances when the client says "tick". Each tick advances a fixed timestep (e.g.
`1/20 s`), renders the sensors, and waits for your control command:
```python
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05            # 20 FPS sim time
world.apply_settings(settings)

while not done:
    world.tick()                               # advance 1 step; sensors produce new data
    img = latest_camera_frame                  # your sensor callback stored it
    control = my_model(img, speed, target)     # <-- your driving policy decides
    ego.apply_control(carla.VehicleControl(throttle=control.throttle,
                                           steer=control.steer, brake=control.brake))
```

So the **"input" and "output" of CARLA** are:
- **What you feed the simulator:** a *map*, a *scenario/route definition*, *weather*, *other traffic*,
  and an *ego vehicle with sensors*.
- **What the simulator gives your model each tick (the model's input):** sensor streams — RGB
  cameras (front + surround views), and optionally depth/semantic/LiDAR/radar — plus ego state
  (speed, GPS/GNSS, IMU) and a navigation *target/command* (where to go next).
- **What your model returns to the simulator (its output):** vehicle **control** — `throttle`
  (0–1), `steer` (−1…1), `brake` (0–1) — or a short **trajectory of waypoints** that a controller
  converts into throttle/steer. *(DeepSight's agent outputs waypoints, then a PID controller turns
  them into throttle/steer — see `bench2drive/team_code/qwen_b2d_agent.py` + `pid_controller.py`.)*

## 5. Maps / Towns

A CARLA **map** ("Town") is a 3D environment plus its logical road network (an OpenDRIVE `.xodr`
describing lanes, junctions, speed limits). CARLA ships **base towns** (Town01–05, Town10HD) in the
main package; **Town06/07/11/12/13/15** come in **AdditionalMaps**. Bench2Drive's 220 routes span
Town01–15, so the full benchmark needs AdditionalMaps; a quick smoke test can use a base town.

## 6. What are those XML "route" files?

The `bench2drive/leaderboard/data/*.xml` files are **route definitions** for the evaluation
framework ("Leaderboard"). One file lists one or more **routes**; each route says:
- **which town** it runs in,
- a **path to drive**, as an ordered list of waypoint positions, and
- the **scenarios** (scripted events) that should trigger along that path.

Simplified shape of one route (this is the Leaderboard-2.0 format Bench2Drive uses):
```xml
<routes>
  <route id="24206" town="Town03">
    <waypoints>                         <!-- the path the ego should follow -->
      <position x="-123.6" y="-135.2" z="0.7"/>
      <position x="-125.4" y="-134.1" z="0.8"/>
      ... (the route as a polyline of x,y,z points) ...
    </waypoints>
    <scenarios>                         <!-- events that fire at trigger points on the route -->
      <scenario name="HardBreakRoute_1" type="HardBreak">
        <trigger_point x="..." y="..." z="..." yaw="..."/>
        ... scenario-specific parameters ...
      </scenario>
    </scenarios>
  </route>
</routes>
```
So a route XML is *both* "where to drive" *and* "what challenges to throw at the car along the way".
`bench2drive220.xml` = the official 220 routes; we created `smoke_test_town03.xml` = just one route.

## 7. What is a "scenario", and how do Bench2Drive scenarios run in CARLA?

A **scenario** is a **scripted, reproducible traffic situation** designed to test a specific driving
skill — e.g. *a pedestrian suddenly crosses*, *a car cuts in*, *an accident blocks the lane*, *yield
to an emergency vehicle*, *run a red light at a junction*. Each scenario controls the **other**
actors (pedestrians/cars) with fixed timing so every run is comparable.

These are implemented by **`scenario_runner`** (`bench2drive/scenario_runner/`): a library of
scenario classes (spawn this NPC, make it brake at this trigger, check if the ego crashed). When the
ego car reaches a scenario's **trigger point** on the route, scenario_runner activates that scenario.
**Bench2Drive defines 44 scenario types**; the 220 routes are these scenarios placed across the towns.

How it all comes together at run time — the **Leaderboard** stack
(`bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py`) is the orchestrator that:
1. **Launches the CARLA server** itself (you don't start it separately for an eval run),
2. loads the route's **town**, spawns the **ego vehicle**, and arms the route's **scenarios**
   (via scenario_runner),
3. loads your **agent** (a Python class implementing CARLA's "autonomous agent" interface — it
   declares which **sensors** it wants and a `run_step()` that returns a `carla.VehicleControl`),
4. **ticks the simulation**: each step it renders the agent's sensors, calls `run_step()`, applies
   the returned control, and lets scenario_runner update the NPCs,
5. **scores** the run — did the ego complete the route, did it collide / run lights / leave lane —
   producing the metrics (**Driving Score, Success Rate, Efficiency, Comfortness, Multi-Ability**),
   written to a results JSON.

For DeepSight, the agent in step 3 is **`bench2drive/team_code/qwen_b2d_agent.py`** (`QwenAgent`): it
requests the front + surround cameras, builds a text+image prompt, runs the Qwen2.5-VL model
(`model.generate()`), parses predicted waypoints from the output, and converts them to throttle/steer
with a PID controller — closing the loop.

## 8. "Open-loop" vs "closed-loop" (why this matters)

- **Open-loop**: feed the model a fixed recorded clip, compare its predicted trajectory to the human's
  (a single L2 distance number). The model's predictions **don't affect** what happens next.
- **Closed-loop** (what this doc is about): the model actually **drives** in the simulator; its
  outputs change the next sensor inputs (real feedback). This is far more meaningful — errors
  compound, and you measure whether the car *actually* reaches the goal without crashing. Bench2Drive
  is a closed-loop benchmark; that's why we need the full CARLA simulator running.

---

# Part II — Operational status, setup log & run roadmap

> ## 🟩 STATUS (2026-06-26): Vulkan UNBLOCKED — all 8 A100s render via Vulkan
> The pod was recreated with graphics capability (`config.yaml` now sets
> `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display`). The NVIDIA container runtime
> then auto-injects the GL/Vulkan libs, the Vulkan ICD (`/etc/vulkan/icd.d/nvidia_icd.json`),
> and `/dev/nvidia-modeset`. Two OS packages still had to be installed in-pod:
> **`libvulkan1`** (Vulkan loader) and **`libegl1`** (GLVND EGL dispatch loader — the NVIDIA
> Vulkan driver `dlopen`s `libEGL.so.1` at init; without it negotiate fails with
> `VK_ERROR_INITIALIZATION_FAILED`). After that:
> `vulkaninfo --summary` → 8× `NVIDIA A100 80GB PCIe`, driver NVIDIA 550.54.15. ✅
>
> **Vulkan adapter index == CUDA index here** (GPU id 0–7 = the A100s; id 8 = software llvmpipe),
> so CARLA's `-graphicsadapter=N` targets CUDA GPU N directly. All 8 GPUs can host CARLA.
>
> ⚠️ The in-pod apt installs live on the **ephemeral** container FS and vanish on pod restart.
> Re-run the persistent setup script **[scripts/setup_carla_vulkan.sh](scripts/setup_carla_vulkan.sh)**
> after any restart (the repo lives on persistent `/home/saab03`), or bake it into the image.

This document is the roadmap for running the **closed-loop CARLA / Bench2Drive**
evaluation of the DeepSight (Qwen2.5-VL) agent on a **headless** remote cluster
(no display). It evaluates the official **220 short routes** and produces the five
closed-loop metrics: **Driving Score (DS), Success Rate (SR), Efficiency,
Comfortness, Multi-Ability**.

> Target environment for this doc: a single node with **8 GPUs**, of which only
> **2–3 can host a CARLA server at a time**. So we run **2–3 parallel eval tasks**,
> each = one CARLA server + one agent (model) instance.

---

## ▶️ RESUME HERE (session 4, 2026-06-29)

### 🟢 CARLA SERVER WORKS HEADLESS ON THE A100 — Phase 0 DONE
Root cause of the earlier "RenderThread timed out / Signal 11" crash was **NFS**: running CARLA
off `/home/saab03` (NFS) stalled UE4 texture streaming past the 60s render-fence. **Fix: run CARLA
from LOCAL disk.** Copied to **`/opt/carla/CARLA_0.9.15`** (local overlay, owned by `carla`).
Verified: server stable >4 min, attaches to GPU 0 (~5.8 GB), Python client connects
(client+server 0.9.15), Town10HD loads, `get_available_maps()` works.

**Canonical working launch (manual server, for ad-hoc testing):**
```bash
pkill -9 CarlaUE4 || true     # NOTE: comm-based, NOT 'pkill -f' (that self-kills the shell!)
mkdir -p /tmp/xdg-carla && chown carla:carla /tmp/xdg-carla && chmod 700 /tmp/xdg-carla
setsid su carla -c "cd /opt/carla/CARLA_0.9.15 && \
  export XDG_RUNTIME_DIR=/tmp/xdg-carla VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json && \
  exec ./CarlaUE4.sh -RenderOffScreen -nosound -graphicsadapter=0 -carla-rpc-port=3000 > /tmp/carla_local.log 2>&1" </dev/null &
```
Critical: **local disk**, run as **carla**, **VK_ICD_FILENAMES=nvidia**, **setsid+</dev/null**,
log redirect **inside** the su (outer root redirect to /tmp hit permission-denied), and
**`pkill CarlaUE4` not `pkill -f CarlaUE4...`** (the `-f` pattern matched & killed our own shell —
this caused many empty-output/exit-1 failures).

### 📘 The NFS problem — what it was and how it was fixed (full explanation)
**Symptom:** every CARLA launch died after ~60s with
`LowLevelFatalError ... GameThread timed out waiting for RenderThread after 60.00 secs` → `Signal 11`
(segfault), identically on GPU 0 and GPU 7.

**Root cause = slow disk I/O, not the GPU.** CARLA is Unreal Engine 4.26, a real-time renderer. Its
**RenderThread** streams ~20 GB of binary assets (tens of thousands of `.uasset`/`.ubulk` texture &
mesh files) from disk into GPU memory at startup/runtime. We had extracted CARLA onto
`/home/saab03`, which is an **NFS mount** (network filesystem, server `10.0.1.x`). Every file
open/read on NFS is a network round-trip (milliseconds), so the RenderThread blocked on slow asset
reads. UE4's GameThread has a 60 s watchdog on the RenderThread fence; when the RenderThread stayed
stuck on NFS I/O past 60 s, the engine aborted (Signal 11). The GPU was never the bottleneck — which
is why changing GPUs didn't help and why CARLA never made rendering progress.

**Fix = run CARLA from local disk.** Copied the extracted tree from NFS
`/home/saab03/carla/CARLA_0.9.15` → **local overlay disk `/opt/carla/CARLA_0.9.15`** (~470 G free,
local NVMe-class latency), `chown -R carla:carla`. Asset streaming is now fast enough to meet the
60 s fence, so the server boots and runs stably. Same GPU, same flags — only the storage changed —
and it went from crashing every time to running >4 min and serving RPC. That A→B isolation is the
proof the cause was NFS.

**Trade-off / caveat:** `/opt` (the container overlay) is **ephemeral** — it's wiped on pod restart.
After a restart, re-copy from the persistent NFS copy:
`cp -a /home/saab03/carla/CARLA_0.9.15 /opt/carla/CARLA_0.9.15 && chown -R carla:carla /opt/carla/CARLA_0.9.15`
(the tarballs + an extracted copy stay on NFS at `/home/saab03/carla/`). To avoid re-copying, bake
CARLA into the container image, or keep the local copy and just re-chown.

### ⏸️ STATUS: paused after Phase 0. Phase 2 (wire eval scripts) NOT started.
- The manual CARLA test server (PID was 116792, GPU 0, port 3000) may still be running — kill with
  `pkill CarlaUE4` (NOT `pkill -f`) when not needed; it holds ~5.8 GB on shared GPU 0.
- Pending Phase 2 edits (not yet applied): make `-graphicsadapter` env-driven + force NVIDIA ICD in
  `leaderboard_evaluator.py:208`; set `CARLA_ROOT=/opt/carla/CARLA_0.9.15` in `run_evaluation.sh`;
  build a wrapper to run the whole eval **as user `carla`** with the `carla-eval` conda env.
- **STILL NEEDED FROM USER: the trained checkpoint path** to evaluate.

### ⚠️ For the leaderboard eval (which launches CARLA itself)
`leaderboard_evaluator.py` spawns `CarlaUE4.sh` via subprocess. So the **whole eval must run as
user `carla`** (else spawned CARLA is root and refuses), with `CARLA_ROOT=/opt/carla/CARLA_0.9.15`,
`VK_ICD_FILENAMES` exported, and the hardcoded `-graphicsadapter=4` at `leaderboard_evaluator.py:208`
changed (use a free GPU). `carla` must also reach the conda env `/home/saab03/miniconda3/envs/carla-eval`.

### Done this session
- ✅ Vulkan fully working: 8× A100 via `nvidia_icd.json` + `libvulkan1` + `libegl1`. (`libegl1` was
  the missing piece — driver `dlopen`s `libEGL.so.1` at init.)
- ✅ CARLA **version decided: 0.9.15** (bench2drive pinned to it; PyPI has a `carla==0.9.15` **cp310**
  wheel so client matches server in py3.10 — the shipped eggs are only cp27/cp37).
- ✅ Downloaded CARLA_0.9.15.tar.gz (7.9G) + AdditionalMaps_0.9.15.tar.gz (6.9G) to
  `/home/saab03/carla/`. **Extracted base CARLA only** → `/home/saab03/carla/CARLA_0.9.15/` (20G,
  Town01-05+Town10HD). AdditionalMaps NOT extracted yet (Town12/13/15 need it).
- ✅ `carla-eval` conda env (py3.10) created with: torch 2.7.1+cu126, transformers 4.51.3,
  qwen-vl-utils, accelerate, opencv, scipy, shapely, py-trees 0.8.3, **carla 0.9.15** (cp310). CUDA OK.
- ✅ Resolved transformers question: **use stock transformers** (not vendored `src/transformers`) —
  agent only does `model.generate()`; dino/vis_head keys ignored.
- ✅ Created **non-root user `carla` (uid 1001)** — UE4 refuses to run as root. CARLA files are
  owned by uid 1001 so `carla` has full access. Parent dirs made traversable (`/home/saab03`=777).
- ✅ Single-route smoke test XML: `bench2drive/leaderboard/data/smoke_test_town03.xml` (route 24206,
  Town03, base town — no AdditionalMaps needed).

### KEY FACTS / gotchas learned
- **Run CARLA as user `carla`, NOT root** (UE4 "Refusing to run with the root privileges").
- **Must force NVIDIA-only ICD** or UE4 may pick mesa llvmpipe: `export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.
- **Detach with `setsid`** + `</dev/null` or CARLA dies when the launching shell command exits
  (the leaderboard uses `subprocess.Popen(..., preexec_fn=os.setsid)`).
- Shared multi-tenant node: **`nvidia-smi` total memory is NOT ours** — use
  `nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid` and match GPU UUID to attribute usage.
  GPU UUIDs: 0=ce34 1=eafc 2=f999 3=2f29 4=9bde 5=e516 6=5dbe 7=0d34. **User said use GPU 7.**
- Container PID 1 is `tail -f /dev/null` → it does NOT reap orphans; dead CarlaUE4.sh lingers as
  harmless `<defunct>` zombies. Ignore them.
- CARLA install is on **NFS** (`/home/saab03`) — a suspect for the RenderThread timeout (see below).

### Working launch command (as carla, GPU 7, detached)
```bash
pkill -9 -f CarlaUE4-Linux-Shipping || true ; sleep 1 ; : > /tmp/carla_server.log
setsid su carla -c "cd /home/saab03/carla/CARLA_0.9.15 && \
  export XDG_RUNTIME_DIR=/tmp/xdg-carla VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json && \
  exec ./CarlaUE4.sh -RenderOffScreen -nosound -graphicsadapter=7 -carla-rpc-port=3000" \
  > /tmp/carla_server.log 2>&1 < /dev/null &
# /tmp/xdg-carla must exist & be chown carla. Server log -> /tmp/carla_server.log
```

### Next: debug the RenderThread timeout (hypotheses, in priority order)
1. **NFS I/O** — CARLA assets are on NFS; UE4 texture streaming may stall >60s. TEST: copy
   `/home/saab03/carla/CARLA_0.9.15` to local ephemeral `/root/carla` (overlay, ~476G free, fast)
   and launch from there. If it boots, NFS is the cause → run CARLA from local disk each session.
2. **Was CARLA actually on the GPU?** Wasn't confirmed (the "27GB on GPU0" was another tenant).
   During a run, check `nvidia-smi --query-compute-apps=...|grep <uuid>` to see if CarlaUE4 attaches.
   If it never attaches → Vulkan device-create hang, not I/O.
3. Try `-quality-level=Low` and/or `-ini:Engine:[ConsoleVariables]:r.Streaming.PoolSize=...` (A100
   80GB → UE4 may size a huge texture pool). Also try bumping render-fence timeout.
4. Shipping build logs nothing to stdout; for real UE4 logs check `/home/carla/.config/Epic/` and
   `CarlaUE4/Saved/Logs/` (none appeared yet). `-stdout -FullStdOutLogOutput` did not help.

### After CARLA renders: remaining roadmap
- Phase 2: wire `run_evaluation.sh` (CARLA_ROOT=/home/saab03/carla/CARLA_0.9.15) + `run_evaluation_qwen.sh`
  (TEAM_CONFIG=checkpoint, BASE_ROUTES=.../smoke_test_town03, GPU_RANK/graphicsadapter, SAVE_PATH).
  **Run the whole eval as user `carla`** (so the CARLA subprocess isn't root) — ensure `carla` can
  use the conda env `/home/saab03/miniconda3/envs/carla-eval` (world-readable) + repo path.
- Fix hardcoded `-graphicsadapter=4` in `leaderboard_evaluator.py:208` (make env-driven; use GPU 7).
- **STILL NEED FROM USER: the trained checkpoint path** (scripts point at NAS `/mnt/nas-data-1/...`).
- Phase 3: smoke test 1 route → Phase 4: parallel 220 → Phase 5: metrics.

### First commands to run on resume
```bash
bash scripts/setup_carla_vulkan.sh                       # restore Vulkan if pod restarted
# recreate carla user if pod restarted: useradd -u 1001 -m -s /bin/bash carla; chmod o+x /home/saab03/carla
mkdir -p /tmp/xdg-carla && chown carla:carla /tmp/xdg-carla && chmod 700 /tmp/xdg-carla
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p $XDG_RUNTIME_DIR
vulkaninfo --summary | grep -c "NVIDIA A100"             # expect 8
# then attack hypothesis #1: run CARLA from local disk
```

---

## Phase 0 findings — actual state of THIS node (probed 2026-06-26)

| Item | State | Implication |
|------|-------|-------------|
| GPUs | 8× **A100 80GB PCIe** | compute cards, no display; CARLA must render via Vulkan offscreen |
| GPU load | 0–3 busy (~67 GB, training), 4–7 ~48 GB free | run CARLA tasks on **GPUs 4–7**; do NOT disturb 0–3 |
| NVIDIA driver | **550.54.15**, installed via **.run** (not apt; no `nvidia-driver-*` dpkg) | userspace GL/Vulkan libs must match this exact version |
| Vulkan loader (`libvulkan1`) | **missing** | install via apt (version-independent, safe) |
| `vulkaninfo` / vulkan-tools | **missing** | install via apt for diagnostics |
| NVIDIA Vulkan ICD (`nvidia_icd.json`, `libnvidia-gl`) | **missing** (compute-only driver install) | **the key blocker** — see below |
| apt `libnvidia-gl-550` candidate | **550.163.01** ≠ running 550.54.15 | apt version would MISMATCH the kernel module |
| Matching `NVIDIA-*550.54.15.run` on disk | **not found** | must download from NVIDIA to do userspace-only GL/Vulkan install |
| Root / apt | **yes** (uid=root) | can install system packages |
| Docker | **not installed** | container route unavailable |
| OS | Ubuntu 22.04.5 LTS | — |
| conda | `~/miniconda3`, env **`deepsight`** (py3.10) exists | eval env should be **py3.10** (matches repo + carla 0.9.16 wheels) |

### ⛔ HARD BLOCKER (root-caused 2026-06-26): pod has no graphics capability
This environment is a **Kubernetes pod** (`cgroup: /kubepods.slice/...cri-containerd-...`)
provisioned by the NVIDIA container runtime with:
```
NVIDIA_DRIVER_CAPABILITIES=compute,utility     # graphics / display NOT requested
```
Consequences (all confirmed by probing):
- The NVIDIA GL/Vulkan userspace libs were not mounted (we manually added them from the .run).
- `/dev/nvidia-modeset` was absent (we manually created it).
- `/proc/driver/nvidia/capabilities/` is absent (cannot be created — it's runtime-mounted).
- **NVIDIA's Vulkan ICD `vk_icdNegotiateLoaderICDInterfaceVersion` returns `-3`
  (`VK_ERROR_INITIALIZATION_FAILED`) in pure userspace, before any `/dev/nvidia*` access.**
  vulkaninfo → `ERROR_INCOMPATIBLE_DRIVER / Found no drivers!`

**Why it can't be fixed from inside the running pod:** `NVIDIA_DRIVER_CAPABILITIES` is read by
`nvidia-container-runtime` **at container creation** to decide what to mount and how to wire the
driver's graphics capability. Exporting it now has no effect. CARLA needs Vulkan graphics, so the
**pod must be (re)created** with graphics capability, e.g.:
```yaml
env:
  - name: NVIDIA_DRIVER_CAPABILITIES
    value: "all"          # or: compute,utility,graphics,display
```
(or the equivalent `--gpus 'all,"capabilities=...,graphics,display"'` for plain Docker).
This is a cluster/pod-spec change — owner action, possibly via cluster admin.

What we DID install on this node (still useful once a graphics-capable pod exists / persists in the
image): `libvulkan1` (LunarG 1.4.313) + `vulkan-tools`, `kmod`, `strace`, `gcc`, and the matching
**550.54.15 userspace GL/Vulkan libs** via `NVIDIA-Linux-x86_64-550.54.15.run --no-kernel-module`.

### The Vulkan blocker and the safe fix (generic notes)
The driver is a **compute-only** install: the kernel module (550.54.15) is loaded, but the
NVIDIA **graphics userspace** (`libnvidia-gl-550` → provides `nvidia_icd.json`, `libGLX_nvidia`,
Vulkan ICD) was never installed. apt only offers **550.163.01**, which would **not match** the
running kernel module — and changing the kernel module requires a **reboot that would kill the
training jobs on GPUs 0–3**. So the safe path is **userspace-only**, no reboot:

1. `apt install libvulkan1 vulkan-tools` — Vulkan loader + `vulkaninfo` (version-independent, safe).
2. Download **`NVIDIA-Linux-x86_64-550.54.15.run`** and install **userspace graphics only**:
   `sh NVIDIA-Linux-x86_64-550.54.15.run --no-kernel-module --no-x --silent`
   (installs `nvidia_icd.json` + GL/Vulkan libs matching the already-loaded kernel module; does
   **not** touch the kernel module, so running jobs are unaffected).
3. Verify: `vulkaninfo --summary` should list the A100(s) as Vulkan devices.

> ⚠️ Do not `apt install libnvidia-gl-550` (pulls 550.163.01) and do not reboot — either risks
> the in-flight training on GPUs 0–3.

---

## 0. How the eval is wired (read this first)

The launch chain is:

```
run_evaluation_qwen.sh                       # sets paths, GPU, ports
  └─ leaderboard/scripts/run_evaluation.sh   # exports env, calls the evaluator
       └─ leaderboard/leaderboard/leaderboard_evaluator.py
            ├─ launches the CARLA server ITSELF (subprocess.Popen)   # line ~208
            └─ loads the agent: team_code/qwen_b2d_agent.py
                 └─ init_model() -> Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
```

Key facts discovered in the code:

- **You do NOT start CARLA manually.** `leaderboard_evaluator.py` boots it per run:
  `leaderboard_evaluator.py:208`
  ```
  CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=<port> -graphicsadapter=4
  ```
  `-RenderOffScreen` already makes it headless — **no X server / virtual display needed**,
  but a working **Vulkan + GPU driver** is required (CARLA still renders camera sensors on GPU).
- **CARLA is NOT controlled by `CUDA_VISIBLE_DEVICES`.** Its render GPU is chosen by the
  `-graphicsadapter=N` flag. This value is **hardcoded** (see step 2 below) and must be fixed.
- The **agent/model** GPU *is* controlled by `CUDA_VISIBLE_DEVICES` (set via `GPU_RANK` in the scripts).
- Eval runs in a **separate conda env** from training (CARLA client + its own torch/transformers).
  See frozen `example.txt`: `carla 0.9.16`, `torch 2.7.1`, `transformers 4.51.3`, `flash_attn`, `vllm`.
- Official routes: `bench2drive/leaderboard/data/bench2drive220.xml` (220 routes, confirmed).
- CARLA crashes often → `RESUME=True` lets a relaunch continue; `tools/clean_carla.sh` kills stragglers.

---

## 1. Roadmap (phases)

### Phase 0 — Cluster prerequisites  ✅ COMPLETE (2026-06-29)
- [x] **Vulkan check** — `vulkaninfo --summary` → 8× NVIDIA A100 (driver 550.54.15). Required
      pod graphics caps (`config.yaml`) + `libvulkan1` + **`libegl1`**. Re-setup: `scripts/setup_carla_vulkan.sh`.
- [x] **Identify which GPUs can run CARLA** — ALL 8 render; **Vulkan adapter index == CUDA index**.
      Pick a free GPU at run time via `nvidia-smi --query-compute-apps`+UUID (shared node).
- [x] **Install CARLA 0.9.15** — downloaded to `/home/saab03/carla/` (tarballs persist on NFS);
      **base extracted + copied to LOCAL disk `/opt/carla/CARLA_0.9.15`** (run from local, NOT NFS —
      see "NFS problem" above). `CARLA_ROOT=/opt/carla/CARLA_0.9.15`. **AdditionalMaps NOT extracted**
      (only needed for Town06/07/11/12/13/15; base Town01-05+10HD suffice for the smoke test).

### Phase 1 — Eval conda environment  ✅ COMPLETE (2026-06-29)
- [x] Created py3.10 env **`carla-eval`** (`~/miniconda3/envs/carla-eval`).
- [x] CARLA python client = **`pip install carla==0.9.15`** (a **cp310** wheel exists on PyPI, so it
      matches the 0.9.15 server in py3.10 — the shipped eggs are cp27/cp37 only, unusable here).
- [x] Installed torch 2.7.1+cu126, transformers 4.51.3, qwen-vl-utils, accelerate, opencv, scipy,
      shapely, py-trees 0.8.3, etc. (CUDA verified available).
- [x] **OPEN QUESTION RESOLVED → use stock transformers** (not vendored `src/transformers`). Agent
      only does `model.generate()`; the training-only world-model path (`vis_head`/DINOv3) isn't
      exercised, so the `dino*`/`vis_head*` checkpoint keys are harmlessly ignored.

### Phase 2 — Wire up paths  ⏳ NEXT (not started; edits reverted when paused)
- [ ] `leaderboard/scripts/run_evaluation.sh`: set `CARLA_ROOT=/opt/carla/CARLA_0.9.15`.
- [ ] `leaderboard/scripts/run_evaluation_qwen.sh`:
      `TEAM_CONFIG` → **checkpoint (NEEDED FROM USER)**, `BASE_ROUTES=.../smoke_test_town03`,
      `SAVE_PATH`, `CHECKPOINT_ENDPOINT`, `GPU_RANK` (a free GPU).
- [ ] Make `-graphicsadapter` env-driven + force NVIDIA ICD in `leaderboard_evaluator.py:208`
      (planned diff: read `CARLA_GRAPHICSADAPTER` and prefix `VK_ICD_FILENAMES=` on `cmd1`).
- [ ] Build a wrapper to run the **whole eval as user `carla`** (UE4 refuses root) using the
      `carla-eval` conda env; ensure `carla` can read the env + repo paths.

### Phase 3 — Single-route smoke test (1 GPU)  ⬜ TODO
- [x] Smoke-test route prepared: `bench2drive/leaderboard/data/smoke_test_town03.xml` (route 24206,
      Town03 — base town, no AdditionalMaps needed). (devtest route is Town12 → needs AdditionalMaps.)
- [ ] Run as `carla`: `cd bench2drive && bash leaderboard/scripts/run_evaluation_qwen.sh`
- [ ] Confirm: CARLA boots offscreen → client connects → model loads → route finishes → result JSON.
- [ ] Use `pkill CarlaUE4` (NOT `pkill -f`) / `tools/clean_carla.sh` between attempts.

### Phase 4 — Full 220-route eval (parallel tasks)  ⬜ TODO
- [ ] Split routes with `tools/split_xml.py`; **extract AdditionalMaps first** (Town12/13 dominate
      the 220). Run one CARLA+agent per free GPU from local disk.
- [ ] Launch N tasks on staggered ports/GPUs (template in §3, adapted for `qwen_b2d_agent.py`).
- [ ] Wrap in a **restart-until-done loop** (`RESUME=True` continues; CARLA is crash-prone).

### Phase 5 — Metrics  ⬜ TODO
- [ ] Merge shard JSONs: `leaderboard/scripts/merge_statistics.py`.
- [ ] Compute paper metrics: `tools/ability_benchmark.py`,
      `tools/efficiency_smoothness_benchmark.py` (DS/SR/Efficiency/Comfort/Multi-Ability).

---

## 2. The `-graphicsadapter` gotcha (critical for this cluster)

`leaderboard_evaluator.py:208` hardcodes `-graphicsadapter={int(4)}`
(and `leaderboard_evaluator1.py` uses `3`). On this node we want CARLA to render only on
the **2–3 GPUs that support it**. Two things to handle:

1. **It must be a valid adapter index.** Per bench2drive README (line 129), the Vulkan
   adapter index is **not** always equal to the CUDA index — with multiple GPUs the mapping
   can be offset (e.g. CUDA GPU1 ↔ `-graphicsadapter=2`). Find the real mapping with:
   ```bash
   vulkaninfo --summary        # lists GPUs in Vulkan adapter order
   nvidia-smi -L               # lists GPUs in CUDA order
   ```
2. **Make it configurable instead of hardcoded.** Recommended change: have the evaluator read
   the adapter from an env var, e.g.
   ```python
   gadapter = os.environ.get("CARLA_GRAPHICSADAPTER", "0")
   cmd1 = f"{os.path.join(self.carla_path, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port={args.port} -graphicsadapter={gadapter}"
   ```
   Then each parallel task exports its own `CARLA_GRAPHICSADAPTER`.

> Plan for this node: pick the 2–3 GPU indices that pass the Vulkan check and map them to the
> correct `-graphicsadapter` values; run one CARLA server per such GPU.

---

## 3. Parallel launch template (2–3 tasks)

Adapt from `leaderboard/scripts/run_evaluation_multi_uniad.sh`. Skeleton (fill in real
checkpoint, and confirm graphicsadapter mapping first):

```bash
#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/qwen_b2d_agent.py
TEAM_CONFIG=/path/to/your/checkpoint            # e.g. .../checkpoint-20000
PLANNER_TYPE=only_traj
ALGO=qwen
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

# Split 220 routes into N shards once
TASK_NUM=3
python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE

# The GPUs that can render CARLA, and their Vulkan adapter indices (VERIFY with vulkaninfo!)
GPU_RANK_LIST=(0 1 2)          # CUDA index for the model (CUDA_VISIBLE_DEVICES)
GADAPTER_LIST=(0 1 2)          # -graphicsadapter for CARLA (may differ from CUDA index)
TASK_LIST=(0 1 2)

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/eval_${TASK_LIST[$i]}.json"
    export CARLA_GRAPHICSADAPTER=${GADAPTER_LIST[$i]}    # requires the evaluator change in §2
    bash -e leaderboard/scripts/run_evaluation.sh \
        $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG \
        $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE ${GPU_RANK_LIST[$i]} \
        2>&1 > task_${i}.log &
    sleep 60     # stagger CARLA boots; increase on slower machines
done
wait
```

Restart-until-done wrapper (CARLA is crash-prone):
```bash
until bash run_eval_qwen_parallel.sh; do
    echo "eval crashed, cleaning and resuming..."; bash tools/clean_carla.sh; sleep 10
done
```

---

## 4. Quick reference — files & paths

| Thing | Path |
|-------|------|
| Single-task launcher | `bench2drive/leaderboard/scripts/run_evaluation_qwen.sh` |
| Core eval wrapper | `bench2drive/leaderboard/scripts/run_evaluation.sh` |
| Evaluator (boots CARLA) | `bench2drive/leaderboard/leaderboard/leaderboard_evaluator.py` (CARLA launch ~L208) |
| Agent (loads model) | `bench2drive/team_code/qwen_b2d_agent.py` (`init_model` ~L60, `setup` ~L131) |
| Official 220 routes | `bench2drive/leaderboard/data/bench2drive220.xml` |
| Smoke-test routes | `bench2drive/leaderboard/data/routes_devtest.xml` |
| Route splitter | `bench2drive/tools/split_xml.py` |
| Multi-GPU template | `bench2drive/leaderboard/scripts/run_evaluation_multi_uniad.sh` |
| Kill stuck CARLA | `bench2drive/tools/clean_carla.sh` |
| Merge results | `bench2drive/leaderboard/scripts/merge_statistics.py` |
| Metric computation | `bench2drive/tools/ability_benchmark.py`, `bench2drive/tools/efficiency_smoothness_benchmark.py` |
| Eval env reference | `example.txt` (frozen pip list) |

---

## 5. Open questions to resolve before the full run

1. ~~**Transformers/checkpoint loading**~~ ✅ RESOLVED: use **stock transformers** (not the vendored
   `src/transformers`). Agent + `infer_for_debug.py` both use stock
   `from transformers import Qwen2_5_VLForConditionalGeneration` + `model.generate()`; the
   training-only world-model path (`vis_head`/DINOv3, needs `label_bev_masks`) isn't exercised at
   inference, so the `dino*`/`vis_head*` checkpoint keys are ignored. Authors' frozen eval env
   (`example.txt`) used **transformers 4.51.3**.
2. ~~**Vulkan GPU mapping**~~ ✅ RESOLVED: Vulkan adapter index == CUDA index; all 8 A100s render.
3. **Checkpoint location**: scripts point at NAS (`/mnt/nas-data-1/...`). Need the trained
   checkpoint on a reachable path. **STILL OPEN** — ask user / locate the checkpoint to eval.
4. ~~**CARLA version**~~ ✅ RESOLVED: **0.9.15** (bench2drive framework here is pinned to it; no
   0.9.16 refs in `bench2drive/`). The carla python client comes from the 0.9.15 install's egg
   (PYTHONPATH), not pip. (`example.txt`'s pip `carla 0.9.16` is the authors' listing but doesn't
   match this bench2drive checkout.)

### carla-eval env plan (Python 3.10, created)
- carla python: 0.9.15 egg from `$CARLA_ROOT/PythonAPI/carla/dist/` → PYTHONPATH (or carla.pth).
- model deps (stock): torch, **transformers 4.51.3**, `qwen_vl_utils`, accelerate; flash_attn optional.
- bench2drive deps: numpy, opencv-python, scipy, shapely, networkx, py-trees, tabulate, etc.
  (install as import errors surface during the smoke test).

---

## 6. Unblocking the pod (graphics capability) — owner action

### How to find out who controls the pod / how it's launched
```bash
# Confirm it's a k8s pod and get its identifiers
cat /proc/1/cgroup                       # shows .../kubepods.../cri-containerd-<id>
env | grep -E "NVIDIA_DRIVER_CAPABILITIES|HOSTNAME|KUBERNETES_"
# HOSTNAME is usually the pod name. Then, from a machine with kubectl access:
#   kubectl get pod <name> -n <ns> -o yaml      # see the pod spec / who owns it (Deployment/Job)
#   kubectl describe pod <name> -n <ns>         # events, node, owner references
```
If you don't have `kubectl`/cluster access, this is the info to hand your **cluster admin**:
> "The pod is GPU compute-only (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`). I need it
> recreated with **graphics** capability for CARLA/Vulkan: set
> `NVIDIA_DRIVER_CAPABILITIES=all` (or `compute,utility,graphics,display`) in the pod/container
> spec. A100s support Vulkan; this only changes which driver components the runtime exposes."

### Resume checklist (once a graphics-capable pod exists)
1. Verify graphics works:
   ```bash
   echo $NVIDIA_DRIVER_CAPABILITIES        # must include graphics (or =all)
   vulkaninfo --summary | grep -i deviceName   # must list "NVIDIA A100", not just llvmpipe
   ```
   (If the new pod is from the same compute-only image, re-run the install steps in §"What we DID
   install" — ideally bake them into the image instead.)
2. Then proceed Phase 0 → 5 as written above (install CARLA 0.9.15, create the `carla-eval`
   py3.10 env, wire paths, smoke test, parallel 220-route run, metrics).

### Reproducible Vulkan/GL setup performed in this pod (for image baking or a fresh pod)
```bash
# 1) Vulkan loader + tools (LunarG repo gives a current loader; distro 1.3.204 is too old to even diagnose)
apt-get update && apt-get install -y kmod gcc strace
wget -qO /etc/apt/trusted.gpg.d/lunarg-signing-key-pub.asc https://packages.lunarg.com/lunarg-signing-key-pub.asc
wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
apt-get update && apt-get install -y libvulkan1 vulkan-tools
# 2) Matching NVIDIA 550.54.15 userspace GL/Vulkan libs (NO kernel module — must match running driver)
wget https://us.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run
sh NVIDIA-Linux-x86_64-550.54.15.run --silent --no-kernel-module --no-nouveau-check --no-x-check
# 3) modeset node (only needed if the runtime didn't create it; a graphics-capable pod usually does)
#    mknod -m 666 /dev/nvidia-modeset c 195 254
# NOTE: none of this works until the POD has graphics capability (see above). On a compute-only
# pod the NVIDIA Vulkan driver returns VK_ERROR_INITIALIZATION_FAILED regardless of these installs.
```

---

## Closed-loop eval is slow — where the time goes & how to speed it up

**Root cause (not CARLA, not the visualization): the per-tick LLM inference.** The agent calls
`model.generate(max_new_tokens=15000)` on **every** sim tick (`qwen_b2d_agent.py` `run_step`), and
the sim runs at **10 Hz** (`leaderboard_evaluator.py` `frame_rate=10`). So every 0.1 s of sim-time
it does a full **3B-VLM autoregressive generation of up to 15k tokens**, batch=1, HF transformers,
**flash-attention off** (commented out + `flash_attn` not installed). A ~37 s route ≈ 370 ticks ×
seconds each = tens of minutes. CARLA render and frame-saving are negligible next to this.

Note: you **can't** just disable visualization — the agent **saves the 6 surround-camera JPGs to
disk and re-reads them by path** to feed the model (`run_step` → `save_cur_frame` →
`images = [save_path/camera/...]`), so `SAVE_PATH` is part of the **input pipeline**. It currently
writes to **NFS**, which is the only reason "viz" costs anything.

### Levers, by bang-for-buck
| # | Change | Win | Cost / risk |
|---|--------|-----|-------------|
| 1 | **Re-plan every N ticks, not every tick.** Model plans waypoints at 0.5 s intervals for 2 s ahead → one inference already covers ~20 ticks. Infer every ~5 ticks (2 Hz) and let the PID follow the existing waypoints between. | **~5× fewer `generate` calls** (biggest structural win) | agent change; verify DS/SR don't drop |
| 2 | **Lower `max_new_tokens`** (15000 → what's actually emitted, e.g. 2–4k). | large, ~linear in tokens generated | tiny; just measure output length |
| 3 | **Enable flash-attention** — `pip install flash-attn` in `carla-eval`, uncomment `attn_implementation="flash_attention_2"`. | moderate (long BEV+history context) | easy |
| 4 | **`SAVE_PATH` → local disk** (not NFS); better, feed images in-memory and skip the disk roundtrip. | small–moderate (per-tick JPG write+read off NFS) | easy / small code change |
| 5 | **Separate GPUs** for CARLA vs. the model. | modest — they run mostly sequentially per tick (render→infer→tick), so limited overlap | easy |
| 6 | **vLLM backend** (repo's production path: merge weights stripping `dino*`/`vis_head*`, serve). | biggest per-call speedup | most setup; agent calls server vs `model.generate` |
| 7 | **Parallelize across GPU pairs** for the full 220 routes. | wall-clock for the whole benchmark (not a single route) | see §3 multi-launch |

**Suggested order:** cheap+safe first — #4 (local-disk SAVE_PATH) + #3 (flash-attn) + #2
(`max_new_tokens`), then measure. If still slow, do #1 (re-plan cadence — biggest win, needs DS
validation). Reserve #6 (vLLM) for the full benchmark, with #7 for parallelism.
