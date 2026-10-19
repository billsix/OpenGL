# Python port — fidelity audit vs the C originals (L2)

Up: `README.md`. **What is TRUE about each Python demo relative to its C original** — the
`ExampleNN/src/main.c` (+ `ExampleNN/shader/` + GLUS source) that sits beside it in this
repository, measured at upstream **v3.2.7** — compared for: constants and resources, render
structure and GL state, uniforms/attributes, animation, controls (C keys → imgui widgets,
`c-to-python-rules.md` §3), simplifications, and outright bugs. Python paths are
`python/src/mcnopper/exampleNN.py` (example13/48/49/50 nested as
`exampleNN/exampleNN.py`). **Line numbers are as of when the audit was
taken and will rot; the file + symbol is the anchor.**

**Verdict scale:** FAITHFUL (same picture, same controls) · MINOR DRIFT (cosmetic/constant
differences a viewer might notice) · FUNCTIONAL GAP (a feature/control/pass missing or
approximated) · BROKEN (cannot be producing the C's picture).

**Status:** every demo below is now FAITHFUL on the maintainer's display. All of 01–47 were
audited (by six read-only reader passes, one per batch of eight) and every finding was
fixed; a display pass of all 52 confirmed the fixes. Examples 48–52 (glTF/PBR/IBL and
Gaussian splatting) were added with the v3.2.7 resync and verified on the display at the
same time. The scoreboard and per-demo findings are kept as the record of what the audit
found — the state BEFORE the fixes — and as the latent-risk list.

## Scoreboard (47 of 47 audited in depth)

| Verdict | Demos |
|---|---|
| FAITHFUL (14) | 01, 05, 07, 08, 09, 10, 16, 17, 18, 21, 29, 38, 39, 44 |
| MINOR DRIFT (16) | 02, 03, 04, 06, 11, 12, 13, 15, 20, 23, 27, 34, 35, 36, 40, 47 |
| FUNCTIONAL GAP (11) | 14, 22, 24, 25, 26, 28, 30, 31, 32, 41, 46 |
| BROKEN (6) | 19, 33, 37, 42, 43, 45 |

## Cross-cutting root causes (fixed once, in `common.py`)

These account for most of the drift; the fix is in the shared helper, not the call sites.

1. **Window size.** Every C demo in 01–24 opens **640×480**; 33/35/36/40/41/42/46 open
   **1024×768**; 45 opens 512²; 47 1024×768. `run_demo` defaults to 1280×720, so a demo
   that passes nothing is letterboxed. Cosmetic for fixed-vertical-FOV demos, but it
   *changes the picture* for NDC/ortho/fixed-FBO demos (02, 03, 19, 21, 28/30/32/33/35/42).
   Fix: each port passes its C `width=`/`height=` (and `samples=`: 25/26 want 8, 41 wants 4,
   43 wants 8).
2. **`make_torus(a, b, …)` semantics.** GLUS `glusShapeCreateTorusf(inner, outer, …)` has
   tube = (outer − inner)/2, ring = outer − tube. The port originally read a = tube, b =
   ring, so callers passing GLUS-style args got tori 25–50 % wider with tubes up to 2×
   fatter. Fix: `make_torus` uses GLUS inner/outer semantics; call sites stay.
3. **`make_rectangular_grid_plane(w, h, …)` takes HALF extents; GLUS takes FULL.** Callers
   passing GLUS-style full extents got grid planes 2× too large (41's camera mid-ocean, 40's
   cloth twice the sphere). Fix: full-extent semantics; call sites stay.
4. **`make_cube` uv table ≠ GLUS's** (`glusShapeCreateCubef` `cubeTexCoords`): ±X/±Y faces
   show textures rotated 90° (06). Fix: transcribe GLUS's 24-vertex tables.
5. **`load_obj` merges all `usemtl` groups and ignored the `.mtl`.** 25, 31, 43, 45, 46 need
   group ranges + `.mtl` parsing + per-group draws with the group's texture. One shared fix:
   `ObjGroup`/`ObjMaterial`/`load_mtl` + `draw_obj_group`.
6. **Port-added defaults that differ from the C:** 04 and 42 auto-rotate (C static), 46
   sphere orbit runs (C paused), 47 reshuffle auto-starts the sort (C waits for Space).
7. **Docstrings** of 01–24 cited `/nehe/ExampleNN`; the originals are McNopper (cosmetic).
8. **Shader drift (mechanical diff, 61 files):** 52 whitespace-only; of the content diffs,
   only `30/raytrace.comp` carried real missed upstream fixes (TIR → reflect fallback
   instead of `normalize(vec3(0))` = NaN; shadow-ray distance bound). 13's
   `tessellation.cont` is the one deliberate port edit (`c-to-python-rules.md` §6). Net: the
   audit reduced to re-running the shader copy on every demo plus the 13 decision.

## Per-demo findings

### 01–08 (confidence high throughout)

- **01 FAITHFUL.** Window only. Port re-issues `glClearColor` per frame for its colour picker.
- **02 MINOR DRIFT.** NDC triangle; window aspect is the picture (C `main.c:181` 640×480).
- **03 MINOR DRIFT.** `desert.tga` 1024×768: C centre-crops at 640×480, port letterboxes at
  1280×720. Low-confidence risk: `reshape` uploaded the MVP without `glUseProgram` (05–08
  guard this) — add `GL.glUseProgram(state["prog"])`.
- **04 MINOR DRIFT.** C model is a static `RotateRzRxRy(0,45,45)`; port defaulted
  `auto_rotate: True` at 30°/s. Default it off.
- **05 FAITHFUL.** Window only.
- **06 MINOR DRIFT.** `make_cube` uv mismatch (root cause 4): crate texture rotated 90° on
  ±X/±Y faces; with the C's Rx(45)·Ry(angle) pose those are the faces you see.
- **07 FAITHFUL.** Tangent frame derived on both sides and equal.
- **08 FAITHFUL.** Only positions+normals of the cube are used, so the uv mismatch is moot.

### 09–16 (confidence high)

- **09 FAITHFUL.** Particle system matches line for line (16×16, RGBA32F ping-pong, blend).
- **10 FAITHFUL.**
- **11 MINOR DRIFT.** Torus (root cause 2): C `(0.25, 1.0)` = tube 0.375/ring 0.625; port =
  tube 0.25/ring 1.0 (25 % larger, thinner, bigger hole).
- **12 MINOR DRIFT.** Torus (root cause 2). Also: shadow texture left bound on unit 0 while
  it is the depth attachment (C unbinds, `main.c:428`); dead shadow VAO.
- **13 MINOR DRIFT.** `tessellation.cont.glsl` deliberately changed (root cause 8);
  `wireframe` checkbox is a no-op (geometry stage emits `line_strip`).
- **14 FUNCTIONAL GAP.** Top view renders the wrong picture: C feeds pass-one culling and the
  pass-two LOD centre from the *person* camera (`main.c:790-820`) and only the view matrix
  from the active view; port overwrote `cam_pos/cam_dir` with the top view first → static
  uniform-LOD disc instead of the travelling FOV wedge.
- **15 MINOR DRIFT.** `Water.frag.glsl` diff is comment-only. Background sphere drawn with
  culling disabled + a stale comment; C uses `glFrontFace(GL_CW)` with culling on
  (`renderBackground.c:171`). Visually identical.
- **16 FAITHFUL.**

### 17–24 (confidence high except 22 medium)

- **17 FAITHFUL.** Port adds a clip-plane-height slider.
- **18 FAITHFUL.**
- **19 BROKEN.** `_draw_obj` uploaded `proj @ view` to `u_projectionMatrix` *and* `view @
  model` to the modelview → P·V·V·M (monkey too far, half size, pitched twice).
  Projective-texture bias matrix built from the window projection instead of the mirror
  FBO's → reflection misregistered. Mirror RTT 1024² vs C's window-sized 640×480.
- **20 MINOR DRIFT.** Monkey missing `Translate(0,-1.5,0)` (`wavefront.c:196-197`); material
  magenta in C, cyan in port.
- **21 FAITHFUL** (window: pixel-ortho quad fills a 640×480 window only).
- **22 FUNCTIONAL GAP.** Torus (root cause 2). `_build_adjacency` keys on indices only; GLUS
  falls back to position matching (`glus_shape_adjacency.c:266-276`), so the torus seams
  produce open shadow volumes → stencil streaks.
- **23 MINOR DRIFT.** Grid plane 2× too large (root cause 3; C `(3,3,16,16)` → ±1.5).
- **24 FUNCTIONAL GAP.** `_perlin_3d` was nearest-neighbour-upsampled white noise, min-max
  normalised; C `glusPerlinCreateNoise3D(64³, seed 0, freq 8, amp 192, persist 0.5, 4
  octaves)` is trilinearly interpolated lattice noise, truncated to bytes, not normalised.
  Blocky pits vs smooth. Port the GLUS routine.

### 25–32 (confidence high, 29 medium-high)

- **25 FUNCTIONAL GAP.** Per-group materials dropped (root cause 5): C draws Sub0 with
  `ChessKing.tga` and Sub1 with `ChessColumn.tga` (`main.c:233-267`); port one mesh, one
  texture → column textured with the king's image, model darker. Window 640×480.
- **26 FUNCTIONAL GAP.** **Texgen axis bug:** C `glusShapeTexGenByAxesf(bunny, 2,0, 2,0,
  0,0)` = `s = 2x, t = 2y`; port built `t` from z → stripes run the wrong way. Fur strength
  noise and mipmapping also differed. Window 640×480 + 8 MSAA not passed.
- **27 MINOR DRIFT.** Torus (root cause 2). Window 640×480.
- **28 FUNCTIONAL GAP.** Wrong model (`monkey.obj` vs C `teapot.obj`, `wavefront.c:126`);
  material cyan in C vs pink; 1024² square G-buffer with aspect 1 blitted over 16:9.
  `ssao.frag.glsl` 1-line diff is comment-only.
- **29 FAITHFUL.** Two latent divergences with no effect on this scene: shadow occluders not
  bounded to `0 < t <= lightDistance` (C `main.c:265-276`); `_intersect` rejects `tca < 0`
  rays that start inside a sphere (GLUS `glus_intersect.c:40` does not). Window passed.
- **30 FUNCTIONAL GAP.** `raytrace.comp.glsl` **missed two upstream fixes** (root cause 8):
  TIR → `reflect` fallback (port had `normalize(refract(...))` = NaN at grazing exits) and
  the shadow-ray `t0 <= lightDistance` bound. Re-copy. 640×480 image stretched.
- **31 FUNCTIONAL GAP.** Pawn textures/materials dropped (root cause 5): C draws each
  `ChessPawn.obj` group with `ChessPawn.tga`/`ChessColumn.tga` and `u_useTexture = 1`; port
  grey, untextured. Window 1024×768.
- **32 FUNCTIONAL GAP.** **Depth test never enabled** (C `main.c:443`, toggled around the
  resolve; port enabled culling only) → Venus overdraws in index order. 4:3 MSAA FBO
  stretched onto 16:9. The IBL path matches the C 32 (direct importance sampling of the
  RGB32F panorama; GLUS's cubemap prefilter is not used by the C either).

### 33–40 (confidence high)

- **33 BROKEN.** `GL_DEPTH_TEST` never enabled (C `main.c:759`) → Venus overdraws in index
  order. 4:3 FBO stretched onto the 16:9 default window. Colour keys `5`–`0` had no widget.
  **IBL build itself matches the C 33** (own prefilter shaders sampling the 2-D panorama).
- **34 MINOR DRIFT.** Slider ranges ≠ C clamps; depth texture left bound as attachment; window.
- **35 MINOR DRIFT.** Window (C 1024×768 = peel-texture size); otherwise verified equal.
- **36 MINOR DRIFT.** Linked-list SSBO sized `MAX_NODES*24+8` vs std430 stride 32 (C
  `main.c:263`) → held 0.75·MAX_NODES nodes while `u_maxNodes` said MAX_NODES (latent
  overflow). Context 4.3 requested for `#version 440` shaders (C 4.4).
- **37 BROKEN.** Image upside down (`ndc_y` sign; GLUS fills rows bottom-up
  `glus_raytrace.c:55`); oriented box drawn axis-aligned (`_box_dist` discarded the 20° Y
  orientation); background grey in C vs blue; NEAREST vs LINEAR filter.
- **38 FAITHFUL. 39 FAITHFUL.** (window 640×480.)
- **40 MINOR DRIFT.** Cloth grid 2× too large (root cause 3; C `(2,2)` → ±1); sphere 64
  slices in C vs 32; window 1024×768; `cloth.comp.glsl` brace/comment drift only.

### 41–47 (confidence high; 46 medium on the GI tint)

- **41 FUNCTIONAL GAP.** Ocean grid 2× too large (root cause 3): camera mid-ocean,
  wavelengths doubled. Window 1024×768 + 4 MSAA not passed.
- **42 BROKEN.** Wrong model (`monkey.obj` vs C `elephant.obj`, `wavefront.c:126`); missing
  `Translate(0,-1.5,0)`; material differs; 4:3 FBO stretched to 16:9; auto-rotate inverted.
- **43 BROKEN.** Per-group materials dropped (root cause 5): C draws Cone green / Sphere red /
  Cube blue from `three_objects.mtl`; port one grey mesh. Camera differs. Window 640×480.
- **44 FAITHFUL.**
- **45 BROKEN.** The voxel colour *is* `texture(u_diffuse)` (`voxelize.frag.glsl:124`); C
  loads each material's `map_Kd` and binds per group; port never loaded a texture → black
  silhouette. Needs root cause 5.
- **46 FUNCTIONAL GAP.** All 25 Sponza materials (24 textured) replaced by the C's
  no-material fallback grey (root cause 5); sphere orbit default running (C paused); W/S/A/D
  walk has no widget.
- **47 MINOR DRIFT.** Camera differs; reshuffle auto-starts the sort (C waits for Space);
  step-delay range 0..1000.

## Fix status (all applied; display-checked)

| Finding | Fix |
|---|---|
| Root cause 1, window size | every port passes its C `width`/`height`/`samples` |
| Root cause 2, `make_torus` | GLUS inner/outer semantics + GLUS vertex/index order + tangents |
| Root cause 3, grid plane | GLUS full extents + GLUS top-row-first layout (40/41's shaders index it) |
| Root cause 4, `make_cube` | GLUS's 24-vertex tables, array-equal to `glus_shape.c` |
| Root cause 5, `load_obj` | `ObjGroup`/`ObjMaterial`/`load_mtl`; 25/31/43/45/46 draw per group via `draw_obj_group` |
| Root cause 6, defaults | 04/42 static, 46 orbit paused, 47 reshuffle waits for the sort button |
| Root cause 7, docstrings | `/nehe/` → `McNopper/OpenGL` |
| Root cause 8, shaders | all shaders re-copied verbatim; 13's `u_tessLevel` slider kept as the one documented port edit |
| 03 | `glUseProgram` in reshape |
| 12, 34 | depth texture unbound before the pass that renders into it |
| 14 | culling + LOD inputs from the person camera; active view only picks the view matrix |
| 15 | background drawn with `GL_CW` front face and culling on |
| 19 | projection-only upload, bias from the mirror pass's VP, 640×480 mirror FBO |
| 20, 42 | `Translate(0,−1.5,0)`; corrected materials; 42 elephant |
| 22 | GLUS adjacency: by index, then by position within 0.001 (seams close) |
| 24, 26 | `perlin.py` = `glus_perlin.c` incl. glibc `rand()`; 26 texgen s=2x, t=2y; strength map mipmapped |
| 25, 31 | per-group materials + `map_Kd` textures |
| 28 | teapot, cyan, 640×480 G-buffer |
| 29 | GLUS ray-sphere semantics; shadow occluders bounded to 0 < t ≤ light distance |
| 32, 33 | `GL_DEPTH_TEST` on/off around the resolve; slider floors; 33 colour-material widget |
| 36 | linked-list SSBO 32 B per node |
| 37 | rewritten to the C's `march` (GLUS rays, oriented boxes, 0.8 background, NEAREST), pixel-equal ±1 to a C build |
| 40 | sphere 64 slices |
| 43 | per-group `.mtl` colours |
| 45 | per-group `map_Kd` on unit 1 |
| 46 | per-group `.mtl` materials/textures in both passes, anisotropy |
| 47 | camera matched, step delay 0–1000 |

Left as-is on purpose: 21's pixel-ortho quad (window size fixes it); 32/33's exposure/gamma
sliders (the C's shader has the uniforms; `c-to-python-rules.md` §3 keeps adjustable
parameters as widgets); 41's NEAREST image textures (no visible effect); 46's W/S/A/D walk
(yaw/pitch/eye widgets exist).

## How to use this doc

Every fix is Python-side; the shaders are verbatim (`c-to-python-rules.md` §6) except 13's
recorded edit. If a demo is later found to differ from the C, record the new finding here
and open a task for it.
