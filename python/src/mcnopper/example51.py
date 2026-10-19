"""McNopper Example 51: 3D Gaussian splatting (KHR_gaussian_splatting).

Port of the C Example51 (upstream github.com/McNopper/OpenGL, v3.2.7): a
renderer for the KHR_gaussian_splatting glTF 2.0 extension. Splats are
sorted back-to-front every frame by a GPU bitonic sort (two compute shaders)
and drawn as instanced screen-space quads with full spherical-harmonics
colour (bands 0-3). The SH coefficients are stored in the cloud's local
frame; this example rotates them into world space with Wigner-D matrices
built on the CPU (GLUS's ``glus_sh.c`` = the shared ``sh`` module, proven
against the C). Example52 is the same renderer with the other convention.

All four shaders are verbatim copies of the C's (``copy_upstream_shaders.py``),
including their ``#version 460``: compute shaders and SSBOs need a 4.3+
context, so this demo requests 4.6 like example47. As in the C, the splat
stride and SH degree are injected as ``#define``s at compile time, and the
raw interleaved glTF buffer view is uploaded unchanged as the splat SSBO --
the C relies on the attributes being interleaved in exactly the shader's
order (pos, rot, scale, opacity, SH0, SH1..3); the port checks that.

C controls (arrow keys orbit, Page Up/Down zoom, Space auto-orbit) are the
panel's sliders and checkbox. Asset: ``lego.gltf`` + ``lego.bin`` under
``mcnopper/media/`` (or ``$MCNOPPER_MEDIA``).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from mcnopper import common, gltf, sh
from mcnopper.common import GL

WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720

SORT_LOCAL_SIZE: int = 256

ORBIT_SPEED: float = 0.5  # radians per second when auto-orbiting

# std430 layout of the ModelData SSBO, mirrored exactly by the shaders:
# worldMatrix | wigner1 | wigner2 | wigner3 - tightly packed floats.
MODEL_SSBO_SIZE: int = 396

GLTF_MODEL: str = "lego.gltf"

# The shader's float offsets per splat: the order the C assumes the glTF
# attributes are interleaved in (checked at load).
SPLAT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("POSITION", 0),
    ("KHR_gaussian_splatting:ROTATION", 3),
    ("KHR_gaussian_splatting:SCALE", 7),
    ("KHR_gaussian_splatting:OPACITY", 10),
    ("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", 11),
)

state: dict = {}
# Camera orbit state = the C globals; auto-orbit off for reproducible framing.
ui: dict[str, float] = {
    "theta": 0.0,
    "phi": 0.3,
    "radius": 5.0,
    "auto_orbit": 0.0,
}


def _next_power_of_two(n: int) -> int:
    p: int = 1
    while p < n:
        p <<= 1
    return p


def _stride_for_degree(degree: int) -> int:
    """Floats per splat: 14 base + 9/15/21 for SH bands 1/2/3."""
    table: tuple[int, ...] = (14, 23, 38, 59)
    return table[degree] if 0 <= degree <= 3 else 59


def _inject_defines(src: str, stride: int, degree: int | None) -> str:
    """Insert the SPLAT_STRIDE (and SH_DEGREE) defines after ``#version``.

    The C inserts after the file's first line, which is its ``#version``;
    the port's copies carry a provenance comment above it, so anchor on the
    directive itself.
    """
    m: re.Match | None = re.search(r"^#version[^\n]*\n", src, flags=re.MULTILINE)
    if m is None:
        raise ValueError("shader has no #version line")
    define: str = f"#define SPLAT_STRIDE {stride}u\n"
    if degree is not None:
        define += f"#define SH_DEGREE {degree}\n"
    return src[: m.end()] + define + src[m.end() :]


def _read_shader(name: str) -> str:
    with open(common.find_shader(script_file=__file__, filename=name)) as f:
        return f.read()


def _load_splats() -> dict:
    """Read the glTF the way the C does: raw buffer view + node transform."""
    doc, blobs = gltf.load_document(common.find_media(GLTF_MODEL))
    if not doc.meshes or not doc.meshes[0].primitives:
        raise ValueError("glTF has no meshes or primitives")
    prim: Any = doc.meshes[0].primitives[0]
    attrs: dict[str, int] = {
        k: v for k, v in vars(prim.attributes).items() if v is not None
    }
    # Detect SH degree from the attribute names, as the C does.
    degree: int = 0
    for name in attrs:
        if "SH_DEGREE_3" in name:
            degree = 3
        elif "SH_DEGREE_2" in name:
            degree = max(degree, 2)
        elif "SH_DEGREE_1" in name:
            degree = max(degree, 1)
    stride: int = _stride_for_degree(degree)
    if "POSITION" not in attrs:
        raise ValueError("no POSITION attribute in glTF primitive")
    # The shaders index one interleaved float array, so the attributes must
    # sit at the offsets the C assumes (it does not check; the port does).
    pos_off: int = doc.accessors[attrs["POSITION"]].byteOffset or 0
    for name, floats in SPLAT_LAYOUT:
        if name in attrs:
            off: int = doc.accessors[attrs[name]].byteOffset or 0
            if off - pos_off != floats * 4:
                raise ValueError(
                    f"{name} at byte {off - pos_off}, expected {floats * 4}"
                )
    data, view_stride = gltf.accessor_view_bytes(
        gltf=doc, blobs=blobs, accessor_index=attrs["POSITION"]
    )
    byte_stride: int = view_stride or stride * 4
    if byte_stride != stride * 4:
        raise ValueError(
            f"byteStride {byte_stride} != {stride * 4} (SH degree {degree})"
        )
    count: int = doc.accessors[attrs["POSITION"]].count
    # World matrix of the node that carries mesh 0 (identity if none).
    graph: gltf.Scene = gltf.load_node_graph(doc)
    world: np.ndarray = np.eye(4, dtype=np.float32)
    for node in graph.nodes:
        if node.mesh == 0:
            world = node.world
            break
    return {
        "data": data,
        "count": count,
        "degree": degree,
        "stride": stride,
        "world": world,
    }


def _wigner_matrices(world: np.ndarray, degree: int) -> np.ndarray:
    """The three Wigner-D bands for the node rotation, 3DGS basis (= C).

    Returns the 83 floats the ModelData SSBO holds after the world matrix:
    wigner1[9] | wigner2[25] | wigner3[49], column-major as the C stores
    them. The per-splat SH coefficients live in the cloud's local frame, so
    the matrices are built from the *inverse* node rotation, then each band
    is conjugated by the Condon-Shortley phase diag((-1)^m) to bridge the
    standard real-SH basis and the glTF/Inria convention the shader uses.
    """
    # Normalised rotation columns of the world matrix, column-major flat.
    cols: np.ndarray = world[:3, :3].astype(np.float32)
    norms: np.ndarray = np.linalg.norm(cols, axis=0)
    norms[norms <= 1e-6] = 1.0
    rotation: np.ndarray = (cols / norms).flatten(order="F")
    # Inverse node rotation R^T, column-major flat.
    rotation_inv: np.ndarray = (cols / norms).T.flatten(order="F")

    w1: np.ndarray = np.eye(3, dtype=np.float32).flatten(order="F")
    w2: np.ndarray = np.eye(5, dtype=np.float32).flatten(order="F")
    w3: np.ndarray = np.eye(7, dtype=np.float32).flatten(order="F")
    if degree >= 1:
        sh.sh_build_rotation1(result=w1, rotation=rotation_inv)
    if degree >= 2:
        sh.sh_build_rotation2(result=w2, sh1=w1)
    if degree >= 3:
        sh.sh_build_rotation3(result=w3, sh1=w1, sh2=w2)
    # Conjugate each band by S = diag((-1)^m), m = -l..+l (index i -> m = i-l).
    for band_l, m in ((1, w1), (2, w2), (3, w3)):
        if band_l > degree:
            break
        n: int = 2 * band_l + 1
        for c in range(n):
            sc: float = -1.0 if (c - band_l) & 1 else 1.0
            for r in range(n):
                sr: float = -1.0 if (r - band_l) & 1 else 1.0
                m[c * n + r] *= sr * sc
    del rotation  # the C keeps it only to derive rotation_inv
    return np.concatenate([w1, w2, w3]).astype(np.float32)


def _make_ssbo(binding: int, nbytes: int, data=None, usage=None) -> int:
    buf: int = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buf)
    GL.glBufferData(
        target=GL.GL_SHADER_STORAGE_BUFFER,
        size=nbytes,
        data=data,
        usage=usage if usage is not None else GL.GL_DYNAMIC_DRAW,
    )
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, buf)
    return buf


def init() -> bool:
    print("[Example 51] Loading splats...")
    splats: dict = _load_splats()
    count: int = splats["count"]
    padded: int = max(_next_power_of_two(count), SORT_LOCAL_SIZE)
    stride: int = splats["stride"]
    degree: int = splats["degree"]
    world: np.ndarray = splats["world"]
    print(
        f"[Example 51] SH degree {degree}, stride {stride} floats, "
        f"{count} splats (padded {padded})."
    )

    # binding 0: splat float data (the raw interleaved buffer view).
    raw: np.ndarray = np.frombuffer(splats["data"], dtype=np.uint8)
    splat_ssbo: int = _make_ssbo(
        binding=0, nbytes=raw.nbytes, data=raw, usage=GL.GL_STATIC_DRAW
    )
    # binding 1/2: sorted indices and view-space distances, padded size.
    index_ssbo: int = _make_ssbo(binding=1, nbytes=padded * 4)
    depth_ssbo: int = _make_ssbo(binding=2, nbytes=padded * 4)
    # binding 3: world matrix (column-major) + Wigner-D matrices.
    model: np.ndarray = np.concatenate(
        [world.T.flatten(), _wigner_matrices(world=world, degree=degree)]
    ).astype(np.float32)
    assert model.nbytes == MODEL_SSBO_SIZE
    model_ssbo: int = _make_ssbo(
        binding=3, nbytes=model.nbytes, data=model, usage=GL.GL_STATIC_DRAW
    )

    # UBO binding 0: WorldData (std140: proj, view, focal, viewport, camPos).
    world_ubo: int = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, world_ubo)
    GL.glBufferData(
        target=GL.GL_UNIFORM_BUFFER,
        size=160,
        data=None,
        usage=GL.GL_DYNAMIC_DRAW,
    )
    GL.glBindBufferBase(GL.GL_UNIFORM_BUFFER, 0, world_ubo)
    GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    prog_depth: int = common.compile_program(
        comp_src=_inject_defines(
            src=_read_shader("depth.comp.glsl"), stride=stride, degree=None
        )
    )
    prog_sort: int = common.compile_program(
        comp_src=_read_shader("sort.comp.glsl")
    )
    prog_splat: int = common.compile_program(
        vert_src=_inject_defines(
            src=_read_shader("splat.vert.glsl"), stride=stride, degree=degree
        ),
        frag_src=_read_shader("splat.frag.glsl"),
    )

    # Quad corners in NDC, instanced per splat.
    quad: np.ndarray = np.array(
        [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32
    )
    quad_vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(quad_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, common.make_vbo(quad))
    GL.glVertexAttribPointer(
        index=0,
        size=2,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(0)
    GL.glBindVertexArray(0)

    # Global GL state: no depth, premultiplied-alpha blending over black.
    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)

    state.update(
        count=count,
        padded=padded,
        degree=degree,
        world=world,
        splat_ssbo=splat_ssbo,
        index_ssbo=index_ssbo,
        depth_ssbo=depth_ssbo,
        model_ssbo=model_ssbo,
        world_ubo=world_ubo,
        prog_depth=prog_depth,
        prog_sort=prog_sort,
        prog_splat=prog_splat,
        quad_vao=quad_vao,
        depth_world=common.uniform(prog=prog_depth, name="u_worldMatrix"),
        depth_n=common.uniform(prog=prog_depth, name="u_numSplats"),
        depth_np=common.uniform(prog=prog_depth, name="u_numSplatsPadded"),
        sort_j=common.uniform(prog=prog_sort, name="u_j"),
        sort_k=common.uniform(prog=prog_sort, name="u_k"),
    )
    print("[Example 51] Ready.")
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, max(h, 1)
    GL.glViewport(0, 0, w, h)
    state["proj"] = common.perspective(
        fovy_deg=45.0, aspect=w / max(h, 1), near=0.1, far=1000.0
    )


def update(dt: float) -> bool:
    if "proj" not in state:
        reshape(w=WINDOW_WIDTH, h=WINDOW_HEIGHT)
    w, h = state["w"], state["h"]
    if ui["auto_orbit"]:
        ui["theta"] += dt * ORBIT_SPEED

    # Camera orbit.
    theta: float = ui["theta"]
    phi: float = ui["phi"]
    radius: float = ui["radius"]
    eye: list[float] = [
        radius * np.cos(phi) * np.sin(theta),
        radius * np.sin(phi),
        radius * np.cos(phi) * np.cos(theta),
    ]
    proj: np.ndarray = state["proj"]
    view: np.ndarray = common.look_at(eye=eye, center=[0, 0, 0], up=[0, 1, 0])

    # WorldData UBO (matrices column-major; focal = proj diag * half size).
    world_data: np.ndarray = np.concatenate(
        [
            proj.T.flatten(),
            view.T.flatten(),
            [proj[0, 0] * w * 0.5, proj[1, 1] * h * 0.5, float(w), float(h)],
            [eye[0], eye[1], eye[2], 1.0],
        ]
    ).astype(np.float32)
    GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, state["world_ubo"])
    GL.glBufferSubData(
        target=GL.GL_UNIFORM_BUFFER,
        offset=0,
        size=world_data.nbytes,
        data=world_data,
    )
    GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    # Depth init compute pass.
    padded: int = state["padded"]
    groups: int = padded // SORT_LOCAL_SIZE
    GL.glUseProgram(state["prog_depth"])
    common.upload_matrix(loc=state["depth_world"], m=state["world"])
    GL.glUniform1ui(state["depth_n"], state["count"])
    GL.glUniform1ui(state["depth_np"], padded)
    GL.glDispatchCompute(groups, 1, 1)
    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)

    # Bitonic sort, descending by view-space depth (back-to-front).
    GL.glUseProgram(state["prog_sort"])
    k: int = 2
    while k <= padded:
        j: int = k >> 1
        while j >= 1:
            GL.glUniform1ui(state["sort_j"], j)
            GL.glUniform1ui(state["sort_k"], k)
            GL.glDispatchCompute(groups, 1, 1)
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            j >>= 1
        k <<= 1

    # Splat render pass.
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog_splat"])
    GL.glBindVertexArray(state["quad_vao"])
    GL.glDrawArraysInstanced(GL.GL_TRIANGLE_STRIP, 0, 4, state["count"])
    GL.glBindVertexArray(0)
    GL.glUseProgram(0)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["theta"] = imgui.slider_float(
        label="orbit (theta)", v=ui["theta"], v_min=-3.2, v_max=3.2
    )
    _, ui["phi"] = imgui.slider_float(
        label="elevation (phi)", v=ui["phi"], v_min=-1.5, v_max=1.5
    )
    _, ui["radius"] = imgui.slider_float(
        label="radius", v=ui["radius"], v_min=0.5, v_max=50.0
    )
    _, auto = imgui.checkbox(label="auto-orbit", v=bool(ui["auto_orbit"]))
    ui["auto_orbit"] = float(auto)
    imgui.text(
        f"splats: {state.get('count', 0)}  SH degree: {state.get('degree', 0)}"
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 51 - Gaussian splatting",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=6,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
