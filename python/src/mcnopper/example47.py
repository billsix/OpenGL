"""McNopper Example 47: 3D spatial color sort via compute shader.

Original: McNopper/OpenGL Example47 (GL 4.6). N^3 = 32^3 RGB-cube colors are
shuffled, then an odd-even transposition sort runs along each axis
(X sorts R, Y sorts G, Z sorts B). After GRID_N * 3 visible steps the
cube has crystallized from random noise into the canonical RGB cube,
rendered as a point cloud.
"""

from __future__ import annotations

import math
import re

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WINDOW_WIDTH: int = 1024
WINDOW_HEIGHT: int = 768
GRID_N: int = 32
ORBIT_SPEED: float = 18.0
state: dict = {}
ui: dict = {
    "sorting": True,
    "step_delay_ms": 100,
    "orbit_angle": 0.0,
    "sort_step": 0,
    "accum": 0.0,
}


def _shuffled_colors(rng: np.random.Generator) -> np.ndarray:
    n: int = GRID_N
    out: np.ndarray = np.zeros((n, n, n, 4), dtype=np.uint8)
    for z in range(n):
        for y in range(n):
            for x in range(n):
                out[z, y, x] = (
                    x * 255 // (n - 1),
                    y * 255 // (n - 1),
                    z * 255 // (n - 1),
                    255,
                )
    flat: np.ndarray = out.reshape(-1, 4)
    perm: np.ndarray = rng.permutation(len(flat))
    flat = flat[perm].copy()
    return flat.reshape(n, n, n, 4)


def _inject_grid_n(src: str) -> str:
    """Insert the GRID_N define right after the ``#version`` line.

    The C inserts after the file's first line, which is its ``#version``;
    the port's copy carries a provenance comment above it, so find the
    line instead of assuming its position.
    """
    m: re.Match | None = re.search(
        r"^#version[^\n]*\n", src, flags=re.MULTILINE
    )
    if m is None:
        raise ValueError("shader has no #version line")
    return src[: m.end()] + f"#define GRID_N {GRID_N}\n" + src[m.end() :]


def init() -> bool:
    rng: np.random.Generator = np.random.default_rng(42)

    sort_src_path: str = common.find_shader(
        script_file=__file__, filename="sort.comp.glsl"
    )
    with open(sort_src_path) as f:
        sort_src: str = _inject_grid_n(f.read())
    prog_sort: int = common.compile_program(comp_src=sort_src)
    prog_render: int = common.compile_program_files(
        script_file=__file__, vert="render.vert.glsl", frag="render.frag.glsl"
    )

    GL.glUseProgram(prog_render)
    GL.glUniform1ui(common.uniform(prog=prog_render, name="u_gridN"), GRID_N)
    GL.glUniform1f(common.uniform(prog=prog_render, name="u_pointSize"), 8.0)
    GL.glUniform1i(common.uniform(prog=prog_render, name="u_colorTexture"), 0)

    color_data: np.ndarray = _shuffled_colors(rng)
    color_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, color_tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D,
        0,
        GL.GL_RGBA8,
        GRID_N,
        GRID_N,
        GRID_N,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        color_data,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glBindTexture(GL.GL_TEXTURE_3D, 0)

    vao: int = GL.glGenVertexArrays(1)

    GL.glClearColor(0, 0, 0, 1)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

    state.update(
        prog_sort=prog_sort,
        prog_render=prog_render,
        vao=vao,
        color_tex=color_tex,
        rng=rng,
        axis_loc=common.uniform(prog=prog_sort, name="u_axis"),
        pass_loc=common.uniform(prog=prog_sort, name="u_pass"),
        mvp_loc=common.uniform(prog=prog_render, name="u_mvpMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    ui["orbit_angle"] = (ui["orbit_angle"] + ORBIT_SPEED * dt) % 360.0
    total_steps: int = GRID_N * 3

    if ui["sorting"]:
        steps_this_frame: int = 0
        if ui["step_delay_ms"] > 0:
            ui["accum"] += dt
            if ui["accum"] >= ui["step_delay_ms"] * 0.001:
                steps_this_frame = 1
                ui["accum"] = 0.0
        else:
            steps_this_frame = 3

        if steps_this_frame:
            GL.glUseProgram(state["prog_sort"])
            GL.glBindImageTexture(
                0,
                state["color_tex"],
                0,
                GL.GL_TRUE,
                0,
                GL.GL_READ_WRITE,
                GL.GL_RGBA8,
            )
            for _ in range(steps_this_frame):
                if ui["sort_step"] >= total_steps:
                    ui["sorting"] = False
                    break
                axis: int = ui["sort_step"] % 3
                for p in range(GRID_N):
                    GL.glUniform1i(state["axis_loc"], axis)
                    GL.glUniform1i(state["pass_loc"], p)
                    GL.glDispatchCompute(GRID_N, GRID_N, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
                ui["sort_step"] += 1
            GL.glBindImageTexture(
                0, 0, 0, GL.GL_TRUE, 0, GL.GL_READ_WRITE, GL.GL_RGBA8
            )

    GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    rad: float = math.radians(ui["orbit_angle"])
    eye_x: float = math.sin(rad) * 3.5
    eye_z: float = math.cos(rad) * 3.5
    view: np.ndarray = common.look_at(
        eye=[eye_x, 2.5, eye_z], center=[0, 0, 0], up=[0, 1, 0]
    )
    w, h = state.get("w", WINDOW_WIDTH), state.get("h", WINDOW_HEIGHT)
    proj: np.ndarray = common.perspective(
        fovy_deg=60.0, aspect=w / max(h, 1), near=0.1, far=100.0
    )

    GL.glUseProgram(state["prog_render"])
    common.upload_matrix(loc=state["mvp_loc"], m=proj @ view)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, state["color_tex"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_POINTS, 0, GRID_N**3)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    if imgui.button("reshuffle"):
        data: np.ndarray = _shuffled_colors(state["rng"])
        GL.glBindTexture(GL.GL_TEXTURE_3D, state["color_tex"])
        GL.glTexSubImage3D(
            GL.GL_TEXTURE_3D,
            0,
            0,
            0,
            0,
            GRID_N,
            GRID_N,
            GRID_N,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_3D, 0)
        ui["sort_step"] = 0
        ui["sorting"] = False  # C: reshuffle resets; the sort waits for Space
    imgui.same_line()
    if imgui.button("start sort" if not ui["sorting"] else "running..."):
        if not ui["sorting"]:
            ui["sorting"] = True
            ui["sort_step"] = 0
    _, ui["step_delay_ms"] = imgui.slider_int(
        label="step delay (ms)", v=ui["step_delay_ms"], v_min=0, v_max=1000
    )
    imgui.text(f"step {ui['sort_step']}/{GRID_N * 3}")
    imgui.text(f"grid: {GRID_N}^3 = {GRID_N**3:,} colors")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 47 - color sort",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=6,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
