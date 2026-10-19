"""McNopper Example 44: Conservative rasterization via geometry shader.

Original: McNopper/OpenGL Example44. The geom stage expands every triangle by a
half-pixel in NDC so any pixel even partially covered emits a fragment
- a software emulation of GL_NV_conservative_raster.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def init() -> bool:
    verts: np.ndarray = np.array(
        [
            -10,
            -5,
            0,
            1,
            10,
            5,
            0,
            1,
            0,
            10,
            0,
            1,
        ],
        dtype=np.float32,
    )
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="conservative.vert.glsl",
        geom="conservative.geom.glsl",
        frag="conservative.frag.glsl",
    )
    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo: int = common.make_vbo(verts)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo, name="a_vertex", size=4)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 50], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )

    GL.glClearColor(0.2, 0.2, 0.2, 1.0)
    state.update(
        prog=prog,
        vao=vao,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        hp_loc=common.uniform(prog=prog, name="u_halfPixelSize"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    GL.glUniform2f(state["hp_loc"], 1.0 / w, 1.0 / h)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("Geometry shader expands triangle by half-pixel in NDC.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 44 - conservative raster",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
