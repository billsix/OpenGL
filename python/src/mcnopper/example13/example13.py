"""McNopper Example 13: Hardware tessellation (vert/cont/eval/geom/frag).

Original: McNopper/OpenGL Example13. A 4-vertex patch is tessellated by the
TCS/TES, amplified by the geometry stage into line strips, and shaded flat.
The C hard-codes level 4; the port's control shader takes it from the
``u_tessLevel`` uniform so the imgui slider can change it (the one port-side
shader edit, see c-to-python-rules.md section 8).
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"tess_level": 4}  # C: 4.0


def init() -> bool:
    plane: common.Shape = common.make_plane(1.0)
    plane_indices: np.ndarray = np.array([0, 1, 3, 2], dtype=np.uint32)

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="tessellation.vert.glsl",
        tesc="tessellation.cont.glsl",
        tese="tessellation.eval.glsl",
        geom="tessellation.geom.glsl",
        frag="tessellation.frag.glsl",
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(plane.vertices)
    vbo_i: int = common.make_vbo(
        data=plane_indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_modelViewMatrix"), m=view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=4,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        tess_loc=common.uniform(prog=prog, name="u_tessLevel"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glUniform1f(state["tess_loc"], float(ui["tess_level"]))
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_PATCHES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("Patch (4 verts) -> TCS -> TES -> GS -> FS")
    _, ui["tess_level"] = imgui.slider_int(
        label="tess level", v=ui["tess_level"], v_min=1, v_max=64
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 13 - tessellation",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
