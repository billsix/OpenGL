"""McNopper Example 10: Geometry shader that duplicates a sphere.

Original: McNopper/OpenGL Example10. The geometry stage emits each input triangle
twice (translated apart) so a single VBO of one sphere produces two
shaded spheres. View-space directional light uses the same matrix
upload pattern as Example 05.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def init() -> bool:
    sphere: common.Shape = common.make_sphere(radius=1.0, slices=32)
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="dublicate.vert.glsl",
        geom="dublicate.geom.glsl",
        frag="dublicate.frag.glsl",
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(sphere.vertices)
    vbo_n: int = common.make_vbo(sphere.normals)
    vbo_i: int = common.make_vbo(
        data=sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 6], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )
    common.upload_matrix3(
        loc=common.uniform(prog=prog, name="u_normalMatrix"), m=view[:3, :3]
    )

    light_world: np.ndarray = np.array([0.0, 0.5, 2.0, 1.0], dtype=np.float32)
    light_view: np.ndarray = view @ light_world
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_lightPosition"), *light_view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(sphere.indices),
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
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
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("Geometry shader emits each triangle twice -> two spheres.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 10 - geometry shader",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
