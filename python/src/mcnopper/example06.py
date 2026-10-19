"""McNopper Example 06: Textured cube spinning around Y.

Original: McNopper/OpenGL Example06 - crate.tga textured cube, mipmapped, single
directional light. Port reuses the GLUS shaders verbatim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {
    "angle": 0.0,
    "tilt": 45.0,
    "auto_rotate": True,
    "light_dir": [1.0, 1.0, 1.0],
}


def init() -> bool:
    cube: common.Shape = common.make_cube(0.5)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="basic.vert.glsl", frag="texture.frag.glsl"
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(cube.vertices)
    vbo_n: int = common.make_vbo(cube.normals)
    vbo_t: int = common.make_vbo(cube.texcoords)
    vbo_i: int = common.make_vbo(
        data=cube.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    tex: int = common.upload_tga_2d(common.find_media("crate.tga"))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(cube.indices),
        tex=tex,
        vp_loc=common.uniform(prog=prog, name="u_viewProjectionMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        n_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog, name="u_lightDirection"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["vp_loc"], m=proj @ view)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 90.0 * dt) % 360.0
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])
    model: np.ndarray = common.rotate_x(ui["tilt"]) @ common.rotate_y(
        ui["angle"]
    )
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["n_loc"], m=model[:3, :3])
    ld: np.ndarray = np.asarray(ui["light_dir"], dtype=np.float32)
    n: np.floating[Any] = np.linalg.norm(ld)
    if n > 1e-8:
        ld = ld / n
    GL.glUniform3f(state["light_loc"], *ld)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["tex"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["auto_rotate"] = imgui.checkbox(
        label="auto rotate", v=ui["auto_rotate"]
    )
    _, ui["angle"] = imgui.slider_float(
        label="angle", v=ui["angle"], v_min=0, v_max=360
    )
    _, ui["tilt"] = imgui.slider_float(
        label="tilt X", v=ui["tilt"], v_min=-180, v_max=180
    )
    _, ld = imgui.input_float3(label="light dir", v=ui["light_dir"])
    ui["light_dir"] = list(ld)


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 06 - textured cube",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
