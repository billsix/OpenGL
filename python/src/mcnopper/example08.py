"""McNopper Example 08: Cubemap reflection on a cube.

Original: McNopper/OpenGL Example08 - 6 cm_*.tga TGAs uploaded as a cubemap, sampled
in the fragment shader through the world-space reflection of the
view-to-fragment vector. Cube is animated around Y.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    cube: common.Shape = common.make_cube(0.5)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="cubemap.vert.glsl", frag="cubemap.frag.glsl"
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(cube.vertices)
    vbo_n: int = common.make_vbo(cube.normals)
    vbo_i: int = common.make_vbo(
        data=cube.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    cube_tex: int = common.upload_cubemap_tgas(
        [
            common.find_media("cm_pos_x.tga"),
            common.find_media("cm_neg_x.tga"),
            common.find_media("cm_pos_y.tga"),
            common.find_media("cm_neg_y.tga"),
            common.find_media("cm_pos_z.tga"),
            common.find_media("cm_neg_z.tga"),
        ]
    )
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, cube_tex)
    GL.glUniform1i(common.uniform(prog=prog, name="u_cubemapTexture"), 0)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(cube.indices),
        view=view,
        cubemap=cube_tex,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        n_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        inv_view_loc=common.uniform(prog=prog, name="u_inverseViewMatrix"),
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
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 20.0 * dt) % 360.0
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])

    model: np.ndarray = common.rotate_x(15.0) @ common.rotate_y(ui["angle"])
    mv: np.ndarray = state["view"] @ model
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["n_loc"], m=mv[:3, :3])
    # inverse view (rotation only, orthogonal -> transpose)
    inv_view_3: np.ndarray = state["view"][:3, :3].T
    common.upload_matrix3(loc=state["inv_view_loc"], m=inv_view_3)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["cubemap"])
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


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 08 - cubemap reflection",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
