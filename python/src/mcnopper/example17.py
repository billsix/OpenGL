"""McNopper Example 17: gl_ClipDistance plane on a teapot.

Original: McNopper/OpenGL Example17. Loads teapot.obj, computes a clip plane just
below the cap, and discards fragments above it via gl_ClipDistance[0].
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True, "plane_h": 2.5}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("teapot.obj"))
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="phong_clip.vert.glsl",
        frag="phong_clip.frag.glsl",
    )
    GL.glUseProgram(prog)

    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)

    view: np.ndarray = common.look_at(
        eye=[0, 10, 10], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    view_ld: np.ndarray = view[:3, :3] @ light
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_light.direction"), *view_ld
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.ambientColor"),
        0.3,
        0.3,
        0.3,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.diffuseColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.specularColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.ambientColor"),
        0.0,
        0.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        0.0,
        0.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.specularColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_material.specularExponent"), 20.0
    )

    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CLIP_DISTANCE0)

    state.update(
        prog=prog,
        vao=vao,
        count=len(pos),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        plane_loc=common.uniform(prog=prog, name="u_plane"),
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
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])
    model: np.ndarray = common.rotate_y(ui["angle"])
    mv: np.ndarray = state["view"] @ model
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    # Plane: ax + by + cz + d = 0 with normal (0,-1,0) through (0, plane_h, 0).
    GL.glUniform4f(state["plane_loc"], 0.0, -1.0, 0.0, ui["plane_h"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])
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
    _, ui["plane_h"] = imgui.slider_float(
        label="clip plane height", v=ui["plane_h"], v_min=-2.0, v_max=5.0
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 17 - clip plane",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        samples=8,
    )
