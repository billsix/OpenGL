"""McNopper Example 16: Phong-lit Wavefront OBJ (suzanne / monkey).

Original: McNopper/OpenGL Example16. Loads monkey.obj, renders with the same
Phong shaders as Example 05, MSAA 8x. Port animates rotation around Y.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("monkey.obj"))
    prog: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
    )
    GL.glUseProgram(prog)

    # Expand to per-vertex attribute arrays via the index buffer (glDrawArrays in C).
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    light_dir: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    view_ld: np.ndarray = view[:3, :3] @ light_dir

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
        1.0,
        0.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        0.0,
        1.0,
        0.0,
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

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        count=len(pos),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
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
    mv: np.ndarray = state["view"] @ common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
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
    imgui.text(f"vertices: {state.get('count', 0)}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 16 - phong OBJ",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        samples=8,
    )
