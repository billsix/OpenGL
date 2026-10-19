"""McNopper Example 04: Lit cube with directional light (Lambert).

Original: McNopper/OpenGL Example04 - first depth-tested 3D demo, draws a cube
shaded by a single directional light in world space. Port adds an
imgui panel for the rotation angles and light direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {
    "rot_y": 45.0,
    "rot_x": 45.0,
    "auto_rotate": False,  # the C shows a static Rx(45)*Ry(45) pose
    "color": [1.0, 0.0, 0.0, 1.0],
    "light_dir": [1.0, 1.0, 1.0],
}


def init() -> bool:
    cube: common.Shape = common.make_cube(0.5)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="basic.vert.glsl", frag="color.frag.glsl"
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

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(cube.indices),
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
        normal_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog, name="u_lightDirection"),
        color_loc=common.uniform(prog=prog, name="u_color"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    state["aspect"] = w / max(h, 1)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["rot_y"] = (ui["rot_y"] + 30.0 * dt) % 360.0

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])

    model: np.ndarray = common.rotate_x(ui["rot_x"]) @ common.rotate_y(
        ui["rot_y"]
    )
    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=state.get("aspect", 1.0), near=1.0, far=100.0
    )
    mvp: np.ndarray = proj @ view @ model
    common.upload_matrix(loc=state["mvp_loc"], m=mvp)
    common.upload_matrix3(loc=state["normal_loc"], m=model[:3, :3])

    ld: np.ndarray = np.asarray(ui["light_dir"], dtype=np.float32)
    n: np.floating[Any] = np.linalg.norm(ld)
    if n > 1e-8:
        ld = ld / n
    GL.glUniform3f(state["light_loc"], *ld)
    GL.glUniform4f(state["color_loc"], *ui["color"])

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
    _, ui["rot_y"] = imgui.slider_float(
        label="rot Y", v=ui["rot_y"], v_min=0, v_max=360
    )
    _, ui["rot_x"] = imgui.slider_float(
        label="rot X", v=ui["rot_x"], v_min=-180, v_max=180
    )
    _, c = imgui.color_edit3(label="color", col=ui["color"][:3])
    ui["color"][:3] = list(c)
    _, ld = imgui.input_float3(label="light dir", v=ui["light_dir"])
    ui["light_dir"] = list(ld)


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 04 - lit cube",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
