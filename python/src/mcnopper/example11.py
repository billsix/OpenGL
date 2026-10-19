"""McNopper Example 11: Glass torus inside a cubemap-skybox sphere.

Original: McNopper/OpenGL Example11. The background is an inside-out sphere
sampling the cubemap directly; the torus uses a refraction shader that
also samples the cubemap. Camera orbits the origin.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

CIRCLE_RADIUS: float = 5.0
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    torus: common.Shape = common.make_torus(
        inner_radius=0.25, outer_radius=1.0, slices=32, stacks=32
    )
    bg_sphere: common.Shape = common.make_sphere(
        radius=CIRCLE_RADIUS, slices=32
    )

    prog: int = common.compile_program_files(
        script_file=__file__, vert="glass.vert.glsl", frag="glass.frag.glsl"
    )
    prog_bg: int = common.compile_program_files(
        script_file=__file__,
        vert="background.vert.glsl",
        frag="background.frag.glsl",
    )

    cubemap: int = common.upload_cubemap_tgas(
        [
            common.find_media("cm_pos_x.tga"),
            common.find_media("cm_neg_x.tga"),
            common.find_media("cm_pos_y.tga"),
            common.find_media("cm_neg_y.tga"),
            common.find_media("cm_pos_z.tga"),
            common.find_media("cm_neg_z.tga"),
        ]
    )

    GL.glUseProgram(prog)
    GL.glUniform1i(common.uniform(prog=prog, name="u_cubemap"), 0)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(torus.vertices)
    vbo_n: int = common.make_vbo(torus.normals)
    vbo_i: int = common.make_vbo(
        data=torus.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    GL.glUseProgram(prog_bg)
    GL.glUniform1i(common.uniform(prog=prog_bg, name="u_cubemap"), 0)
    vao_bg: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_bg)
    vbo_bv: int = common.make_vbo(bg_sphere.vertices)
    vbo_bi: int = common.make_vbo(
        data=bg_sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_bg, vao=vao_bg, vbo=vbo_bv, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_bi)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        prog_bg=prog_bg,
        vao=vao,
        vao_bg=vao_bg,
        n_torus=len(torus.indices),
        n_bg=len(bg_sphere.indices),
        cubemap=cubemap,
        vp_loc=common.uniform(prog=prog, name="u_viewProjectionMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        n_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        cam_loc=common.uniform(prog=prog, name="u_camera"),
        vp_loc_bg=common.uniform(prog=prog_bg, name="u_viewProjectionMatrix"),
        m_loc_bg=common.uniform(prog=prog_bg, name="u_modelMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    state["proj"] = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0
    a: float = math.radians(ui["angle"])
    cam: np.ndarray = np.array(
        [CIRCLE_RADIUS * -math.sin(a), 0, CIRCLE_RADIUS * math.cos(a), 1.0],
        dtype=np.float32,
    )
    view: np.ndarray = common.look_at(
        eye=[cam[0], 0, cam[2]], center=[0, 0, 0], up=[0, 1, 0]
    )
    vp: np.ndarray = state["proj"] @ view
    model: np.ndarray = common.translate(x=0, y=-0.5, z=0) @ common.rotate_x(
        45.0
    )
    nm = model[:3, :3]

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["cubemap"])

    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["vp_loc_bg"], m=vp)
    common.upload_matrix(loc=state["m_loc_bg"], m=model)
    GL.glFrontFace(GL.GL_CW)
    GL.glBindVertexArray(state["vao_bg"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_bg"], GL.GL_UNSIGNED_INT, None)

    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["vp_loc"], m=vp)
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["n_loc"], m=nm)
    GL.glUniform4f(state["cam_loc"], *cam)
    GL.glFrontFace(GL.GL_CCW)
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_torus"], GL.GL_UNSIGNED_INT, None
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
        title="McNopper Example 11 - glass torus + skybox",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
