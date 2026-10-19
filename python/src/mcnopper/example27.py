"""McNopper Example 27: Planar shadow projection (matrix trick).

Original: McNopper/OpenGL Example27. Render the torus normally, then re-render
its silhouette projected onto the floor plane via a planar-shadow
matrix derived from the light direction. No stencil involved.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

CAMERA_POS: list[float] = [-5.0, 5.0, 10.0]
LIGHT_DIR: list[float] = [0.0, 0.0, 10.0]
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    torus: common.Shape = common.make_torus(
        inner_radius=0.5, outer_radius=1.0, slices=32, stacks=32
    )
    plane: common.Shape = common.make_plane(10.0)

    prog: int = common.compile_program_files(
        script_file=__file__, vert="color.vert.glsl", frag="color.frag.glsl"
    )
    prog_sh: int = common.compile_program_files(
        script_file=__file__, vert="shadow.vert.glsl", frag="shadow.frag.glsl"
    )

    vbo_v: int = common.make_vbo(torus.vertices)
    vbo_n: int = common.make_vbo(torus.normals)
    vbo_i: int = common.make_vbo(
        data=torus.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    vbo_pv: int = common.make_vbo(plane.vertices)
    vbo_pn: int = common.make_vbo(plane.normals)
    vbo_pi: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )

    GL.glUseProgram(prog)
    vao_t: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_t)
    common.attach_attrib(
        prog=prog, vao=vao_t, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_t, vbo=vbo_n, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    vao_p: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_p)
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_pi)

    GL.glUseProgram(prog_sh)
    vao_sh: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_sh)
    common.attach_attrib(
        prog=prog_sh, vao=vao_sh, vbo=vbo_v, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    light: np.ndarray = np.array(LIGHT_DIR, dtype=np.float32)
    light /= np.linalg.norm(light)
    view: np.ndarray = common.look_at(
        eye=CAMERA_POS, center=[0, 0, 0], up=[0, 1, 0]
    )
    light_view: np.ndarray = view[:3, :3] @ light
    GL.glUseProgram(prog)
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_lightDirection"), *light_view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        prog_sh=prog_sh,
        vao_t=vao_t,
        vao_p=vao_p,
        vao_sh=vao_sh,
        n_t=len(torus.indices),
        n_p=len(plane.indices),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        view_loc=common.uniform(prog=prog, name="u_viewMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        color_loc=common.uniform(prog=prog, name="u_shapeColor"),
        proj_loc_sh=common.uniform(prog=prog_sh, name="u_projectionMatrix"),
        view_loc_sh=common.uniform(prog=prog_sh, name="u_viewMatrix"),
        sp_loc=common.uniform(prog=prog_sh, name="u_shadowProjectionMatrix"),
        m_loc_sh=common.uniform(prog=prog_sh, name="u_modelMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    GL.glUseProgram(state["prog_sh"])
    common.upload_matrix(loc=state["proj_loc_sh"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 20.0 * dt) % 360.0

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["view_loc"], m=state["view"])

    plane_m: np.ndarray = common.translate(x=0, y=0, z=-5)
    plane_mv: np.ndarray = state["view"] @ plane_m
    common.upload_matrix(loc=state["m_loc"], m=plane_m)
    common.upload_matrix3(loc=state["nm_loc"], m=plane_mv[:3, :3])
    GL.glUniform4f(state["color_loc"], 0.0, 0.5, 0.0, 1.0)
    GL.glBindVertexArray(state["vao_p"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_p"], GL.GL_UNSIGNED_INT, None)

    # Shadow projection: plane is z = -5 in world space, so (0,0,1,5) plane equation.
    shadow_plane = (0.0, 0.0, 1.0, 5.0)
    sp: np.ndarray = common.planar_shadow_directional(
        plane=shadow_plane, light_dir=LIGHT_DIR
    )

    GL.glUseProgram(state["prog_sh"])
    common.upload_matrix(loc=state["view_loc_sh"], m=state["view"])
    common.upload_matrix(loc=state["sp_loc"], m=sp)
    torus_m: np.ndarray = common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["m_loc_sh"], m=torus_m)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glBindVertexArray(state["vao_sh"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_t"], GL.GL_UNSIGNED_INT, None)
    GL.glEnable(GL.GL_DEPTH_TEST)

    GL.glUseProgram(state["prog"])
    torus_mv: np.ndarray = state["view"] @ torus_m
    common.upload_matrix(loc=state["m_loc"], m=torus_m)
    common.upload_matrix3(loc=state["nm_loc"], m=torus_mv[:3, :3])
    GL.glUniform4f(state["color_loc"], 0.33, 0.0, 0.5, 1.0)
    GL.glBindVertexArray(state["vao_t"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_t"], GL.GL_UNSIGNED_INT, None)
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
        title="McNopper Example 27 - planar shadow",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
