"""McNopper Example 18: Stencil-buffer cap on a clipped sphere.

Original: McNopper/OpenGL Example18. The vertex shader writes gl_ClipDistance[0]
against a rotating plane equation; back faces of the sphere then
"cap" the clipped opening using stencil INVERT trickery, and a textured
plane fills exactly the visible cap region (StencilFunc EQUAL 1,1).
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    sphere: common.Shape = common.make_sphere(radius=0.5, slices=64)
    plane: common.Shape = common.make_plane(0.5)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
    )
    GL.glUseProgram(prog)

    vao_s: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_s)
    vbo_sv: int = common.make_vbo(sphere.vertices)
    vbo_sn: int = common.make_vbo(sphere.normals)
    vbo_si: int = common.make_vbo(
        data=sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog, vao=vao_s, vbo=vbo_sv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_s, vbo=vbo_sn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_si)

    vao_p: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_p)
    vbo_pv: int = common.make_vbo(plane.vertices)
    vbo_pn: int = common.make_vbo(plane.normals)
    vbo_pi: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_pi)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 3], center=[0, 0, 0], up=[0, 1, 0]
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
        0.1,
        0.1,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        0.0,
        0.8,
        0.8,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.specularColor"),
        0.6,
        0.6,
        0.6,
        1.0,
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_material.specularExponent"), 60.0
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_STENCIL_TEST)

    state.update(
        prog=prog,
        vao_s=vao_s,
        vao_p=vao_p,
        n_sphere=len(sphere.indices),
        n_plane=len(plane.indices),
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
        ui["angle"] = (ui["angle"] + 45.0 * dt) % 360.0

    GL.glClear(
        GL.GL_COLOR_BUFFER_BIT  # ty: ignore[unsupported-operator]
        | GL.GL_DEPTH_BUFFER_BIT
        | GL.GL_STENCIL_BUFFER_BIT
    )
    GL.glUseProgram(state["prog"])

    # Plane equation in world space, derived by rotating (0,0,-1,0).
    plane_m: np.ndarray = (
        common.rotate_y(ui["angle"])
        @ common.translate(x=0, y=0.25, z=0.25)
        @ common.rotate_x(-30.0)
    )
    plane_inv_t: np.ndarray = np.linalg.inv(plane_m).T.astype(np.float32)
    clip_plane: np.ndarray = plane_inv_t @ np.array(
        [0, 0, -1, 0], dtype=np.float32
    )
    GL.glUniform4f(state["plane_loc"], *clip_plane)

    # Sphere with stencil INVERT
    model: np.ndarray = common.rotate_y(ui["angle"])
    mv: np.ndarray = state["view"] @ model
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glStencilFunc(GL.GL_ALWAYS, 0, 0)
    GL.glStencilOp(GL.GL_KEEP, GL.GL_INVERT, GL.GL_INVERT)
    GL.glEnable(GL.GL_CLIP_DISTANCE0)
    GL.glBindVertexArray(state["vao_s"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_sphere"], GL.GL_UNSIGNED_INT, None
    )

    # Plane fills the cap where stencil & 1 == 1
    cap_model: np.ndarray = (
        model @ common.translate(x=0, y=0.25, z=0.25) @ common.rotate_x(-30.0)
    )
    cap_mv: np.ndarray = state["view"] @ cap_model
    common.upload_matrix(loc=state["m_loc"], m=cap_model)
    common.upload_matrix3(loc=state["nm_loc"], m=cap_mv[:3, :3])
    GL.glStencilFunc(GL.GL_EQUAL, 1, 1)
    GL.glDisable(GL.GL_CLIP_DISTANCE0)
    # AMD-bug workaround from C source: pin the plane far away so unenabled
    # gl_ClipDistance writes can't accidentally clip the cap.
    GL.glUniform4f(state["plane_loc"], 0, -1, 0, 100.0)
    GL.glBindVertexArray(state["vao_p"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_plane"], GL.GL_UNSIGNED_INT, None
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
        title="McNopper Example 18 - stencil cap",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        stencil_bits=8,
        samples=16,
    )
