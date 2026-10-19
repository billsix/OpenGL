"""McNopper Example 12: Shadow mapping with GL_TEXTURE_COMPARE_MODE.

Original: McNopper/OpenGL Example12. First pass renders torus depth into a 1024x1024
depth texture from the light's POV; second pass samples that depth
texture as a sampler2DShadow to attenuate diffuse shading on a torus
and a backing plane.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

SHADOW_SIZE: int = 1024
CAMERA_POS: list[float] = [-5.0, 5.0, 10.0]
LIGHT_POS: list[float] = [0.0, 0.0, 10.0]
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    torus: common.Shape = common.make_torus(
        inner_radius=0.5, outer_radius=1.0, slices=32, stacks=32
    )
    plane: common.Shape = common.make_plane(10.0)

    prog_shadow: int = common.compile_program_files(
        script_file=__file__,
        vert="rendershadow.vert.glsl",
        frag="rendershadow.frag.glsl",
    )
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="useshadow.vert.glsl",
        frag="useshadow.frag.glsl",
    )

    shadow_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, shadow_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_DEPTH_COMPONENT24,
        SHADOW_SIZE,
        SHADOW_SIZE,
        0,
        GL.GL_DEPTH_COMPONENT,
        GL.GL_UNSIGNED_INT,
        None,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_COMPARE_MODE,
        GL.GL_COMPARE_REF_TO_TEXTURE,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LESS)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_TEXTURE_2D,
        shadow_tex,
        0,
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # Buffers
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

    GL.glUseProgram(prog_shadow)
    vao_st: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_st)
    common.attach_attrib(
        prog=prog_shadow, vao=vao_st, vbo=vbo_v, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    vao_sp: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_sp)
    common.attach_attrib(
        prog=prog_shadow, vao=vao_sp, vbo=vbo_pv, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_pi)

    GL.glUseProgram(prog)
    GL.glUniform1i(common.uniform(prog=prog, name="u_shadowTexture"), 0)
    ld: np.ndarray = np.array(LIGHT_POS, dtype=np.float32)
    ld = ld / np.linalg.norm(ld)
    view: np.ndarray = common.look_at(
        eye=CAMERA_POS, center=[0, 0, 0], up=[0, 1, 0]
    )
    view_ld: np.ndarray = view[:3, :3] @ ld
    GL.glUniform3f(common.uniform(prog=prog, name="u_lightDirection"), *view_ld)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glPolygonOffset(1.0, 0.0)

    state.update(
        prog=prog,
        prog_shadow=prog_shadow,
        fbo=fbo,
        shadow_tex=shadow_tex,
        vao_t=vao_t,
        vao_p=vao_p,
        vao_st=vao_st,
        vao_sp=vao_sp,
        n_torus=len(torus.indices),
        n_plane=len(plane.indices),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        view_loc=common.uniform(prog=prog, name="u_viewMatrix"),
        model_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        sm_loc=common.uniform(prog=prog, name="u_shadowMatrix"),
        color_loc=common.uniform(prog=prog, name="u_shapeColor"),
        proj_loc_s=common.uniform(prog=prog_shadow, name="u_projectionMatrix"),
        mv_loc_s=common.uniform(prog=prog_shadow, name="u_modelViewMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    bias: np.ndarray = np.array(
        [
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    proj_shadow: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=1.0, near=1.0, far=100.0
    )
    state["shadow_bias_proj"] = bias @ proj_shadow
    GL.glUseProgram(state["prog_shadow"])
    common.upload_matrix(loc=state["proj_loc_s"], m=proj_shadow)

    GL.glUseProgram(state["prog"])
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 20.0 * dt) % 360.0

    light_view: np.ndarray = common.look_at(
        eye=LIGHT_POS, center=[0, 0, 0], up=[0, 1, 0]
    )
    shadow_matrix: np.ndarray = state["shadow_bias_proj"] @ light_view

    # depth pass: the shadow map is this FBO's depth attachment, so it must
    # not stay bound as a texture while it is rendered into (C main.c)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glViewport(0, 0, SHADOW_SIZE, SHADOW_SIZE)
    GL.glColorMask(False, False, False, False)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog_shadow"])
    model_torus: np.ndarray = common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["mv_loc_s"], m=light_view @ model_torus)
    GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glFrontFace(GL.GL_CW)
    GL.glBindVertexArray(state["vao_st"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_torus"], GL.GL_UNSIGNED_INT, None
    )
    GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glFrontFace(GL.GL_CCW)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glColorMask(True, True, True, True)
    GL.glViewport(0, 0, state.get("w", 1), state.get("h", 1))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["shadow_tex"])

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["view_loc"], m=state["view"])
    common.upload_matrix(loc=state["sm_loc"], m=shadow_matrix)

    # plane
    plane_model: np.ndarray = common.translate(x=0, y=0, z=-5)
    common.upload_matrix(loc=state["model_loc"], m=plane_model)
    common.upload_matrix3(
        loc=state["nm_loc"], m=(state["view"] @ plane_model)[:3, :3]
    )
    GL.glUniform4f(state["color_loc"], 0.0, 0.5, 0.0, 1.0)
    GL.glBindVertexArray(state["vao_p"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_plane"], GL.GL_UNSIGNED_INT, None
    )

    # torus
    common.upload_matrix(loc=state["model_loc"], m=model_torus)
    common.upload_matrix3(
        loc=state["nm_loc"], m=(state["view"] @ model_torus)[:3, :3]
    )
    GL.glUniform4f(state["color_loc"], 0.33, 0.0, 0.5, 1.0)
    GL.glBindVertexArray(state["vao_t"])
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
    imgui.text(f"shadow map: {SHADOW_SIZE}x{SHADOW_SIZE}, sampler2DShadow")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 12 - shadow mapping",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
