"""McNopper Example 19: Reflective floor via projective texturing.

Original: McNopper/OpenGL Example19. The monkey is rendered upside-down into
a window-sized offscreen FBO, then a textured floor projects that mirror
render through a bias matrix to fake a planar reflection. Surface texture
is multiplied with the projected mirror sample.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

TEXTURE_WIDTH: int = 640  # the C's mirror FBO is the window size
TEXTURE_HEIGHT: int = 480
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def _init_phong_obj() -> dict:
    mesh: common.ObjMesh = common.load_obj(common.find_media("monkey.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
    )
    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
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
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        0.0,
        1.0,
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
    return dict(
        prog=prog,
        vao=vao,
        count=len(pos),
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog, name="u_light.direction"),
    )


def init() -> bool:
    obj = _init_phong_obj()

    plane: common.Shape = common.make_plane(3.0)
    prog_floor: int = common.compile_program_files(
        script_file=__file__,
        vert="basic_proj.vert.glsl",
        frag="texture_multi_proj.frag.glsl",
    )
    GL.glUseProgram(prog_floor)
    vao_f: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_f)
    vbo_v: int = common.make_vbo(plane.vertices)
    vbo_n: int = common.make_vbo(plane.normals)
    vbo_t: int = common.make_vbo(plane.texcoords)
    vbo_i: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_floor, vao=vao_f, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_floor, vao=vao_f, vbo=vbo_n, name="a_normal", size=3
    )
    common.attach_attrib(
        prog=prog_floor, vao=vao_f, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    surface_tex: int = common.upload_tga_2d(common.find_media("ice.tga"))

    mirror_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, mirror_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        TEXTURE_WIDTH,
        TEXTURE_HEIGHT,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        None,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    depth_rb: int = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(
        GL.GL_RENDERBUFFER,
        GL.GL_DEPTH_COMPONENT24,
        TEXTURE_WIDTH,
        TEXTURE_HEIGHT,
    )

    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        mirror_tex,
        0,
    )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    GL.glUseProgram(prog_floor)
    GL.glUniform1i(common.uniform(prog=prog_floor, name="u_texture"), 0)
    GL.glUniform1i(common.uniform(prog=prog_floor, name="u_mirrorTexture"), 1)
    GL.glUniform1f(common.uniform(prog=prog_floor, name="u_repeat"), 6.0)

    view: np.ndarray = common.look_at(
        eye=[0, 1, 7], center=[0, 0, 0], up=[0, 1, 0]
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        obj=obj,
        prog_floor=prog_floor,
        vao_f=vao_f,
        n_floor=len(plane.indices),
        surface_tex=surface_tex,
        mirror_tex=mirror_tex,
        fbo=fbo,
        view=view,
        vp_loc=common.uniform(prog=prog_floor, name="u_viewProjectionMatrix"),
        vpb_loc=common.uniform(
            prog=prog_floor, name="u_viewProjectionBiasTextureMatrix"
        ),
        m_loc=common.uniform(prog=prog_floor, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog_floor, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog_floor, name="u_lightDirection"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def _draw_obj(
    scale_matrix: np.ndarray, proj: np.ndarray, view: np.ndarray
) -> None:
    """C ``updateWavefront``: the phong shader takes projection and
    modelview separately, so ``proj`` is the projection alone."""
    obj = state["obj"]
    GL.glUseProgram(obj["prog"])
    common.upload_matrix(loc=obj["proj_loc"], m=proj)
    model: np.ndarray = (
        scale_matrix
        @ common.translate(x=0, y=1.25, z=0)
        @ common.rotate_y(ui["angle"])
    )
    mv: np.ndarray = view @ model
    common.upload_matrix(loc=obj["mv_loc"], m=mv)
    common.upload_matrix3(loc=obj["nm_loc"], m=mv[:3, :3])
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view = (view @ scale_matrix)[:3, :3] @ light
    GL.glUniform3f(obj["light_loc"], *light_view)
    GL.glBindVertexArray(obj["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, obj["count"])


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0

    w = state.get("w", TEXTURE_WIDTH)
    h = state.get("h", TEXTURE_HEIGHT)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    proj_off: np.ndarray = common.perspective(
        fovy_deg=40.0,
        aspect=TEXTURE_WIDTH / TEXTURE_HEIGHT,
        near=1.0,
        far=100.0,
    )
    view: np.ndarray = state["view"]
    vp: np.ndarray = proj @ view
    vp_off: np.ndarray = proj_off @ view

    # Mirror pass
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glViewport(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT)
    GL.glClearColor(1, 1, 1, 0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glFrontFace(GL.GL_CW)
    scale: np.ndarray = common.scale_m(x=1, y=-1, z=1)
    _draw_obj(scale_matrix=scale, proj=proj_off, view=view)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # Bias matrix [-1,1] -> [0,1], applied to the view-projection the mirror
    # pass rendered with (C builds it right after reshape(TEXTURE_WIDTH,
    # TEXTURE_HEIGHT)); the floor is drawn with the window's.
    bias: np.ndarray = np.array(
        [
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    vp_bias: np.ndarray = bias @ vp_off

    # Scene pass
    GL.glViewport(0, 0, w, h)
    GL.glClearColor(0, 0, 0, 0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glFrontFace(GL.GL_CCW)
    _draw_obj(scale_matrix=np.eye(4, dtype=np.float32), proj=proj, view=view)

    GL.glUseProgram(state["prog_floor"])
    common.upload_matrix(loc=state["vp_loc"], m=vp)
    common.upload_matrix(loc=state["vpb_loc"], m=vp_bias)
    floor_model: np.ndarray = common.translate(x=0, y=0, z=3) @ common.rotate_x(
        -90
    )
    floor_mv: np.ndarray = view @ floor_model
    common.upload_matrix(loc=state["m_loc"], m=floor_model)
    common.upload_matrix3(loc=state["nm_loc"], m=floor_mv[:3, :3])
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    GL.glUniform3f(state["light_loc"], *(view[:3, :3] @ light))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["surface_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["mirror_tex"])
    GL.glBindVertexArray(state["vao_f"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_floor"], GL.GL_UNSIGNED_INT, None
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
    imgui.text(f"mirror RTT: {TEXTURE_WIDTH}x{TEXTURE_HEIGHT}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 19 - reflective floor",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
