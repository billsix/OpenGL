"""McNopper Example 28: Screen-space ambient occlusion (SSAO).

Original: McNopper/OpenGL Example28 (Chapman SSAO tutorial). Three teapots and
a wood floor are rendered into an MRT (color + view-space normal +
depth); the SSAO pass samples a hemispheric kernel rotated by a 4x4
random-noise texture and writes occlusion into a buffer; a final blur
pass combines color * (1-ssao) into the framebuffer.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

KERNEL_SIZE: int = 32
SSAO_RADIUS: float = 1.0
ROTATION_NOISE_SIDE: int = 4
TEXTURE_WIDTH: int = 640  # the C's G-buffer is the window size
TEXTURE_HEIGHT: int = 480
state: dict = {}
ui = {"no_ssao": False}


def _make_color_attachment(
    w: int, h: int, fmt=GL.GL_RGB8, internal_fmt=None
) -> int:
    if internal_fmt is None:
        internal_fmt = fmt
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    if internal_fmt == GL.GL_DEPTH_COMPONENT32F:
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            internal_fmt,
            w,
            h,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
    else:
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            internal_fmt,
            w,
            h,
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
    return tex


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("teapot.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    plane: common.Shape = common.make_plane(20.0)
    pp_plane: common.Shape = common.make_plane(1.0)

    prog: int = common.compile_program_files(
        script_file=__file__, vert="texture.vert.glsl", frag="texture.frag.glsl"
    )
    prog_phong: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
    )
    prog_ssao: int = common.compile_program_files(
        script_file=__file__, vert="ssao.vert.glsl", frag="ssao.frag.glsl"
    )
    prog_blur: int = common.compile_program_files(
        script_file=__file__, vert="blur.vert.glsl", frag="blur.frag.glsl"
    )

    floor_tex: int = common.upload_tga_2d(common.find_media("wood_texture.tga"))

    # SSAO kernel
    rng = np.random.default_rng(0)
    kernel: np.ndarray = np.zeros((KERNEL_SIZE, 3), dtype=np.float32)
    for i in range(KERNEL_SIZE):
        v: np.ndarray = np.array(
            [rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1)],
            dtype=np.float32,
        )
        v /= max(np.linalg.norm(v), 1e-8)
        scale = (i / KERNEL_SIZE) ** 2
        scale = max(0.1, min(1.0, scale))
        kernel[i] = v * scale

    rot_noise: np.ndarray = np.zeros(
        (ROTATION_NOISE_SIDE, ROTATION_NOISE_SIDE, 3), dtype=np.float32
    )
    for r in range(ROTATION_NOISE_SIDE):
        for c in range(ROTATION_NOISE_SIDE):
            v = np.array(
                [rng.uniform(-1, 1), rng.uniform(-1, 1), 0], dtype=np.float32
            )
            v /= max(np.linalg.norm(v), 1e-8)
            rot_noise[r, c] = v

    rot_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, rot_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB32F,
        ROTATION_NOISE_SIDE,
        ROTATION_NOISE_SIDE,
        0,
        GL.GL_RGB,
        GL.GL_FLOAT,
        rot_noise,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

    color_tex = _make_color_attachment(
        w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT, fmt=GL.GL_RGB8
    )
    normal_tex = _make_color_attachment(
        w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT, fmt=GL.GL_RGB8
    )
    depth_tex = _make_color_attachment(
        w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT, internal_fmt=GL.GL_DEPTH_COMPONENT32F
    )
    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        color_tex,
        0,
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT1,
        GL.GL_TEXTURE_2D,
        normal_tex,
        0,
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_TEXTURE_2D,
        depth_tex,
        0,
    )

    blur_tex = _make_color_attachment(
        w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT, fmt=GL.GL_RGB8
    )
    fbo_blur: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo_blur)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        blur_tex,
        0,
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # Floor VAO
    GL.glUseProgram(prog)
    vao_floor: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_floor)
    vbo_pv: int = common.make_vbo(plane.vertices)
    vbo_pn: int = common.make_vbo(plane.normals)
    vbo_pt: int = common.make_vbo(plane.texcoords)
    vbo_pi: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog, vao=vao_floor, vbo=vbo_pv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_floor, vbo=vbo_pn, name="a_normal", size=3
    )
    common.attach_attrib(
        prog=prog, vao=vao_floor, vbo=vbo_pt, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_pi)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)
    GL.glUniform1f(common.uniform(prog=prog, name="u_repeat"), 6.0)

    # Monkey VAO
    GL.glUseProgram(prog_phong)
    vao_obj: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_obj)
    vbo_v: int = common.make_vbo(pos)
    vbo_n = common.make_vbo(nor)
    common.attach_attrib(
        prog=prog_phong, vao=vao_obj, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_phong, vao=vao_obj, vbo=vbo_n, name="a_normal", size=3
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_light.ambientColor"),
        0.3,
        0.3,
        0.3,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_light.diffuseColor"), 1, 1, 1, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_light.specularColor"),
        1,
        1,
        1,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_material.ambientColor"),
        0,
        1,
        1,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_material.diffuseColor"),
        0,
        1,
        1,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_material.specularColor"),
        1,
        1,
        1,
        1,
    )
    GL.glUniform1f(
        common.uniform(prog=prog_phong, name="u_material.specularExponent"),
        20.0,
    )

    # SSAO VAO + uniforms
    GL.glUseProgram(prog_ssao)
    vao_ssao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_ssao)
    vbo_ppv: int = common.make_vbo(pp_plane.vertices)
    vbo_ppt: int = common.make_vbo(pp_plane.texcoords)
    vbo_ppi: int = common.make_vbo(
        data=pp_plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_ssao, vao=vao_ssao, vbo=vbo_ppv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_ssao, vao=vao_ssao, vbo=vbo_ppt, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_ppi)
    GL.glUniform1i(common.uniform(prog=prog_ssao, name="u_texture"), 0)
    GL.glUniform1i(common.uniform(prog=prog_ssao, name="u_normalTexture"), 1)
    GL.glUniform1i(common.uniform(prog=prog_ssao, name="u_depthTexture"), 2)
    GL.glUniform1i(
        common.uniform(prog=prog_ssao, name="u_rotationNoiseTexture"), 3
    )
    GL.glUniform1f(common.uniform(prog=prog_ssao, name="u_radius"), SSAO_RADIUS)
    GL.glUniform3fv(
        common.uniform(prog=prog_ssao, name="u_kernel"), KERNEL_SIZE, kernel
    )
    GL.glUniform2f(
        common.uniform(prog=prog_ssao, name="u_rotationNoiseScale"),
        TEXTURE_WIDTH / ROTATION_NOISE_SIDE,
        TEXTURE_HEIGHT / ROTATION_NOISE_SIDE,
    )

    # Blur VAO + uniforms
    GL.glUseProgram(prog_blur)
    vao_blur: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_blur)
    common.attach_attrib(
        prog=prog_blur, vao=vao_blur, vbo=vbo_ppv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_blur, vao=vao_blur, vbo=vbo_ppt, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_ppi)
    GL.glUniform1i(common.uniform(prog=prog_blur, name="u_colorTexture"), 0)
    GL.glUniform1i(common.uniform(prog=prog_blur, name="u_ssaoTexture"), 1)
    GL.glUniform2f(
        common.uniform(prog=prog_blur, name="u_texelStep"),
        1.0 / TEXTURE_WIDTH,
        1.0 / TEXTURE_HEIGHT,
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    state.update(
        prog=prog,
        prog_phong=prog_phong,
        prog_ssao=prog_ssao,
        prog_blur=prog_blur,
        vao_floor=vao_floor,
        vao_obj=vao_obj,
        vao_ssao=vao_ssao,
        vao_blur=vao_blur,
        n_floor=len(plane.indices),
        n_obj=len(pos),
        n_pp=len(pp_plane.indices),
        floor_tex=floor_tex,
        color_tex=color_tex,
        normal_tex=normal_tex,
        depth_tex=depth_tex,
        rot_tex=rot_tex,
        blur_tex=blur_tex,
        fbo=fbo,
        fbo_blur=fbo_blur,
        vp_loc=common.uniform(prog=prog, name="u_viewProjectionMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog, name="u_lightDirection"),
        proj_loc_phong=common.uniform(
            prog=prog_phong, name="u_projectionMatrix"
        ),
        mv_loc_phong=common.uniform(prog=prog_phong, name="u_modelViewMatrix"),
        nm_loc_phong=common.uniform(prog=prog_phong, name="u_normalMatrix"),
        light_loc_phong=common.uniform(
            prog=prog_phong, name="u_light.direction"
        ),
        ssao_proj_loc=common.uniform(prog=prog_ssao, name="u_projectionMatrix"),
        ssao_inv_proj_loc=common.uniform(
            prog=prog_ssao, name="u_inverseProjectionMatrix"
        ),
        no_ssao_loc=common.uniform(prog=prog_blur, name="u_noSSAO"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    w, h = state.get("w", 1), state.get("h", 1)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0,
        aspect=TEXTURE_WIDTH / TEXTURE_HEIGHT,
        near=1.0,
        far=100.0,
    )
    inv_proj: np.ndarray = np.linalg.inv(proj).astype(np.float32)
    view: np.ndarray = common.look_at(
        eye=[0, 15, 15], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light

    # G-buffer pass
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glDrawBuffers(
        n=2, bufs=[GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1]
    )
    GL.glViewport(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    # 3 teapots
    GL.glUseProgram(state["prog_phong"])
    common.upload_matrix(loc=state["proj_loc_phong"], m=proj)
    GL.glUniform3f(state["light_loc_phong"], *light_view)
    for i in range(3):
        model: np.ndarray = common.translate(
            x=(i - 1) * 4.1, y=0, z=0
        ) @ common.rotate_y(45.0)
        mv: np.ndarray = view @ model
        common.upload_matrix(loc=state["mv_loc_phong"], m=mv)
        common.upload_matrix3(loc=state["nm_loc_phong"], m=mv[:3, :3])
        GL.glBindVertexArray(state["vao_obj"])
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["n_obj"])

    # Floor
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["vp_loc"], m=proj @ view)
    GL.glUniform3f(state["light_loc"], *light_view)
    floor_m: np.ndarray = common.rotate_x(-90.0)
    common.upload_matrix(loc=state["m_loc"], m=floor_m)
    common.upload_matrix3(loc=state["nm_loc"], m=(view @ floor_m)[:3, :3])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["floor_tex"])
    GL.glBindVertexArray(state["vao_floor"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_floor"], GL.GL_UNSIGNED_INT, None
    )

    # SSAO pass
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo_blur"])
    GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["color_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["normal_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE2)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["depth_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE3)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["rot_tex"])
    GL.glUseProgram(state["prog_ssao"])
    common.upload_matrix(loc=state["ssao_proj_loc"], m=proj)
    common.upload_matrix(loc=state["ssao_inv_proj_loc"], m=inv_proj)
    GL.glBindVertexArray(state["vao_ssao"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_pp"], GL.GL_UNSIGNED_INT, None)

    # Blur to screen
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, w, h)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["color_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["blur_tex"])
    GL.glUseProgram(state["prog_blur"])
    GL.glUniform1f(state["no_ssao_loc"], 1.0 if ui["no_ssao"] else 0.0)
    GL.glBindVertexArray(state["vao_blur"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_pp"], GL.GL_UNSIGNED_INT, None)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["no_ssao"] = imgui.checkbox(label="disable SSAO", v=ui["no_ssao"])
    imgui.text(f"kernel: {KERNEL_SIZE} samples, radius {SSAO_RADIUS}")
    imgui.text(f"rotation noise: {ROTATION_NOISE_SIDE}x{ROTATION_NOISE_SIDE}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 28 - SSAO",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
