"""McNopper Example 33: Full PBR with split-sum IBL.

Original: McNopper/OpenGL Example33. Generates a prefiltered specular cubemap
array (NUMBER_ROUGHNESS layers x 6 faces), a diffuse irradiance cubemap,
a BRDF integration LUT, and a sharp background cubemap - all on the
GPU at startup from doge2.hdr. The model BRDF then samples those for
realtime split-sum IBL.
"""

from __future__ import annotations

import numpy as np
from OpenGL.constant import Constant

from mcnopper import common
from mcnopper.common import GL

NUMBER_ROUGHNESS: int = 6
SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
MSAA: int = 4
BG_SIZE: int = 512
SPEC_SIZE: int = 512
DIFF_SIZE: int = 128
LUT_SIZE: int = 512
state: dict = {}
ui: dict = {
    "roughness": 0.1,
    "R0": 0.2,
    "color": [0.8, 0.8, 0.8],
    "exposure": 3.0,
    "gamma": 2.2,
}


def _make_cubemap(size: int, fmt: int | Constant = GL.GL_RGB32F) -> int:
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, tex)
    for face in range(6):
        GL.glTexImage2D(
            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,  # ty: ignore[unsupported-operator]
            0,
            fmt,
            size,
            size,
            0,
            GL.GL_RGB,
            GL.GL_FLOAT,
            None,
        )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_EDGE
    )
    return tex


def _make_cubemap_array(size: int, layers: int) -> int:
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY,
        0,
        GL.GL_RGB32F,
        size,
        size,
        layers,
        0,
        GL.GL_RGB,
        GL.GL_FLOAT,
        None,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    return tex


def init() -> bool:
    print("[Example 33] Generating prefiltered IBL textures (~5s)...")  # noqa: T201
    panorama: int = common.upload_hdr_2d(common.find_media("doge2.hdr"))
    GL.glBindTexture(GL.GL_TEXTURE_2D, panorama)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    # Programs
    prog_brdf: int = common.compile_program_files(
        script_file=__file__, vert="brdf.vert.glsl", frag="brdf.frag.glsl"
    )
    prog_bg: int = common.compile_program_files(
        script_file=__file__,
        vert="background.vert.glsl",
        frag="background.frag.glsl",
    )
    prog_full: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen.vert.glsl",
        frag="fullscreen.frag.glsl",
    )
    prog_spec_pre: int = common.compile_program_files(
        script_file=__file__,
        vert="panorama_to_cubemap.vert.glsl",
        frag="prefilter_specular.frag.glsl",
    )
    prog_diff_pre: int = common.compile_program_files(
        script_file=__file__,
        vert="panorama_to_cubemap.vert.glsl",
        frag="prefilter_diffuse.frag.glsl",
    )
    prog_lut: int = common.compile_program_files(
        script_file=__file__,
        vert="panorama_to_cubemap.vert.glsl",
        frag="brdf_lut.frag.glsl",
    )
    prog_p2c: int = common.compile_program_files(
        script_file=__file__,
        vert="panorama_to_cubemap.vert.glsl",
        frag="panorama_to_cubemap.frag.glsl",
    )

    # Output textures
    spec_tex: int = _make_cubemap_array(
        size=SPEC_SIZE, layers=6 * NUMBER_ROUGHNESS
    )
    diff_tex: int = _make_cubemap(DIFF_SIZE)
    lut_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, lut_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RG32F,
        LUT_SIZE,
        LUT_SIZE,
        0,
        GL.GL_RG,
        GL.GL_FLOAT,
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
    bg_tex: int = _make_cubemap(BG_SIZE)

    # Prefilter quad VAO
    quad_vbo: int = common.make_vbo(
        np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype=np.float32)
    )
    quad_vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(quad_vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, quad_vbo)
    GL.glVertexAttribPointer(
        index=0,
        size=2,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(0)

    pre_fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, pre_fbo)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, panorama)

    # Pass 1: specular cubemap-array
    GL.glUseProgram(prog_spec_pre)
    GL.glUniform1i(
        common.uniform(prog=prog_spec_pre, name="u_panoramaTexture"), 0
    )
    GL.glViewport(0, 0, SPEC_SIZE, SPEC_SIZE)
    for k in range(NUMBER_ROUGHNESS):
        r: float = k / max(NUMBER_ROUGHNESS - 1, 1)
        GL.glUniform1f(
            common.uniform(prog=prog_spec_pre, name="u_roughness"), r
        )
        for face in range(6):
            GL.glFramebufferTextureLayer(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                spec_tex,
                0,
                k * 6 + face,
            )
            GL.glUniform1i(
                common.uniform(prog=prog_spec_pre, name="u_face"), face
            )
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    # Pass 2: diffuse cubemap
    GL.glUseProgram(prog_diff_pre)
    GL.glUniform1i(
        common.uniform(prog=prog_diff_pre, name="u_panoramaTexture"), 0
    )
    GL.glViewport(0, 0, DIFF_SIZE, DIFF_SIZE)
    for face in range(6):
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,  # ty: ignore[unsupported-operator]
            diff_tex,
            0,
        )
        GL.glUniform1i(common.uniform(prog=prog_diff_pre, name="u_face"), face)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    # Pass 3: BRDF LUT
    GL.glUseProgram(prog_lut)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, lut_tex, 0
    )
    GL.glViewport(0, 0, LUT_SIZE, LUT_SIZE)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    # Pass 4: sharp background cubemap
    GL.glUseProgram(prog_p2c)
    GL.glUniform1i(common.uniform(prog=prog_p2c, name="u_panoramaTexture"), 0)
    GL.glViewport(0, 0, BG_SIZE, BG_SIZE)
    for face in range(6):
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,  # ty: ignore[unsupported-operator]
            bg_tex,
            0,
        )
        GL.glUniform1i(common.uniform(prog=prog_p2c, name="u_face"), face)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(1, [pre_fbo])
    GL.glDeleteVertexArrays(1, [quad_vao])
    GL.glDeleteTextures(size=1, array=[panorama])
    print("[Example 33] IBL prefilter done.")  # noqa: T201

    # Main scene
    full_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, full_tex)
    GL.glTexImage2DMultisample(
        GL.GL_TEXTURE_2D_MULTISAMPLE,
        MSAA,
        GL.GL_RGB32F,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        GL.GL_TRUE,
    )
    full_depth: int = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, full_depth)
    GL.glRenderbufferStorageMultisample(
        GL.GL_RENDERBUFFER,
        MSAA,
        GL.GL_DEPTH_COMPONENT24,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )
    full_fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, full_fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D_MULTISAMPLE,
        full_tex,
        0,
    )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_RENDERBUFFER,
        full_depth,
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    mesh: common.ObjMesh = common.load_obj(common.find_media("venusm.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    bg_sphere: common.Shape = common.make_sphere(radius=500.0, slices=32)

    GL.glUseProgram(prog_brdf)
    vao_m: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_m)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(
        prog=prog_brdf, vao=vao_m, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_brdf, vao=vao_m, vbo=vbo_n, name="a_normal", size=3
    )
    GL.glUniform4f(common.uniform(prog=prog_brdf, name="u_eye"), 0, 2.5, 6, 1)
    GL.glUniform1i(common.uniform(prog=prog_brdf, name="u_textureSpecular"), 0)
    GL.glUniform1i(common.uniform(prog=prog_brdf, name="u_textureDiffuse"), 1)
    GL.glUniform1i(common.uniform(prog=prog_brdf, name="u_textureLUT"), 2)
    GL.glUniform3f(
        common.uniform(prog=prog_brdf, name="u_colorMaterial"), 0.8, 0.8, 0.8
    )
    GL.glUniform1f(
        common.uniform(prog=prog_brdf, name="u_roughnessScale"),
        float(NUMBER_ROUGHNESS - 1),
    )

    GL.glUseProgram(prog_bg)
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
    GL.glUniform1i(common.uniform(prog=prog_bg, name="u_texture"), 3)

    GL.glUseProgram(prog_full)
    vao_full: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_full)
    GL.glUniform1i(
        common.uniform(prog=prog_full, name="u_framebufferTexture"), 0
    )
    GL.glUniform1i(common.uniform(prog=prog_full, name="u_msaaSamples"), MSAA)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog_brdf=prog_brdf,
        prog_bg=prog_bg,
        prog_full=prog_full,
        vao_m=vao_m,
        vao_bg=vao_bg,
        vao_full=vao_full,
        n_obj=len(pos),
        n_bg=len(bg_sphere.indices),
        spec_tex=spec_tex,
        diff_tex=diff_tex,
        lut_tex=lut_tex,
        bg_tex=bg_tex,
        full_tex=full_tex,
        full_fbo=full_fbo,
        vp_loc_m=common.uniform(prog=prog_brdf, name="u_viewProjectionMatrix"),
        m_loc_m=common.uniform(prog=prog_brdf, name="u_modelMatrix"),
        nm_loc_m=common.uniform(prog=prog_brdf, name="u_normalMatrix"),
        rough_loc=common.uniform(prog=prog_brdf, name="u_roughnessMaterial"),
        R0_loc=common.uniform(prog=prog_brdf, name="u_R0Material"),
        color_loc=common.uniform(prog=prog_brdf, name="u_colorMaterial"),
        vp_loc_bg=common.uniform(prog=prog_bg, name="u_viewProjectionMatrix"),
        exp_loc=common.uniform(prog=prog_full, name="u_exposure"),
        gam_loc=common.uniform(prog=prog_full, name="u_gamma"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    w, h = state.get("w", 1), state.get("h", 1)
    proj: np.ndarray = common.perspective(
        fovy_deg=60.0, aspect=SCREEN_WIDTH / SCREEN_HEIGHT, near=1.0, far=1000.0
    )
    view: np.ndarray = common.look_at(
        eye=[0, 2.5, 6], center=[0, 2, 0], up=[0, 1, 0]
    )
    vp: np.ndarray = proj @ view

    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, state["spec_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["diff_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE2)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["lut_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE3)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["bg_tex"])

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["full_fbo"])
    GL.glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    GL.glEnable(GL.GL_MULTISAMPLE)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    GL.glFrontFace(GL.GL_CW)
    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["vp_loc_bg"], m=vp)
    GL.glBindVertexArray(state["vao_bg"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_bg"], GL.GL_UNSIGNED_INT, None)
    GL.glFrontFace(GL.GL_CCW)

    GL.glUseProgram(state["prog_brdf"])
    GL.glUniform1f(state["rough_loc"], ui["roughness"])
    GL.glUniform1f(state["R0_loc"], ui["R0"])
    GL.glUniform3f(state["color_loc"], *ui["color"])
    model: np.ndarray = common.scale_m(x=0.001, y=0.001, z=0.001)
    common.upload_matrix(loc=state["vp_loc_m"], m=vp)
    common.upload_matrix(loc=state["m_loc_m"], m=model)
    common.upload_matrix3(loc=state["nm_loc_m"], m=model[:3, :3])
    GL.glBindVertexArray(state["vao_m"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["n_obj"])

    # Resolve to screen (C: depth test off around the fullscreen pass)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDisable(GL.GL_MULTISAMPLE)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glViewport(0, 0, w, h)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, state["full_tex"])
    GL.glUseProgram(state["prog_full"])
    GL.glUniform1f(state["exp_loc"], ui["exposure"])
    GL.glUniform1f(state["gam_loc"], ui["gamma"])
    GL.glBindVertexArray(state["vao_full"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    GL.glEnable(GL.GL_DEPTH_TEST)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["roughness"] = imgui.slider_float(
        label="roughness", v=ui["roughness"], v_min=0.0, v_max=1.0
    )
    _, ui["R0"] = imgui.slider_float(
        label="R0 (Schlick)", v=ui["R0"], v_min=0.0, v_max=1.0
    )
    _, ui["color"] = imgui.color_edit3(label="material color", col=ui["color"])
    _, ui["exposure"] = imgui.slider_float(
        label="exposure", v=ui["exposure"], v_min=0.1, v_max=10.0
    )
    _, ui["gamma"] = imgui.slider_float(
        label="gamma", v=ui["gamma"], v_min=1.0, v_max=3.0
    )
    imgui.text(f"specular: {SPEC_SIZE}^2 x {NUMBER_ROUGHNESS} levels")
    imgui.text(f"diffuse: {DIFF_SIZE}^2, BRDF LUT: {LUT_SIZE}^2")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 33 - PBR IBL",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        gl_major=4,
        gl_minor=1,
    )
