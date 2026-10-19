"""McNopper Example 32: Image-based BRDF lighting with HDR panorama.

Original: McNopper/OpenGL Example32. Cook-Torrance BRDF on the venusm OBJ; the
panoramic doge2.hdr is used both as background (inside-out sphere) and
as the importance-sampled environment in the BRDF integral. MSAA FBO,
tone-mapping/gamma resolve in the fullscreen pass.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
MSAA: int = 4
MAX_M: int = 9  # C: the sample-count exponent key clamp
state: dict = {}
ui: dict = {"m": 6, "roughness": 0.1, "R0": 0.2, "exposure": 3.0, "gamma": 2.2}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("venusm.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    bg_sphere: common.Shape = common.make_sphere(radius=500.0, slices=32)

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

    panorama: int = common.upload_hdr_2d(common.find_media("doge2.hdr"))
    GL.glBindTexture(GL.GL_TEXTURE_2D, panorama)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_MIRRORED_REPEAT
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    # MSAA framebuffer
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
    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
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
    GL.glUniform1i(common.uniform(prog=prog_brdf, name="u_panoramaTexture"), 0)
    GL.glUniform3f(
        common.uniform(prog=prog_brdf, name="u_colorMaterial"), 0.8, 0.8, 0.8
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
    GL.glUniform1i(common.uniform(prog=prog_bg, name="u_panoramaTexture"), 0)

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
        panorama=panorama,
        full_tex=full_tex,
        fbo=fbo,
        vp_loc_m=common.uniform(prog=prog_brdf, name="u_viewProjectionMatrix"),
        m_loc_m=common.uniform(prog=prog_brdf, name="u_modelMatrix"),
        nm_loc_m=common.uniform(prog=prog_brdf, name="u_normalMatrix"),
        ns_loc=common.uniform(prog=prog_brdf, name="u_numberSamples"),
        m_loc=common.uniform(prog=prog_brdf, name="u_m"),
        bff_loc=common.uniform(prog=prog_brdf, name="u_binaryFractionFactor"),
        rough_loc=common.uniform(prog=prog_brdf, name="u_roughnessMaterial"),
        R0_loc=common.uniform(prog=prog_brdf, name="u_R0Material"),
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
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["panorama"])
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    GL.glEnable(GL.GL_MULTISAMPLE)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    # Background
    GL.glFrontFace(GL.GL_CW)
    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["vp_loc_bg"], m=vp)
    GL.glBindVertexArray(state["vao_bg"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_bg"], GL.GL_UNSIGNED_INT, None)
    GL.glFrontFace(GL.GL_CCW)

    # BRDF model
    GL.glUseProgram(state["prog_brdf"])
    GL.glUniform1ui(state["ns_loc"], 1 << ui["m"])
    GL.glUniform1ui(state["m_loc"], ui["m"])
    GL.glUniform1f(state["bff_loc"], 1.0 / (2 ** ui["m"] - 1))
    GL.glUniform1f(state["rough_loc"], ui["roughness"])
    GL.glUniform1f(state["R0_loc"], ui["R0"])
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

    _, ui["m"] = imgui.slider_int(
        label="samples (2^m)", v=ui["m"], v_min=1, v_max=MAX_M
    )
    _, ui["roughness"] = imgui.slider_float(
        label="roughness", v=ui["roughness"], v_min=0.0, v_max=1.0
    )
    _, ui["R0"] = imgui.slider_float(
        label="R0 (Schlick)", v=ui["R0"], v_min=0.0, v_max=1.0
    )
    _, ui["exposure"] = imgui.slider_float(
        label="exposure", v=ui["exposure"], v_min=0.1, v_max=10.0
    )
    _, ui["gamma"] = imgui.slider_float(
        label="gamma", v=ui["gamma"], v_min=1.0, v_max=3.0
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 32 - IBL BRDF",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        gl_major=4,
        gl_minor=1,
    )
