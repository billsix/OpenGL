"""McNopper Example 42: FXAA postprocess (Fast Approximate Anti-Aliasing).

Original: McNopper/OpenGL Example42. Phong-lit suzanne is rendered into an FBO,
then a fullscreen FXAA pass anti-aliases the result. Imgui exposes
the threshold/reduce/span knobs that the C version drove from keys.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
state: dict = {}
ui: dict = {
    "angle": 0.0,
    "auto_rotate": False,  # C: g_animationOn starts GL_FALSE (Space)
    "fxaa_on": True,
    "show_edges": False,
    "luma_thresh": 0.5,
    "mul_reduce_recip": 8.0,
    "min_reduce_recip": 128.0,
    "max_span": 8.0,
}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("elephant.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)

    prog_phong: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
    )
    prog_fxaa: int = common.compile_program_files(
        script_file=__file__, vert="fxaa.vert.glsl", frag="fxaa.frag.glsl"
    )

    color_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, color_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
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
        GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, SCREEN_WIDTH, SCREEN_HEIGHT
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
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

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
        0.2,
        0.2,
        0.2,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_light.diffuseColor"),
        0.8,
        0.8,
        0.8,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_light.specularColor"),
        0.8,
        0.8,
        0.8,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_material.ambientColor"),
        0.8,
        0.8,
        0.8,
        1,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_phong, name="u_material.diffuseColor"),
        0.8,
        0.8,
        0.8,
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

    GL.glUseProgram(prog_fxaa)
    vao_fxaa: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_fxaa)
    GL.glUniform2f(
        common.uniform(prog=prog_fxaa, name="u_texelStep"),
        1.0 / SCREEN_WIDTH,
        1.0 / SCREEN_HEIGHT,
    )

    GL.glClearColor(0.2, 0.2, 0.2, 1.0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog_phong=prog_phong,
        prog_fxaa=prog_fxaa,
        vao_obj=vao_obj,
        vao_fxaa=vao_fxaa,
        count=len(pos),
        color_tex=color_tex,
        fbo=fbo,
        proj_loc=common.uniform(prog=prog_phong, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog_phong, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog_phong, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog_phong, name="u_light.direction"),
        show_loc=common.uniform(prog=prog_fxaa, name="u_showEdges"),
        fxaa_on_loc=common.uniform(prog=prog_fxaa, name="u_fxaaOn"),
        luma_loc=common.uniform(prog=prog_fxaa, name="u_lumaThreshold"),
        mul_loc=common.uniform(prog=prog_fxaa, name="u_mulReduce"),
        min_loc=common.uniform(prog=prog_fxaa, name="u_minReduce"),
        span_loc=common.uniform(prog=prog_fxaa, name="u_maxSpan"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog_phong"])
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=SCREEN_WIDTH / SCREEN_HEIGHT, near=1.0, far=1000.0
    )
    view: np.ndarray = common.look_at(
        eye=[0, 0, 10], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light
    GL.glUniform3f(state["light_loc"], *light_view)
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    mv: np.ndarray = (
        view @ common.translate(x=0, y=-1.5, z=0) @ common.rotate_y(ui["angle"])
    )
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glBindVertexArray(state["vao_obj"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, state.get("w", 1), state.get("h", 1))
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["color_tex"])
    GL.glUseProgram(state["prog_fxaa"])
    GL.glUniform1i(state["show_loc"], int(ui["show_edges"]))
    GL.glUniform1i(state["fxaa_on_loc"], int(ui["fxaa_on"]))
    GL.glUniform1f(state["luma_loc"], ui["luma_thresh"])
    GL.glUniform1f(state["mul_loc"], 1.0 / ui["mul_reduce_recip"])
    GL.glUniform1f(state["min_loc"], 1.0 / ui["min_reduce_recip"])
    GL.glUniform1f(state["span_loc"], ui["max_span"])
    GL.glBindVertexArray(state["vao_fxaa"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    GL.glEnable(GL.GL_DEPTH_TEST)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["fxaa_on"] = imgui.checkbox(label="FXAA on", v=ui["fxaa_on"])
    _, ui["show_edges"] = imgui.checkbox(label="show edges", v=ui["show_edges"])
    _, ui["auto_rotate"] = imgui.checkbox(
        label="auto rotate", v=ui["auto_rotate"]
    )
    _, ui["luma_thresh"] = imgui.slider_float(
        label="luma threshold", v=ui["luma_thresh"], v_min=0.0, v_max=1.0
    )
    _, ui["mul_reduce_recip"] = imgui.slider_float(
        label="mul reduce 1/x", v=ui["mul_reduce_recip"], v_min=1.0, v_max=256.0
    )
    _, ui["min_reduce_recip"] = imgui.slider_float(
        label="min reduce 1/x", v=ui["min_reduce_recip"], v_min=1.0, v_max=512.0
    )
    _, ui["max_span"] = imgui.slider_float(
        label="max span", v=ui["max_span"], v_min=1.0, v_max=16.0
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 42 - FXAA",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        gl_major=4,
        gl_minor=3,
    )
