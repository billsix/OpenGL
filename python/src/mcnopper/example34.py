"""McNopper Example 34: Subsurface scattering using depth-from-light.

Original: McNopper/OpenGL Example34. Render dragon depth from the light's POV
into a 4096^2 depth texture; in the scene pass the FS samples back-side
depth and computes a wrap-lit + scatter term whose width/falloff are
exposed as imgui sliders.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

DEPTH_SIZE: int = 4096
LIGHT_POS: list[float] = [2.0, 2.0, 3.0]
NEAR, FAR = 1.0, 100.0
state: dict = {}
ui: dict = {
    "angle": 0.0,
    "auto_rotate": True,
    "wrap": 0.1,
    "width": 0.5,
    "falloff": 5.0,
}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("dragon.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)

    prog_depth: int = common.compile_program_files(
        script_file=__file__,
        vert="renderdepthmap.vert.glsl",
        frag="renderdepthmap.frag.glsl",
    )
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="subsurfacescattering.vert.glsl",
        frag="subsurfacescattering.frag.glsl",
    )

    depth_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_DEPTH_COMPONENT32F,
        DEPTH_SIZE,
        DEPTH_SIZE,
        0,
        GL.GL_DEPTH_COMPONENT,
        GL.GL_FLOAT,
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

    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_TEXTURE_2D,
        depth_tex,
        0,
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glUniform1i(common.uniform(prog=prog, name="u_depthPassTexture"), 0)

    GL.glUseProgram(prog_depth)
    vao_d: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_d)
    common.attach_attrib(
        prog=prog_depth, vao=vao_d, vbo=vbo_v, name="a_vertex", size=4
    )

    bias: np.ndarray = np.array(
        [
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    proj_depth: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=1.0, near=NEAR, far=FAR
    )
    state["depth_pass_matrix"] = bias @ proj_depth
    GL.glUseProgram(prog_depth)
    common.upload_matrix(
        loc=common.uniform(prog=prog_depth, name="u_projectionMatrix"),
        m=proj_depth,
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        prog_depth=prog_depth,
        vao=vao,
        vao_d=vao_d,
        count=len(pos),
        depth_tex=depth_tex,
        fbo=fbo,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        view_loc=common.uniform(prog=prog, name="u_viewMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        dpm_loc=common.uniform(prog=prog, name="u_depthPassMatrix"),
        light_loc=common.uniform(prog=prog, name="u_lightDirection"),
        diff_loc=common.uniform(prog=prog, name="u_diffuseColor"),
        scatter_loc=common.uniform(prog=prog, name="u_scatterColor"),
        nf_loc=common.uniform(prog=prog, name="u_nearFar"),
        wrap_loc=common.uniform(prog=prog, name="u_wrap"),
        width_loc=common.uniform(prog=prog, name="u_scatterWidth"),
        fall_loc=common.uniform(prog=prog, name="u_scatterFalloff"),
        mv_loc_d=common.uniform(prog=prog_depth, name="u_modelViewMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 20.0 * dt) % 360.0

    light_view: np.ndarray = common.look_at(
        eye=LIGHT_POS, center=[0, 0, 0], up=[0, 1, 0]
    )
    depth_pass: np.ndarray = state["depth_pass_matrix"] @ light_view

    # The depth texture is this FBO's attachment: unbind it before the pass
    # that renders into it (C main.c does the same).
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glViewport(0, 0, DEPTH_SIZE, DEPTH_SIZE)
    GL.glColorMask(False, False, False, False)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog_depth"])
    model_d: np.ndarray = common.rotate_y(ui["angle"]) @ common.scale_m(
        x=1.05, y=1.05, z=1.05
    )
    mv_d: np.ndarray = light_view @ model_d
    common.upload_matrix(loc=state["mv_loc_d"], m=mv_d)
    GL.glBindVertexArray(state["vao_d"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glColorMask(True, True, True, True)
    GL.glViewport(0, 0, state.get("w", 1), state.get("h", 1))
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["depth_tex"])

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])
    view: np.ndarray = common.look_at(
        eye=[0, 0, 3], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array(LIGHT_POS, dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view_dir: np.ndarray = view[:3, :3] @ light
    common.upload_matrix(loc=state["view_loc"], m=view)
    common.upload_matrix(loc=state["dpm_loc"], m=depth_pass)
    GL.glUniform3f(state["light_loc"], *light_view_dir)

    model: np.ndarray = common.rotate_y(ui["angle"])
    mv: np.ndarray = view @ model
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glUniform4f(state["diff_loc"], 0.8, 0.0, 0.0, 1.0)
    GL.glUniform4f(state["scatter_loc"], 0.8, 0.8, 0.0, 1.0)
    GL.glUniform2f(state["nf_loc"], NEAR, FAR)
    GL.glUniform1f(state["wrap_loc"], ui["wrap"])
    GL.glUniform1f(state["width_loc"], ui["width"])
    GL.glUniform1f(state["fall_loc"], ui["falloff"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])
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
    # Ranges are the C key handler's clamps.
    _, ui["wrap"] = imgui.slider_float(
        label="wrap", v=ui["wrap"], v_min=0.0, v_max=1.0
    )
    _, ui["width"] = imgui.slider_float(
        label="scatter width", v=ui["width"], v_min=0.0, v_max=1.0
    )
    _, ui["falloff"] = imgui.slider_float(
        label="scatter falloff", v=ui["falloff"], v_min=0.0, v_max=50.0
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 34 - SSS",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
