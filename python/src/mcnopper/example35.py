"""McNopper Example 35: Order-independent transparency via depth peeling.

Original: McNopper/OpenGL Example35. 8 peel passes render the dragon to layers of
a 2D texture array; each layer's depth test compares against the
previous layer's depth (sampler2DShadow with GL_GREATER). A fullscreen
blend pass composites the layers back-to-front in the FS.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
LAYERS: int = 8
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("dragon.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="phong_depth_peel.vert.glsl",
        frag="phong_depth_peel.frag.glsl",
    )
    prog_blend: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen_blend.vert.glsl",
        frag="fullscreen_blend.frag.glsl",
    )

    color_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, color_tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_2D_ARRAY,
        0,
        GL.GL_RGBA8,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        LAYERS,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        None,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D_ARRAY, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    depth_tex = []
    for _ in range(2):
        t: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT32F,
            SCREEN_WIDTH,
            SCREEN_HEIGHT,
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
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_COMPARE_MODE,
            GL.GL_COMPARE_REF_TO_TEXTURE,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_GREATER
        )
        depth_tex.append(t)

    fbo: int = GL.glGenFramebuffers(1)

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    bias: np.ndarray = np.array(
        [
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_biasMatrix"), m=bias
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.ambientColor"),
        0.3,
        0.3,
        0.3,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.diffuseColor"), 1, 1, 1, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.specularColor"), 1, 1, 1, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.ambientColor"), 0, 1, 0, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"), 0, 1, 0, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.specularColor"), 1, 1, 1, 1
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_material.specularExponent"), 20.0
    )
    GL.glUniform1f(common.uniform(prog=prog, name="u_material.alpha"), 0.5)
    GL.glUniform1i(common.uniform(prog=prog, name="u_peelTexture"), 1)

    GL.glUseProgram(prog_blend)
    vao_blend: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_blend)
    GL.glUniform1i(
        common.uniform(prog=prog_blend, name="u_framebufferTexture"), 0
    )
    GL.glUniform1i(common.uniform(prog=prog_blend, name="u_layers"), LAYERS)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 3], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light
    GL.glUseProgram(prog)
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_light.direction"), *light_view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    state.update(
        prog=prog,
        prog_blend=prog_blend,
        vao=vao,
        vao_blend=vao_blend,
        count=len(pos),
        view=view,
        color_tex=color_tex,
        depth_tex=depth_tex,
        fbo=fbo,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        layer_loc=common.uniform(prog=prog, name="u_layer"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    GL.glUseProgram(state["prog"])
    common.upload_matrix(
        loc=state["proj_loc"],
        m=common.perspective(
            fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
        ),
    )


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0

    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    mv: np.ndarray = state["view"] @ common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, 0)
    GL.glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

    for peel_layer in range(LAYERS):
        GL.glUniform1i(state["layer_loc"], peel_layer)
        peel_depth = peel_layer % 2
        default_depth = (peel_layer + 1) % 2
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, state["depth_tex"][peel_depth])
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
        GL.glFramebufferTextureLayer(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            state["color_tex"],
            0,
            peel_layer,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            state["depth_tex"][default_depth],
            0,
        )
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])

    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_ARRAY, state["color_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, state.get("w", 1), state.get("h", 1))
    GL.glUseProgram(state["prog_blend"])
    GL.glBindVertexArray(state["vao_blend"])
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    GL.glEnable(GL.GL_DEPTH_TEST)
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
    imgui.text(f"{LAYERS} peel layers @ {SCREEN_WIDTH}x{SCREEN_HEIGHT}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 35 - depth peeling",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
    )
