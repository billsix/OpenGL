"""McNopper Example 09: GPU particle system via ping-pong RGBA32F textures.

Original: McNopper/OpenGL Example09. 16x16 particles whose state lives in two
textures we ping-pong between. The "update" pass renders points into
the back-buffer texture, advancing position/velocity per fragment;
the "draw" pass samples the front-buffer texture and renders sprites.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcnopper import common
from mcnopper.common import GL

PARTICLE_TEXTURE_WIDTH: int = 16
state: dict = {}


def _make_position_texture() -> int:
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    init_data: np.ndarray = np.zeros(
        (PARTICLE_TEXTURE_WIDTH, PARTICLE_TEXTURE_WIDTH, 4), dtype=np.float32
    )
    init_data[..., 3] = -1.0  # dead -> respawn first frame
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA32F,
        PARTICLE_TEXTURE_WIDTH,
        PARTICLE_TEXTURE_WIDTH,
        0,
        GL.GL_RGBA,
        GL.GL_FLOAT,
        init_data,
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
    return tex


def init() -> bool:
    n: int = PARTICLE_TEXTURE_WIDTH
    points: np.ndarray = np.zeros((n * n, 2), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            points[y * n + x] = (x / n, y / n)

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="particle.vert.glsl",
        frag="particle.frag.glsl",
    )
    prog_update: int = common.compile_program_files(
        script_file=__file__,
        vert="update_points.vert.glsl",
        frag="update_points.frag.glsl",
    )

    sprite: int = common.upload_tga_2d(
        filename=common.find_media("particle.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
    )

    pos_tex: list[int] = [_make_position_texture(), _make_position_texture()]
    fbos: Any = GL.glGenFramebuffers(2)
    fbos = list(fbos) if hasattr(fbos, "__iter__") else [fbos]
    for fbo, tex in zip(fbos, pos_tex):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
        )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    vbo: int = common.make_vbo(points)

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo, name="a_vertex", size=2)

    GL.glUseProgram(prog_update)
    vao_u: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_u)
    common.attach_attrib(
        prog=prog_update, vao=vao_u, vbo=vbo, name="a_vertex", size=2
    )
    GL.glUniform1i(
        common.uniform(prog=prog_update, name="u_positionTexture"), 1
    )
    GL.glUniform1f(
        common.uniform(prog=prog_update, name="u_positionTextureWidth"),
        float(PARTICLE_TEXTURE_WIDTH),
    )

    GL.glUseProgram(prog)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, sprite)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)
    GL.glUniform1i(common.uniform(prog=prog, name="u_positionTexture"), 1)

    GL.glClearColor(0, 0, 0, 0)
    GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)

    state.update(
        prog=prog,
        prog_update=prog_update,
        vao=vao,
        vao_u=vao_u,
        vbo=vbo,
        sprite=sprite,
        pos_tex=pos_tex,
        fbos=fbos,
        flip=0,
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
        time_loc=common.uniform(prog=prog_update, name="u_time"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mvp_loc"], m=proj @ view)


def update(dt: float) -> bool:
    flip: int = state["flip"]

    GL.glUseProgram(state["prog_update"])
    GL.glUniform1f(state["time_loc"], dt)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbos"][(flip + 1) % 2])
    GL.glViewport(0, 0, PARTICLE_TEXTURE_WIDTH, PARTICLE_TEXTURE_WIDTH)
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["pos_tex"][flip])
    GL.glBindVertexArray(state["vao_u"])
    GL.glDrawArrays(
        GL.GL_POINTS, 0, PARTICLE_TEXTURE_WIDTH * PARTICLE_TEXTURE_WIDTH
    )

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, state.get("w", 1280), state.get("h", 720))
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["sprite"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["pos_tex"][flip])
    GL.glBindVertexArray(state["vao"])
    GL.glEnable(GL.GL_BLEND)
    GL.glDrawArrays(
        GL.GL_POINTS, 0, PARTICLE_TEXTURE_WIDTH * PARTICLE_TEXTURE_WIDTH
    )
    GL.glDisable(GL.GL_BLEND)

    state["flip"] = (flip + 1) % 2
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(
        f"{PARTICLE_TEXTURE_WIDTH}x{PARTICLE_TEXTURE_WIDTH} particles "
        "ping-ponged between two RGBA32F textures."
    )
    imgui.text(f"Frame: {state['flip']}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 09 - GPU particles",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
