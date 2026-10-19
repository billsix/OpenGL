"""McNopper Example 21: Compute shader writes a texture, FS samples it.

Original: McNopper/OpenGL Example21. The compute program is dispatched at
(width/16, height/16, 1) and writes RGBA8 directly to an image-bound
texture; the render pass samples that texture as a regular sampler2D.
GL 4.3 minimum.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

IMAGE_W: int = 640
IMAGE_H: int = 480
LOCAL_SIZE: int = 16
state: dict = {}


def init() -> bool:
    plane: common.Shape = common.make_rectangular_plane(
        half_x=IMAGE_W / 2.0, half_y=IMAGE_H / 2.0
    )
    prog: int = common.compile_program_files(
        script_file=__file__, vert="texture.vert.glsl", frag="texture.frag.glsl"
    )
    prog_comp: int = common.compile_program_files(
        script_file=__file__, comp="texture.comp.glsl"
    )

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(plane.vertices)
    vbo_t: int = common.make_vbo(plane.texcoords)
    vbo_i: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA8,
        IMAGE_W,
        IMAGE_H,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        None,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glBindImageTexture(
        0, tex, 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA8
    )

    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)
    GL.glUseProgram(prog_comp)
    GL.glUniform1i(common.uniform(prog=prog_comp, name="u_texture"), 0)

    GL.glClearColor(0, 0, 0, 0)

    state.update(
        prog=prog,
        prog_comp=prog_comp,
        vao=vao,
        n=len(plane.indices),
        tex=tex,
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    proj: np.ndarray = common.ortho(
        left=-w / 2, right=w / 2, bottom=-h / 2, top=h / 2, near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mvp_loc"], m=proj @ view)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog_comp"])
    GL.glDispatchCompute(IMAGE_W // LOCAL_SIZE, IMAGE_H // LOCAL_SIZE, 1)
    GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n"], GL.GL_UNSIGNED_INT, None)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(
        f"image: {IMAGE_W}x{IMAGE_H}, local size {LOCAL_SIZE}x{LOCAL_SIZE}"
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 21 - compute shader",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=3,
    )
