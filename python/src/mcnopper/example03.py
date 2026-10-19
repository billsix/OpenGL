"""McNopper Example 03: Textured rectangle with orthographic projection.

Original: McNopper/OpenGL Example03 - load desert.tga, draw it 1:1 with an ortho
projection sized to the window. Port preserves the C example's MVP =
P * V * M derivation but updates it on every reshape.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def init() -> bool:
    img: common.TgaImage = common.load_tga(common.find_media("desert.tga"))
    plane: common.Shape = common.make_rectangular_plane(
        half_x=img.width / 2.0, half_y=img.height / 2.0
    )

    prog: int = common.compile_program_files(
        script_file=__file__, vert="texture.vert.glsl", frag="grey.frag.glsl"
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
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    fmt = {1: GL.GL_RED, 3: GL.GL_RGB, 4: GL.GL_RGBA}[img.channels]
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        fmt,
        img.width,
        img.height,
        0,
        fmt,
        GL.GL_UNSIGNED_BYTE,
        img.pixels,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)
    GL.glClearColor(0, 0, 0, 0)

    state.update(
        prog=prog,
        vao=vao,
        tex=tex,
        num_indices=len(plane.indices),
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    proj: np.ndarray = common.ortho(
        left=-w / 2.0,
        right=w / 2.0,
        bottom=-h / 2.0,
        top=h / 2.0,
        near=1.0,
        far=100.0,
    )
    mvp: np.ndarray = proj @ view
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mvp_loc"], m=mvp)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["tex"])
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("desert.tga drawn 1:1 with orthographic projection.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 03 - textured rectangle",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
