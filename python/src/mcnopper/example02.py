"""McNopper Example 02: First triangle in a VBO.

Original: McNopper/OpenGL Example02 (Norbert Nopper, MIT). A red triangle in NDC,
drawn from a single VBO bound through a VAO. The Python port reuses the
original GLSL shaders verbatim.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def init() -> bool:
    points: np.ndarray = np.array(
        [
            -0.5,
            0.0,
            0.0,
            1.0,
            0.5,
            0.0,
            0.0,
            1.0,
            0.0,
            0.5,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    )

    prog: int = common.compile_program_files(
        script_file=__file__, vert="simple.vert.glsl", frag="red.frag.glsl"
    )
    vbo: int = common.make_vbo(points)

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo, name="a_vertex", size=4)

    GL.glClearColor(0, 0, 0, 0)
    state.update(prog=prog, vbo=vbo, vao=vao)
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
    return True


def terminate() -> None:
    GL.glDeleteBuffers(1, [state["vbo"]])
    GL.glDeleteVertexArrays(1, [state["vao"]])
    GL.glDeleteProgram(state["prog"])


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("Red triangle drawn from a single VBO.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 02 - first triangle",
        callbacks=common.DemoCallbacks(
            init=init,
            reshape=reshape,
            update=update,
            terminate=terminate,
            imgui_panel=panel,
        ),
        width=640,
        height=480,
    )
