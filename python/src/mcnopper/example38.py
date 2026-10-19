"""McNopper Example 38: Separable shader programs + program pipeline.

Original: McNopper/OpenGL Example38. Builds VS and FS as separable programs and
combines them via glProgramPipeline. Same red triangle as Example 02.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import shaders

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def init() -> bool:
    points: np.ndarray = np.array(
        [
            -0.5,
            0,
            0,
            1,
            0.5,
            0,
            0,
            1,
            0,
            0.5,
            0,
            1,
        ],
        dtype=np.float32,
    )

    with open(
        common.find_shader(script_file=__file__, filename="simple.vert.glsl")
    ) as f:
        vs_src = f.read()
    with open(
        common.find_shader(script_file=__file__, filename="red.frag.glsl")
    ) as f:
        fs_src = f.read()

    vs_obj = shaders.compileShader(vs_src, GL.GL_VERTEX_SHADER)
    fs_obj = shaders.compileShader(fs_src, GL.GL_FRAGMENT_SHADER)

    vs_prog: int = GL.glCreateProgram()
    GL.glProgramParameteri(vs_prog, GL.GL_PROGRAM_SEPARABLE, GL.GL_TRUE)
    GL.glAttachShader(vs_prog, vs_obj)
    GL.glLinkProgram(vs_prog)

    fs_prog: int = GL.glCreateProgram()
    GL.glProgramParameteri(fs_prog, GL.GL_PROGRAM_SEPARABLE, GL.GL_TRUE)
    GL.glAttachShader(fs_prog, fs_obj)
    GL.glLinkProgram(fs_prog)

    pipeline: int = GL.glGenProgramPipelines(1)
    GL.glUseProgramStages(pipeline, GL.GL_VERTEX_SHADER_BIT, vs_prog)
    GL.glUseProgramStages(pipeline, GL.GL_FRAGMENT_SHADER_BIT, fs_prog)

    vbo: int = common.make_vbo(points)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    loc: int = GL.glGetAttribLocation(program=vs_prog, name="a_vertex")
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glVertexAttribPointer(
        index=loc,
        size=4,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(loc)

    GL.glBindProgramPipeline(pipeline)
    GL.glClearColor(0, 0, 0, 0)

    state.update(pipeline=pipeline, vao=vao)
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindProgramPipeline(state["pipeline"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("VS + FS combined via glProgramPipeline.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 38 - separable shaders",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
