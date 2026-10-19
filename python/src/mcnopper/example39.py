"""McNopper Example 39: Shader subroutines on a separable pipeline.

Original: McNopper/OpenGL Example39. Three triangles are drawn from the same VBO
but with different VS+FS subroutine choices each draw - red/green/blue
offsets and colors selected via glUniformSubroutinesuiv.
"""

from __future__ import annotations

import numpy as np
from OpenGL.GL import shaders

from mcnopper import common
from mcnopper.common import GL

state: dict = {}


def _make_separable(stage: int, src: str) -> int:
    obj = shaders.compileShader(src, stage)
    prog: int = GL.glCreateProgram()
    GL.glProgramParameteri(prog, GL.GL_PROGRAM_SEPARABLE, GL.GL_TRUE)
    GL.glAttachShader(prog, obj)
    GL.glLinkProgram(prog)
    return prog


def init() -> bool:
    points: np.ndarray = np.array(
        [
            -0.25,
            0,
            0,
            1,
            0.25,
            0,
            0,
            1,
            0,
            0.25,
            0,
            1,
        ],
        dtype=np.float32,
    )

    with open(
        common.find_shader(script_file=__file__, filename="offset.vert.glsl")
    ) as f:
        vs_src = f.read()
    with open(
        common.find_shader(
            script_file=__file__, filename="red_green_blue.frag.glsl"
        )
    ) as f:
        fs_src = f.read()

    vs_prog = _make_separable(stage=GL.GL_VERTEX_SHADER, src=vs_src)
    fs_prog = _make_separable(stage=GL.GL_FRAGMENT_SHADER, src=fs_src)

    pipeline: int = GL.glGenProgramPipelines(1)
    GL.glUseProgramStages(pipeline, GL.GL_VERTEX_SHADER_BIT, vs_prog)
    GL.glUseProgramStages(pipeline, GL.GL_FRAGMENT_SHADER_BIT, fs_prog)

    sub_off = {
        "red": GL.glGetSubroutineIndex(
            vs_prog, GL.GL_VERTEX_SHADER, "redOffset"
        ),
        "green": GL.glGetSubroutineIndex(
            vs_prog, GL.GL_VERTEX_SHADER, "greenOffset"
        ),
        "blue": GL.glGetSubroutineIndex(
            vs_prog, GL.GL_VERTEX_SHADER, "blueOffset"
        ),
    }
    sub_col = {
        "red": GL.glGetSubroutineIndex(
            fs_prog, GL.GL_FRAGMENT_SHADER, "redColor"
        ),
        "green": GL.glGetSubroutineIndex(
            fs_prog, GL.GL_FRAGMENT_SHADER, "greenColor"
        ),
        "blue": GL.glGetSubroutineIndex(
            fs_prog, GL.GL_FRAGMENT_SHADER, "blueColor"
        ),
    }

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

    state.update(
        pipeline=pipeline,
        vs_prog=vs_prog,
        fs_prog=fs_prog,
        vao=vao,
        sub_off=sub_off,
        sub_col=sub_col,
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def _draw_with(name: str) -> None:
    GL.glActiveShaderProgram(state["pipeline"], state["vs_prog"])
    GL.glUniformSubroutinesuiv(GL.GL_VERTEX_SHADER, 1, [state["sub_off"][name]])
    GL.glActiveShaderProgram(state["pipeline"], state["fs_prog"])
    GL.glUniformSubroutinesuiv(
        GL.GL_FRAGMENT_SHADER, 1, [state["sub_col"][name]]
    )
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBindVertexArray(state["vao"])
    for name in ("red", "green", "blue"):
        _draw_with(name)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("3 triangles from one VBO, subroutine-selected color/offset.")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 39 - subroutines",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
