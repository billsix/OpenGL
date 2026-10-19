"""McNopper Example 01: Window setup + clear color.

Original: McNopper/OpenGL Example01/src/main.c (Norbert Nopper, MIT). The C version
opens an EGL/GLFW window and clears the framebuffer to blue every frame.
The Python port does the same with glfw + PyOpenGL, plus an imgui panel
that lets you change the clear color at runtime.
"""

from __future__ import annotations

from mcnopper import common
from mcnopper.common import GL

clear_color: list[float] = [0.0, 0.0, 1.0, 0.0]


def init() -> bool:
    GL.glClearColor(*clear_color)
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    GL.glClearColor(*clear_color)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    changed, col = imgui.color_edit3(label="clear", col=clear_color[:3])
    if changed:
        clear_color[0:3] = list(col)


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 01 - clear color",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
