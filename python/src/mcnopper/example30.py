"""McNopper Example 30: GPU compute-shader raytracer.

Original: McNopper/OpenGL Example30. Same scene as Example 29 but the trace runs
in a compute shader, with rays/positions/stacks/spheres/lights uploaded
as SSBOs at bindings 1-5. The image binding 0 is the destination texture
sampled by a fullscreen quad.

This port uploads the same packed structs the GLSL expects (with
PADDING_VALUE sentinels matching the C source layout).
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WIDTH: int = 640
HEIGHT: int = 480
LOCAL: int = 16
NUM_STACK_NODES: int = 31  # 2^5 - 1
# std430 layout of the Stack struct in raytrace.comp.glsl:
#   vec4 position (16) + vec3+pad direction (16) + vec4 color (16)
#   + vec4 hitPosition (16) + vec3+pad hitDirection (16)
#   + vec4 biasedPositiveHitPosition (16) + 4 * float (16) = 112B = 28 floats
STACK_FLOATS: int = 28
PADDING: float = -321.123

# (cx, cy, cz, w=1, radius, padding3, emiss[4], diff[4], spec[4], shin, alpha, refl, padding)
SPHERES: list[
    tuple[
        tuple[float, float, float, float],
        float,
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float, float],
        float,
        float,
        float,
    ]
] = [
    (
        (0, -10001, -20, 1),
        10000.0,
        (0, 0, 0, 1),
        (0.4, 0.4, 0.4, 1),
        (0, 0, 0, 1),
        0.0,
        1.0,
        0.0,
    ),
    (
        (0, 0, -10, 1),
        1.0,
        (0, 0, 0, 1),
        (0.8, 0.8, 0.8, 1),
        (0.8, 0.8, 0.8, 1),
        20.0,
        0.2,
        1.0,
    ),
    (
        (1, -0.75, -7, 1),
        0.25,
        (0, 0, 0, 1),
        (0.8, 0.8, 0.8, 1),
        (0.8, 0.8, 0.8, 1),
        20.0,
        1.0,
        0.8,
    ),
    (
        (2, 1, -16, 1),
        2.0,
        (0, 0, 0, 1),
        (0, 0, 0.8, 1),
        (0.8, 0.8, 0.8, 1),
        20.0,
        1.0,
        0.2,
    ),
    (
        (-2, 0.25, -6, 1),
        1.25,
        (0, 0, 0, 1),
        (0, 0.8, 0, 1),
        (0.8, 0.8, 0.8, 1),
        20.0,
        1.0,
        0.2,
    ),
    (
        (3, 0, -8, 1),
        1.0,
        (0, 0, 0, 1),
        (0.8, 0, 0, 1),
        (0.8, 0.8, 0.8, 1),
        20.0,
        1.0,
        0.2,
    ),
]
LIGHTS: list[
    tuple[tuple[float, float, float, float], tuple[float, float, float, float]]
] = [((0, 5, -5, 1), (1, 1, 1, 1))]
state: dict = {}


def _build_ray_grids() -> tuple[np.ndarray, np.ndarray]:
    fov: float = 30.0
    aspect: float = WIDTH / HEIGHT
    angle: np.floating = np.tan(np.radians(fov / 2.0))
    dirs: np.ndarray = np.zeros(
        (HEIGHT, WIDTH, 4), dtype=np.float32
    )  # padding for std430
    for y in range(HEIGHT):
        # y=0 = bottom row so the compute-shader output aligns with the
        # fullscreen quad's UV.v = 0 at the bottom of the screen.
        ndc_y: np.floating = (2 * (y + 0.5) / HEIGHT - 1) * angle
        for x in range(WIDTH):
            ndc_x: np.floating = (2 * (x + 0.5) / WIDTH - 1) * angle * aspect
            d: np.ndarray = np.array([ndc_x, ndc_y, -1.0], dtype=np.float32)
            d /= np.linalg.norm(d)
            dirs[y, x, :3] = d
    pos: np.ndarray = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    pos[..., 3] = 1.0
    return dirs.reshape(-1, 4), pos.reshape(-1, 4)


def _pack_spheres() -> np.ndarray:
    out: list[float] = []
    for center, r, em, df, sp, sh, al, rfl in SPHERES:
        # Sphere struct: center[4], radius, padding[3], material{ emiss[4], diff[4], spec[4], shin, alpha, refl, pad }
        out.extend(center)
        out.append(r)
        out.extend([PADDING, PADDING, PADDING])
        out.extend(em)
        out.extend(df)
        out.extend(sp)
        out.extend([sh, al, rfl, PADDING])
    return np.array(out, dtype=np.float32)


def _pack_lights() -> np.ndarray:
    out: list[float] = []
    for pos, color in LIGHTS:
        out.extend(pos)
        out.extend(color)
    return np.array(out, dtype=np.float32)


def init() -> bool:
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen.vert.glsl",
        frag="texture.frag.glsl",
    )
    prog_comp: int = common.compile_program_files(
        script_file=__file__, comp="raytrace.comp.glsl"
    )

    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA8,
        WIDTH,
        HEIGHT,
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

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)

    dirs, pos = _build_ray_grids()
    stack: np.ndarray = np.zeros(
        (WIDTH * HEIGHT * STACK_FLOATS * NUM_STACK_NODES,), dtype=np.float32
    )

    def _ssbo(binding: int, data: np.ndarray) -> int:
        ssbo: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, ssbo)
        arr: np.ndarray = np.ascontiguousarray(data, dtype=np.float32)
        GL.glBufferData(
            target=GL.GL_SHADER_STORAGE_BUFFER,
            size=arr.nbytes,
            data=arr,
            usage=GL.GL_STATIC_DRAW,
        )
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, binding, ssbo)
        return ssbo

    dir_ssbo: int = _ssbo(binding=1, data=dirs)
    pos_ssbo: int = _ssbo(binding=2, data=pos)
    stack_ssbo: int = _ssbo(binding=3, data=stack)
    sphere_ssbo: int = _ssbo(binding=4, data=_pack_spheres())
    light_ssbo: int = _ssbo(binding=5, data=_pack_lights())

    state.update(
        prog=prog,
        prog_comp=prog_comp,
        vao=vao,
        tex=tex,
        ssbos=[dir_ssbo, pos_ssbo, stack_ssbo, sphere_ssbo, light_ssbo],
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
    GL.glUseProgram(state["prog_comp"])
    GL.glDispatchCompute(WIDTH // LOCAL, HEIGHT // LOCAL, 1)
    GL.glMemoryBarrier(GL.GL_TEXTURE_FETCH_BARRIER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["tex"])
    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(f"{WIDTH}x{HEIGHT}, local size {LOCAL}x{LOCAL}")
    imgui.text(f"{NUM_STACK_NODES} stack nodes (depth 5 ray tree)")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 30 - GPU raytracer",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=3,
    )
