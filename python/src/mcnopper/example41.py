"""McNopper Example 41: GPU FFT-based Tessendorf ocean.

Original: McNopper/OpenGL Example41 (Tessendorf 1999 + GPU FFT). Three compute
passes: update H(t) from Phillips spectrum, run a row-then-column
butterfly inverse FFT to produce the displacement, then build the
normal map. The vertex shader displaces a 512x512 grid, the fragment
shader does directional Lambert shading.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

N: int = 512
LENGTH: float = 250.0
GRAVITY: float = 9.81
AMPLITUDE: float = 0.002
WIND_SPEED: float = 10.0
WIND_DIR: tuple[float, float] = (1.0, 1.0)
LPWA: float = WIND_SPEED * WIND_SPEED / GRAVITY
state: dict = {}


def _phillips(
    amplitude: float,
    largest_wave: float,
    wave_dir: tuple[float, float],
    wind_dir: np.ndarray,
) -> float:
    k: float = math.hypot(*wave_dir)
    if largest_wave == 0 or k == 0:
        return 0.0
    dot_wind: float = wave_dir[0] * wind_dir[0] + wave_dir[1] * wind_dir[1]
    return (
        amplitude
        * math.exp(-1.0 / (k * largest_wave) ** 2)
        / (k**4)
        * dot_wind**2
    )


def _butterfly_indices(n: int) -> np.ndarray:
    """Bit-reverse permutation of 0..n-1 (FFT shuffle)."""
    bits: int = int(math.log2(n))
    out: np.ndarray = np.zeros(n, dtype=np.int32)
    for i in range(n):
        v: int = i
        r: int = 0
        for _ in range(bits):
            r = (r << 1) | (v & 1)
            v >>= 1
        out[i] = r
    return out


def init() -> bool:
    print(f"[Example 41] Generating Phillips spectrum H0 ({N}x{N})...")  # noqa: T201
    rng: np.random.Generator = np.random.default_rng(0)
    wd: np.ndarray = np.array(WIND_DIR, dtype=np.float64)
    wd /= np.linalg.norm(wd)
    h0: np.ndarray = np.zeros((N, N, 2), dtype=np.float32)
    inv_sqrt2: float = 1.0 / math.sqrt(2.0)
    for i in range(N):
        ky: float = (N / 2.0 - i) * (2.0 * math.pi / LENGTH)
        for k in range(N):
            kx: float = (-N / 2.0 + k) * (2.0 * math.pi / LENGTH)
            ph: float = _phillips(
                amplitude=AMPLITUDE,
                largest_wave=LPWA,
                wave_dir=(kx, ky),
                wind_dir=wd,
            )
            h0[i, k, 0] = inv_sqrt2 * rng.normal() * ph
            h0[i, k, 1] = inv_sqrt2 * rng.normal() * ph

    butterfly: np.ndarray = _butterfly_indices(N).astype(np.float32)
    steps: int = int(math.log2(N))

    prog_ht: int = common.compile_program_files(
        script_file=__file__, comp="ocean_update_ht.comp.glsl"
    )
    prog_fft: int = common.compile_program_files(
        script_file=__file__, comp="ocean_fft.comp.glsl"
    )
    prog_norm: int = common.compile_program_files(
        script_file=__file__, comp="ocean_update_normal.comp.glsl"
    )
    prog: int = common.compile_program_files(
        script_file=__file__, vert="ocean.vert.glsl", frag="ocean.frag.glsl"
    )

    GL.glUseProgram(prog_fft)
    GL.glUniform1i(common.uniform(prog=prog_fft, name="u_steps"), steps)

    def make_2d_rg32f(data: np.ndarray | None = None) -> int:
        t: int = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RG32F,
            N,
            N,
            0,
            GL.GL_RG,
            GL.GL_FLOAT,
            data,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        return t

    h0_tex: int = make_2d_rg32f(h0)
    ht_tex: int = make_2d_rg32f()
    disp_tex: list[int] = [make_2d_rg32f(), make_2d_rg32f()]
    normal_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, normal_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA32F,
        N,
        N,
        0,
        GL.GL_RGBA,
        GL.GL_FLOAT,
        None,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )

    indices_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_1D, indices_tex)
    GL.glTexImage1D(
        GL.GL_TEXTURE_1D, 0, GL.GL_R32F, N, 0, GL.GL_RED, GL.GL_FLOAT, butterfly
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_1D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )

    grid: common.Shape = common.make_rectangular_grid_plane(
        horizontal_extend=LENGTH,
        vertical_extend=LENGTH,
        rows=N - 1,
        columns=N - 1,
    )
    rot: np.ndarray = common.rotate_x(-90.0)
    rotated: np.ndarray = (rot @ grid.vertices.T).T

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(rotated)
    vbo_t: int = common.make_vbo(grid.texcoords)
    vbo_i: int = common.make_vbo(
        data=grid.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    GL.glUniform3f(common.uniform(prog=prog, name="u_lightDirection"), *light)
    GL.glUniform4f(common.uniform(prog=prog, name="u_color"), 0, 0.8, 0.8, 1.0)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        prog_ht=prog_ht,
        prog_fft=prog_fft,
        prog_norm=prog_norm,
        vao=vao,
        n_indices=len(grid.indices),
        h0_tex=h0_tex,
        ht_tex=ht_tex,
        disp_tex=disp_tex,
        normal_tex=normal_tex,
        indices_tex=indices_tex,
        total_time=0.0,
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        t_loc=common.uniform(prog=prog_ht, name="u_totalTime"),
        col_loc=common.uniform(prog=prog_fft, name="u_processColumn"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=2000.0
    )
    view: np.ndarray = common.look_at(
        eye=[0, 5, LENGTH / 2], center=[0, 0, 0], up=[0, 1, 0]
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mvp_loc"], m=proj @ view)
    common.upload_matrix3(loc=state["nm_loc"], m=np.eye(3, dtype=np.float32))


def update(dt: float) -> bool:
    state["total_time"] += dt
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    GL.glUseProgram(state["prog_ht"])
    GL.glBindImageTexture(
        0, state["h0_tex"], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RG32F
    )
    GL.glBindImageTexture(
        1, state["ht_tex"], 0, GL.GL_FALSE, 0, GL.GL_WRITE_ONLY, GL.GL_RG32F
    )
    GL.glUniform1f(state["t_loc"], state["total_time"])
    GL.glDispatchCompute(N, N, 1)
    GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    GL.glUseProgram(state["prog_fft"])
    GL.glBindImageTexture(
        2, state["indices_tex"], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_R32F
    )

    GL.glBindImageTexture(
        0, state["ht_tex"], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RG32F
    )
    GL.glBindImageTexture(
        1,
        state["disp_tex"][0],
        0,
        GL.GL_FALSE,
        0,
        GL.GL_WRITE_ONLY,
        GL.GL_RG32F,
    )
    GL.glUniform1i(state["col_loc"], 0)
    GL.glDispatchCompute(1, N, 1)
    GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    GL.glBindImageTexture(
        0, state["disp_tex"][0], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RG32F
    )
    GL.glBindImageTexture(
        1,
        state["disp_tex"][1],
        0,
        GL.GL_FALSE,
        0,
        GL.GL_WRITE_ONLY,
        GL.GL_RG32F,
    )
    GL.glUniform1i(state["col_loc"], 1)
    GL.glDispatchCompute(1, N, 1)
    GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    GL.glUseProgram(state["prog_norm"])
    GL.glBindImageTexture(
        0, state["disp_tex"][1], 0, GL.GL_FALSE, 0, GL.GL_READ_ONLY, GL.GL_RG32F
    )
    GL.glBindImageTexture(
        1,
        state["normal_tex"],
        0,
        GL.GL_FALSE,
        0,
        GL.GL_WRITE_ONLY,
        GL.GL_RGBA32F,
    )
    GL.glDispatchCompute(N, N, 1)
    GL.glMemoryBarrier(
        GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL.GL_TEXTURE_FETCH_BARRIER_BIT  # ty: ignore[unsupported-operator]
    )

    GL.glUseProgram(state["prog"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["disp_tex"][1])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["normal_tex"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(f"{N}x{N} ocean grid, {LENGTH}m extent")
    imgui.text(f"t = {state['total_time']:.1f}s")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 41 - FFT ocean",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        samples=4,
        gl_major=4,
        gl_minor=3,
    )
