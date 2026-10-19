"""McNopper Example 29: CPU-side recursive raytracer baked into a texture.

Original: McNopper/OpenGL Example29 (Scratchapixel-style raytracer). 6 spheres with
reflection/refraction; trace runs once on the CPU and dumps RGB into a
WIDTHxHEIGHT texture sampled by a fullscreen quad.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WIDTH, HEIGHT = 640, 480
MAX_DEPTH: int = 5
AIR: float = 1.0
BUBBLE: float = 1.06
ETA: float = AIR / BUBBLE
R0: float = ((AIR - BUBBLE) ** 2) / ((AIR + BUBBLE) ** 2)

SPHERES: list[tuple[Any, ...]] = [
    # center, radius, emissive, diffuse, specular, shininess, alpha, reflectivity
    (
        (0, -10001, -20),
        10000.0,
        (0, 0, 0),
        (0.4, 0.4, 0.4),
        (0, 0, 0),
        0.0,
        1.0,
        0.0,
    ),
    (
        (0, 0, -10),
        1.0,
        (0, 0, 0),
        (0.8, 0.8, 0.8),
        (0.8, 0.8, 0.8),
        20.0,
        0.2,
        1.0,
    ),
    (
        (1, -0.75, -7),
        0.25,
        (0, 0, 0),
        (0.8, 0.8, 0.8),
        (0.8, 0.8, 0.8),
        20.0,
        1.0,
        0.8,
    ),
    (
        (2, 1, -16),
        2.0,
        (0, 0, 0),
        (0.0, 0.0, 0.8),
        (0.8, 0.8, 0.8),
        20.0,
        1.0,
        0.2,
    ),
    (
        (-2, 0.25, -6),
        1.25,
        (0, 0, 0),
        (0.0, 0.8, 0.0),
        (0.8, 0.8, 0.8),
        20.0,
        1.0,
        0.2,
    ),
    (
        (3, 0, -8),
        1.0,
        (0, 0, 0),
        (0.8, 0.0, 0.0),
        (0.8, 0.8, 0.8),
        20.0,
        1.0,
        0.2,
    ),
]
LIGHTS: list[tuple[Any, ...]] = [((0, 5, -5), (1, 1, 1))]
state: dict = {}


def _intersect(
    origin: np.ndarray, direction: np.ndarray, center: tuple, radius: float
) -> tuple[float, bool] | None:
    """``glusIntersectRaySpheref`` + the C's ``findNearestSphere`` use of
    it: ``(t, inside)`` for the nearest hit, or ``None``. A ray starting
    inside the sphere hits its far side (GLUS clamps the near root to 0 and
    flags ``inside``); a ray starting outside and pointing away misses.
    """
    m: np.ndarray = np.array(origin) - np.array(center)
    b: float = np.dot(m, direction)
    c: float = np.dot(m, m) - radius * radius
    if b > 0 and c > 0:
        return None
    discriminant: float = b * b - c
    if discriminant < 0:
        return None
    sqrt_d: float = np.sqrt(discriminant)
    t0, t1 = -b - sqrt_d, -b + sqrt_d
    inside: bool = t0 < 0
    if inside:
        return t1, True
    return t0, False


def _trace(
    origin: np.ndarray, direction: np.ndarray, depth: int = 0
) -> np.ndarray:
    near: float = float("inf")
    near_sphere: tuple[Any, ...] | None = None
    near_inside: bool = False
    for s in SPHERES:
        hit: tuple[float, bool] | None = _intersect(
            origin=origin, direction=direction, center=s[0], radius=s[1]
        )
        if hit and hit[0] < near:
            near, near_inside = hit
            near_sphere = s
    if near_sphere is None:
        return np.array([0.8, 0.8, 0.8])
    center, radius, emiss, diff, spec, shin, alpha, refl = near_sphere
    hit_pos: np.ndarray = np.array(origin) + direction * near
    nrm: np.ndarray = (hit_pos - np.array(center)) / radius
    if near_inside:
        nrm = -nrm
    bias: float = 1e-4
    pos_pos: np.ndarray = hit_pos + nrm * bias
    pos_neg: np.ndarray = hit_pos - nrm * bias

    color: np.ndarray = np.zeros(3)
    fresnel: float = R0 + (1 - R0) * (1 - max(0, -np.dot(direction, nrm))) ** 5

    refl_color: np.ndarray = np.zeros(3)
    if refl > 0 and depth < MAX_DEPTH:
        d: np.ndarray = direction - 2 * np.dot(direction, nrm) * nrm
        d /= np.linalg.norm(d)
        refl_color = _trace(origin=pos_pos, direction=d, depth=depth + 1)

    refr_color: np.ndarray = np.zeros(3)
    if alpha < 1.0 and depth < MAX_DEPTH:
        eta: float = 1.0 / ETA if near_inside else ETA
        cos_i: float = -np.dot(direction, nrm)
        sin_t2: float = eta * eta * (1 - cos_i * cos_i)
        if sin_t2 < 1:
            d = eta * direction + (eta * cos_i - np.sqrt(1 - sin_t2)) * nrm
            d /= np.linalg.norm(d)
            refr_color = _trace(origin=pos_neg, direction=d, depth=depth + 1)
        else:
            # Total internal reflection: no refracted ray, so reflect fully
            # (matches upstream Example29's TIR fix).
            fresnel = 1.0
    else:
        fresnel = 1.0

    eye_dir: np.ndarray = -direction
    for lp, lc in LIGHTS:
        ld: np.ndarray = np.array(lp) - hit_pos
        light_distance: np.floating[Any] = np.linalg.norm(ld)
        ld /= light_distance
        # Shadow check: only geometry strictly between the hit point and the
        # light occludes it (C: 0 < obstacleT0 <= lightDistance).
        obstacle: bool = False
        for s in SPHERES:
            if s is near_sphere:
                continue
            hit = _intersect(
                origin=pos_pos, direction=ld, center=s[0], radius=s[1]
            )
            if hit and not hit[1] and 0 < hit[0] <= light_distance:
                obstacle = True
                break
        if obstacle:
            continue
        diff_i: float = max(0, np.dot(nrm, ld))
        if diff_i > 0:
            color += diff_i * np.array(diff) * np.array(lc)
            r_dir: np.ndarray = ld - 2 * np.dot(ld, nrm) * nrm
            r_dir = -r_dir
            r_dir /= np.linalg.norm(r_dir)
            edr: float = max(0, np.dot(eye_dir, r_dir))
            if edr > 0 and not near_inside:
                spec_i: float = edr**shin
                color += spec_i * np.array(spec) * np.array(lc)

    color += np.array(emiss)
    return (
        (1 - fresnel) * refr_color * (1 - alpha)
        + color * (1 - refl) * alpha
        + fresnel * refl_color * refl
    )


def _render() -> np.ndarray:
    fov: float = 30.0
    aspect: float = WIDTH / HEIGHT
    angle: float = np.tan(np.radians(fov / 2.0))
    pixels: np.ndarray = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        # y=0 = bottom of image so the texture upload aligns with the
        # fullscreen quad's UV.v = 0 at the bottom of the screen.
        ndc_y: float = (2 * (y + 0.5) / HEIGHT - 1) * angle
        for x in range(WIDTH):
            ndc_x: float = (2 * (x + 0.5) / WIDTH - 1) * angle * aspect
            d: np.ndarray = np.array([ndc_x, ndc_y, -1.0])
            d /= np.linalg.norm(d)
            color: np.ndarray = _trace(
                origin=np.array([0.0, 0.0, 0.0]), direction=d
            )
            pixels[y, x] = np.clip(color, a_min=0, a_max=1) * 255
    return pixels


def init() -> bool:
    print("[Example 29] CPU raytracing 640x480; this takes ~30s on first run.")  # noqa: T201
    pixels: np.ndarray = _render()

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen.vert.glsl",
        frag="texture.frag.glsl",
    )
    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)

    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        WIDTH,
        HEIGHT,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        pixels,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)

    state.update(prog=prog, vao=vao, tex=tex)
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)


def update(dt: float) -> bool:
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

    imgui.text(
        f"{WIDTH}x{HEIGHT}, {len(SPHERES)} spheres, max depth {MAX_DEPTH}"
    )
    imgui.text("rendered once on init, displayed as fullscreen texture")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 29 - CPU raytracer",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=WIDTH,
        height=HEIGHT,
    )
