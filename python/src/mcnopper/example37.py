"""McNopper Example 37: CPU-side raymarcher of distance fields.

Original: McNopper/OpenGL Example37 (Iquilezles distance functions). Ray marches
3 spheres + 2 oriented boxes; full Phong shading + shadow rays. Runs
once at init and uploads to a texture sampled by a fullscreen quad.

Same resolution as the C source (640x480). The C marches one pixel at
a time; the port marches every ray of the frame together in numpy (same
steps, same tests, same shading) so the once-at-init render takes
seconds. The window then sits showing the result.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WIDTH: int = 640
HEIGHT: int = 480
MAX_DIST: float = 50.0
MAX_STEPS: int = 256
EPSILON: float = 0.01
GAMMA: float = 0.001

Vec3: TypeAlias = tuple[float, float, float]
Material: TypeAlias = tuple[
    Vec3, Vec3, Vec3, float
]  # emissive, diffuse, specular, shine

# The C's g_allSpheres: (center, radius).
SPHERES: list[tuple[Vec3, float]] = [
    ((2.0, 1.0, -14.0), 2.0),
    ((-2.0, 0.25, -6.0), 1.25),
    ((3.0, 0.0, -8.0), 1.0),
]
# The C's g_allBoxes: (center, half extents, orientation in degrees x/y/z).
BOXES: list[tuple[Vec3, Vec3, Vec3]] = [
    ((0.0, -2.0, -10.0), (10.0, 1.0, 20.0), (0.0, 0.0, 0.0)),
    ((-1.0, -0.8, -10.0), (0.5, 0.2, 1.0), (0.0, 20.0, 0.0)),
]
# g_allPrimitives' materials, spheres first then boxes.
MATERIALS: list[Material] = [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.8), (0.8, 0.8, 0.8), 20.0),
    ((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.8, 0.8, 0.8), 20.0),
    ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0), (0.8, 0.8, 0.8), 20.0),
    ((0.0, 0.0, 0.0), (0.8, 0.8, 0.8), (0.8, 0.8, 0.8), 20.0),
    ((0.0, 0.0, 0.0), (0.0, 0.8, 0.8), (0.8, 0.8, 0.8), 20.0),
]
LIGHTS: list[tuple[Vec3, Vec3]] = [((0.0, 5.0, -5.0), (1.0, 1.0, 1.0))]
BACKGROUND: Vec3 = (0.8, 0.8, 0.8)
NUM_PRIMITIVES: int = len(SPHERES) + len(BOXES)
state: dict = {}


def _box_rotation(orientation: Vec3) -> np.ndarray:
    """``glusMatrix3x3RotateRzRyRxf``: Rz * Ry * Rx of the box's angles."""
    ox, oy, oz = orientation
    m: np.ndarray = (
        common.rotate_z(oz) @ common.rotate_y(oy) @ common.rotate_x(ox)
    )
    return m[:3, :3]


def _distance(k: int, p: np.ndarray) -> np.ndarray:
    """Signed distance from the points ``p`` (N,3) to primitive ``k``.

    Spheres are ``glusSphereDistancePoint4f``; boxes are
    ``glusOrientedBoxDistancePoint4f``: the point is moved into the box's
    frame (the transposed rotation), then inside distance
    ``min(max(d), 0)`` plus outside distance ``|max(d, 0)|``.
    """
    if k < len(SPHERES):
        center, radius = SPHERES[k]
        return np.linalg.norm(p - np.array(center), axis=1) - radius
    box_center, half, orientation = BOXES[k - len(SPHERES)]
    v: np.ndarray = (p - np.array(box_center)) @ _box_rotation(orientation)
    v = np.abs(v) - np.array(half)
    inside: np.ndarray = np.minimum(v.max(axis=1), 0.0)
    outside: np.ndarray = np.linalg.norm(np.maximum(v, 0.0), axis=1)
    return inside + outside


def _scene(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest distance and primitive index for each point of ``p``."""
    d: np.ndarray = np.stack(
        [_distance(k=k, p=p) for k in range(NUM_PRIMITIVES)], axis=1
    )
    return d.min(axis=1), d.argmin(axis=1)


def _march(
    origin: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The C ``march`` loop for N rays at once (same steps, same tests).

    Returns the hit positions (N,3) and the hit primitive per ray, -1 where
    the ray reached ``MAX_DISTANCE`` or ran out of steps.
    """
    n: int = len(origin)
    t: np.ndarray = np.zeros(n, dtype=np.float32)
    hit: np.ndarray = np.full(n, -1, dtype=np.int64)
    hit_pos: np.ndarray = np.zeros((n, 3), dtype=np.float32)
    active: np.ndarray = np.ones(n, dtype=bool)
    for _ in range(MAX_STEPS):
        idx: np.ndarray = np.nonzero(active)[0]
        if len(idx) == 0:
            break
        p: np.ndarray = origin[idx] + direction[idx] * t[idx, None]
        d, k = _scene(p)
        got: np.ndarray = d < EPSILON
        hit[idx[got]] = k[got]
        hit_pos[idx[got]] = p[got]
        done: np.ndarray = got | (t[idx] > MAX_DIST)
        active[idx[done]] = False
        t[idx[~done]] += d[~done]
    return hit_pos, hit


def _normals(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Central differences of each hit's own primitive, as the C does."""
    nrm: np.ndarray = np.zeros_like(p)
    for prim in range(NUM_PRIMITIVES):
        sel: np.ndarray = np.nonzero(k == prim)[0]
        if len(sel) == 0:
            continue
        q: np.ndarray = p[sel]
        for axis in range(3):
            e: np.ndarray = np.zeros(3, dtype=np.float32)
            e[axis] = GAMMA
            nrm[sel, axis] = _distance(k=prim, p=q + e) - _distance(
                k=prim, p=q - e
            )
    length: np.ndarray = np.linalg.norm(nrm, axis=1, keepdims=True)
    return nrm / np.maximum(length, 1e-12)


def _shade(
    direction: np.ndarray, hit_pos: np.ndarray, hit: np.ndarray
) -> np.ndarray:
    """The C ``march``'s lighting for the rays that hit something."""
    color: np.ndarray = np.zeros((len(hit), 3), dtype=np.float32)
    nrm: np.ndarray = _normals(p=hit_pos, k=hit)
    eye_dir: np.ndarray = -direction
    diffuse: np.ndarray = np.array([MATERIALS[k][1] for k in hit])
    specular: np.ndarray = np.array([MATERIALS[k][2] for k in hit])
    shininess: np.ndarray = np.array([MATERIALS[k][3] for k in hit])
    for light_pos, light_color in LIGHTS:
        light_dir: np.ndarray = np.array(light_pos) - hit_pos
        light_dir /= np.linalg.norm(light_dir, axis=1, keepdims=True)
        # The C marches from the light back towards the hit: the surface is
        # lit when the first thing that ray meets is the hit's own primitive.
        origin: np.ndarray = np.broadcast_to(
            np.array(light_pos, dtype=np.float32), hit_pos.shape
        ).copy()
        _, shadow_hit = _march(origin=origin, direction=-light_dir)
        obstacle: np.ndarray = (shadow_hit != -1) & (shadow_hit != hit)
        diffuse_i: np.ndarray = np.maximum(
            0.0, np.einsum("ij,ij->i", nrm, light_dir)
        )
        lit: np.ndarray = (~obstacle) & (diffuse_i > 0.0)
        color[lit] += (
            diffuse_i[lit, None] * diffuse[lit] * np.array(light_color)
        )
        incident: np.ndarray = -light_dir
        reflection: np.ndarray = (
            incident - 2.0 * np.einsum("ij,ij->i", incident, nrm)[:, None] * nrm
        )
        reflection /= np.maximum(
            np.linalg.norm(reflection, axis=1, keepdims=True), 1e-12
        )
        e_dot_r: np.ndarray = np.maximum(
            0.0, np.einsum("ij,ij->i", eye_dir, reflection)
        )
        spec: np.ndarray = lit & (e_dot_r > 0.0)
        color[spec] += (
            (e_dot_r[spec] ** shininess[spec])[:, None]
            * specular[spec]
            * np.array(light_color)
        )
    color += np.array([MATERIALS[k][0] for k in hit])
    return color


def _render() -> np.ndarray:
    """``renderToPixelBuffer``: ``glusRaytracePerspectivef`` rays from the
    origin looking down -Z, marched and shaded per pixel. Row 0 is the
    bottom row, as the C uploads it."""
    fovy: float = 30.0
    aspect: float = WIDTH / HEIGHT
    y_extend: float = float(np.tan(np.radians(fovy * 0.5)))
    x_extend: float = y_extend * aspect
    x_step: float = x_extend / (WIDTH * 0.5)
    y_step: float = y_extend / (HEIGHT * 0.5)
    i: np.ndarray = np.arange(WIDTH * HEIGHT)
    direction: np.ndarray = np.stack(
        [
            -x_extend + x_step * 0.5 + x_step * (i % WIDTH),
            -y_extend + y_step * 0.5 + y_step * (i // WIDTH),
            np.full(len(i), -1.0),
        ],
        axis=1,
    ).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    origin: np.ndarray = np.zeros_like(direction)

    hit_pos, hit = _march(origin=origin, direction=direction)
    color: np.ndarray = np.tile(
        np.array(BACKGROUND, dtype=np.float32), (len(i), 1)
    )
    got: np.ndarray = hit != -1
    color[got] = _shade(
        direction=direction[got], hit_pos=hit_pos[got], hit=hit[got]
    )
    pixels: np.ndarray = (np.minimum(1.0, color) * 255.0).astype(np.uint8)
    return pixels.reshape(HEIGHT, WIDTH, 3)


def init() -> bool:
    print(f"[Example 37] CPU raymarching {WIDTH}x{HEIGHT}; please wait...")  # noqa: T201
    px: np.ndarray = _render()

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
        px,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
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

    imgui.text(f"{WIDTH}x{HEIGHT}")
    imgui.text(
        f"{len(SPHERES)} spheres + {len(BOXES)} boxes, max {MAX_STEPS} steps"
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 37 - CPU raymarcher",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
