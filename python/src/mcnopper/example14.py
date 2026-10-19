"""McNopper Example 14: LOD terrain with two-pass tessellation + transform feedback.

Original: McNopper/OpenGL Example14 - "Real-Time LOD Terrain Rendering on Modern GPU".
Pass 1 culls/expands a coarse grid via geometry-shader subdivision into
transform-feedback output points (one per visible quad). Pass 2 reads
those points back as patches and runs full hardware tessellation against
the height/color/normal-map rectangle textures.

This port preserves the algorithm and reuses the original GLSL shaders
verbatim. Imgui exposes the wireframe / animation / view toggles that
the C version drove from keyboard.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from OpenGL.constant import Constant

from mcnopper import common
from mcnopper.common import GL

TURN_DURATION: float = 20.0
TURN_RADIUS: float = 6000.0
HORIZONTAL_PIXEL_SPACING: float = 60.0
VERTICAL_PIXEL_RANGE: float = 10004.0
METERS_TO_VIRTUAL_WORLD_SCALE: float = 5.0
MINIMUM_DETAIL_LEVEL: int = 4
DETAIL_LEVEL_FIRST_PASS: int = 2
FOV_RADIUS: float = 10000.0
QUADRANT_STEP: int = 2

state: dict = {}
ui: dict = {
    "wireframe": False,
    "animate": True,
    "top_view": False,
    "angle": 0.0,
}


def _load_rect(path: str) -> tuple[int, int, int]:
    img: common.TgaImage = common.load_tga(path)
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_RECTANGLE, tex)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    fmt: int | Constant = {1: GL.GL_RED, 3: GL.GL_RGB, 4: GL.GL_RGBA}[
        img.channels
    ]
    internal: int | Constant = {1: GL.GL_R8, 3: GL.GL_RGB8, 4: GL.GL_RGBA8}[
        img.channels
    ]
    GL.glTexImage2D(
        GL.GL_TEXTURE_RECTANGLE,
        0,
        internal,
        img.width,
        img.height,
        0,
        fmt,
        GL.GL_UNSIGNED_BYTE,
        img.pixels,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_RECTANGLE, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_RECTANGLE, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_RECTANGLE, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_RECTANGLE, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    return tex, img.width, img.height


def init() -> bool:
    normal_tex, _, _ = _load_rect(common.find_media("grand_canyon_normal.tga"))
    height_tex, sw, sh = _load_rect(
        common.find_media("grand_canyon_height.tga")
    )

    s_max: int = int(math.floor(math.log(sw, 2)))
    t_max: int = int(math.floor(math.log(sh, 2)))
    overall_max: int = min(s_max, t_max)
    detail_step: int = 2 ** (overall_max - MINIMUM_DETAIL_LEVEL)
    s_n: int = int(math.ceil(sw / detail_step)) - 1
    t_n: int = int(math.ceil(sh / detail_step)) - 1

    map_pts: np.ndarray = np.zeros((t_n, s_n, 2), dtype=np.float32)
    for t in range(t_n):
        for s in range(s_n):
            map_pts[t, s] = (
                0.5 + detail_step / 2.0 + s * detail_step,
                0.5 + detail_step / 2.0 + t * detail_step,
            )
    map_pts = map_pts.reshape(-1, 2)
    indices: np.ndarray = np.arange(t_n * s_n, dtype=np.uint32)

    color_tex: int = common.upload_tga_2d(
        filename=common.find_media("grand_canyon_color.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
        wrap=GL.GL_REPEAT,
    )

    prog_one: int = common.compile_program_files(
        script_file=__file__,
        vert="PassOne.vert.glsl",
        geom="PassOne.geom.glsl",
        frag="PassOne.frag.glsl",
        tf_varyings=["v_vertex"],
        tf_mode=GL.GL_SEPARATE_ATTRIBS,
    )
    prog_two: int = common.compile_program_files(
        script_file=__file__,
        vert="PassTwo.vert.glsl",
        tesc="PassTwo.cont.glsl",
        tese="PassTwo.eval.glsl",
        geom="PassTwo.geom.glsl",
        frag="PassTwo.frag.glsl",
    )

    vbo_one: int = common.make_vbo(map_pts)
    vbo_idx: int = common.make_vbo(
        data=indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    # Pass-two scratch buffer (transform feedback target)
    vbo_two: int = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_two)
    capacity: int = s_n * t_n * (4 ** (DETAIL_LEVEL_FIRST_PASS + 1)) * 2
    GL.glBufferData(
        target=GL.GL_ARRAY_BUFFER,
        size=capacity * 4,
        data=None,
        usage=GL.GL_DYNAMIC_DRAW,
    )
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    GL.glUseProgram(prog_one)
    GL.glUniform1f(
        common.uniform(prog=prog_one, name="u_halfDetailStep"),
        detail_step / 2.0,
    )
    GL.glUniform1ui(
        common.uniform(prog=prog_one, name="u_detailLevel"),
        DETAIL_LEVEL_FIRST_PASS,
    )
    GL.glUniform1f(
        common.uniform(prog=prog_one, name="u_fovRadius"),
        FOV_RADIUS / HORIZONTAL_PIXEL_SPACING * METERS_TO_VIRTUAL_WORLD_SCALE,
    )

    vao_one: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_one)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_one)
    GL.glVertexAttribPointer(
        index=0,
        size=2,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(0)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_idx)

    GL.glUseProgram(prog_two)
    light: np.ndarray = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    GL.glUniform3f(
        common.uniform(prog=prog_two, name="u_lightDirection"), *light
    )
    GL.glUniform1ui(
        common.uniform(prog=prog_two, name="u_maxTessellationLevel"),
        overall_max - (MINIMUM_DETAIL_LEVEL + DETAIL_LEVEL_FIRST_PASS),
    )
    GL.glUniform1i(
        common.uniform(prog=prog_two, name="u_quadrantStep"), QUADRANT_STEP
    )
    GL.glUniform1i(common.uniform(prog=prog_two, name="u_heightMapTexture"), 0)
    GL.glUniform1i(common.uniform(prog=prog_two, name="u_colorMapTexture"), 1)
    GL.glUniform1i(common.uniform(prog=prog_two, name="u_normalMapTexture"), 2)

    vao_two: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_two)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_two)
    GL.glVertexAttribPointer(
        index=0,
        size=2,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(0)

    # World <-> texture matrices
    h: float = HORIZONTAL_PIXEL_SPACING * METERS_TO_VIRTUAL_WORLD_SCALE
    v: float = VERTICAL_PIXEL_RANGE * METERS_TO_VIRTUAL_WORLD_SCALE
    tex_to_world: np.ndarray = np.array(
        [
            [h, 0, 0, -sw / 2.0 * h],
            [0, v, 0, 0],
            [0, 0, -h, sh / 2.0 * h],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    world_to_tex: np.ndarray = np.linalg.inv(tex_to_world).astype(np.float32)
    world_to_tex_normal: np.ndarray = np.linalg.inv(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
    ).astype(np.float32)

    # PyOpenGL returns a 1-element array from glGenQueries(1); glBeginQuery
    # wants a scalar int.
    _qarr: Any = GL.glGenQueries(1)
    query: int = int(_qarr[0]) if hasattr(_qarr, "__len__") else int(_qarr)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 4)

    state.update(
        prog_one=prog_one,
        prog_two=prog_two,
        vao_one=vao_one,
        vao_two=vao_two,
        vbo_two=vbo_two,
        s_n=s_n,
        t_n=t_n,
        height_tex=height_tex,
        color_tex=color_tex,
        normal_tex=normal_tex,
        sw=sw,
        sh=sh,
        query=query,
        tex_to_world=tex_to_world,
        world_to_tex=world_to_tex,
        world_to_tex_normal=world_to_tex_normal,
        proj=common.perspective(
            fovy_deg=60.0, aspect=1.0, near=1.0, far=1_000_000.0
        ),
        pos_loc_one=common.uniform(
            prog=prog_one, name="u_positionTextureSpace"
        ),
        ln_loc=common.uniform(prog=prog_one, name="u_leftNormalTextureSpace"),
        rn_loc=common.uniform(prog=prog_one, name="u_rightNormalTextureSpace"),
        bn_loc=common.uniform(prog=prog_one, name="u_backNormalTextureSpace"),
        tmvp_loc=common.uniform(prog=prog_two, name="u_tmvpMatrix"),
        pos_loc_two=common.uniform(
            prog=prog_two, name="u_positionTextureSpace"
        ),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    fov: float = 40.0 if ui["top_view"] else 60.0
    state["proj"] = common.perspective(
        fovy_deg=fov, aspect=w / max(h, 1), near=1.0, far=1_000_000.0
    )


def update(dt: float) -> bool:
    if ui["animate"]:
        ui["angle"] += dt
    a: float = 2.0 * math.pi * ui["angle"] / TURN_DURATION
    cam_pos: np.ndarray = np.array(
        [
            -math.cos(a) * TURN_RADIUS * METERS_TO_VIRTUAL_WORLD_SCALE,
            4700.0 * METERS_TO_VIRTUAL_WORLD_SCALE,
            -math.sin(a) * TURN_RADIUS * METERS_TO_VIRTUAL_WORLD_SCALE,
        ],
        dtype=np.float32,
    )
    cam_dir: np.ndarray = np.array(
        [math.sin(a), 0.0, -math.cos(a)], dtype=np.float32
    )
    cam_up: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # The active view only picks the view matrix. Pass one's culling and
    # pass two's LOD centre always take the PERSON camera (C main.c uses
    # g_personView for them), so the top view shows the travelling wedge.
    view_pos: np.ndarray = cam_pos
    view_dir: np.ndarray = cam_dir
    view_up: np.ndarray = cam_up
    if ui["top_view"]:
        view_pos = np.array(
            [0, 30000.0 * METERS_TO_VIRTUAL_WORLD_SCALE, 0], dtype=np.float32
        )
        view_dir = np.array([0, -1, 0], dtype=np.float32)
        view_up = np.array([0, 0, -1], dtype=np.float32)

    view: np.ndarray = common.look_at(
        eye=view_pos, center=view_pos + view_dir, up=view_up
    )
    tmvp: np.ndarray = state["proj"] @ view @ state["tex_to_world"]

    xz: np.ndarray = np.array(
        [cam_pos[0], 0.0, cam_pos[2], 1.0], dtype=np.float32
    )
    pos_tex: np.ndarray = state["world_to_tex"] @ xz
    # dir_tex: texture-space camera direction, kept for parity; not sampled.
    dir_tex: np.ndarray = (  # noqa: F841
        state["world_to_tex"][:3, :3] @ cam_dir
    )
    fov_h: float = (
        60.0 * (state.get("w", 1) / max(state.get("h", 1), 1))
    ) / 2.0
    rot_l: np.ndarray = common.rotate_y(fov_h + 90.0)
    rot_r: np.ndarray = common.rotate_y(-fov_h - 90.0)
    rot_b: np.ndarray = common.rotate_y(180.0)
    ln: np.ndarray = state["world_to_tex_normal"][:3, :3] @ (
        rot_l[:3, :3] @ cam_dir
    )
    rn: np.ndarray = state["world_to_tex_normal"][:3, :3] @ (
        rot_r[:3, :3] @ cam_dir
    )
    bn: np.ndarray = state["world_to_tex_normal"][:3, :3] @ (
        rot_b[:3, :3] @ cam_dir
    )

    GL.glPolygonMode(
        GL.GL_FRONT_AND_BACK, GL.GL_LINE if ui["wireframe"] else GL.GL_FILL
    )
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    # Pass one: cull + expand into transform-feedback buffer
    GL.glEnable(GL.GL_RASTERIZER_DISCARD)
    GL.glUseProgram(state["prog_one"])
    GL.glUniform4f(state["pos_loc_one"], *pos_tex)
    GL.glUniform3f(state["ln_loc"], *ln)
    GL.glUniform3f(state["rn_loc"], *rn)
    GL.glUniform3f(state["bn_loc"], *bn)
    GL.glBindVertexArray(state["vao_one"])
    GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, state["vbo_two"])
    GL.glBeginQuery(GL.GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, state["query"])
    GL.glBeginTransformFeedback(GL.GL_POINTS)
    GL.glDrawElements(
        GL.GL_POINTS, state["s_n"] * state["t_n"], GL.GL_UNSIGNED_INT, None
    )
    GL.glEndTransformFeedback()
    GL.glEndQuery(GL.GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN)
    GL.glDisable(GL.GL_RASTERIZER_DISCARD)
    GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0)

    # Pass two: tessellate + render
    GL.glUseProgram(state["prog_two"])
    common.upload_matrix(loc=state["tmvp_loc"], m=tmvp)
    GL.glUniform4f(state["pos_loc_two"], *pos_tex)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_RECTANGLE, state["height_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["color_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE2)
    GL.glBindTexture(GL.GL_TEXTURE_RECTANGLE, state["normal_tex"])
    GL.glBindVertexArray(state["vao_two"])
    primitives: int = GL.glGetQueryObjectuiv(state["query"], GL.GL_QUERY_RESULT)
    GL.glDrawArrays(GL.GL_PATCHES, 0, primitives)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["wireframe"] = imgui.checkbox(label="wireframe", v=ui["wireframe"])
    _, ui["animate"] = imgui.checkbox(label="animate", v=ui["animate"])
    changed, ui["top_view"] = imgui.checkbox(label="top view", v=ui["top_view"])
    if changed:
        reshape(w=state.get("w", 1280), h=state.get("h", 720))


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 14 - LOD terrain",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
