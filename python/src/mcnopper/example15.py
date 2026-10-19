"""McNopper Example 15: Animated Gerstner-wave water surface.

Original: McNopper/OpenGL Example15. A flat 128x128 water grid is displaced in the
vertex shader by a sum of 4 Gerstner waves; a normal-map texture is
re-rendered every frame from the same wave equations and used to perturb
the cubemap reflection in the water shader. Background is an inside-out
cubemap sphere.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

NUMBERWAVES: int = 4
WATER_PLANE_LENGTH: int = 128
TEXTURE_SIZE: int = 1024

state: dict = {}


def _make_water_grid() -> tuple[np.ndarray, np.ndarray]:
    n = WATER_PLANE_LENGTH
    pts: np.ndarray = np.zeros((n * n, 4), dtype=np.float32)
    for z in range(n):
        for x in range(n):
            pts[x + z * n] = (-n / 2 + 0.5 + x, 0.0, n / 2 - 0.5 - z, 1.0)
    indices: np.ndarray = np.zeros(n * (n - 1) * 2, dtype=np.uint32)
    for k in range(n - 1):
        for i in range(n):
            base = (i + k * n) * 2
            if k % 2 == 0:
                indices[base] = i + (k + 1) * n
                indices[base + 1] = i + k * n
            else:
                indices[base] = n - 1 - i + k * n
                indices[base + 1] = n - 1 - i + (k + 1) * n
    return pts, indices


def init() -> bool:
    pts, indices = _make_water_grid()

    # Programs
    prog_water: int = common.compile_program_files(
        script_file=__file__, vert="Water.vert.glsl", frag="Water.frag.glsl"
    )
    prog_bg: int = common.compile_program_files(
        script_file=__file__,
        vert="Background.vert.glsl",
        frag="Background.frag.glsl",
    )
    prog_wt: int = common.compile_program_files(
        script_file=__file__,
        vert="WaterTexture.vert.glsl",
        frag="WaterTexture.frag.glsl",
    )

    # Cubemap
    cubemap: int = common.upload_cubemap_tgas(
        [
            common.find_media("water_pos_x.tga"),
            common.find_media("water_neg_x.tga"),
            common.find_media("water_pos_y.tga"),
            common.find_media("water_neg_y.tga"),
            common.find_media("water_pos_z.tga"),
            common.find_media("water_neg_z.tga"),
        ]
    )

    # Mirror render-target for normal map
    mirror_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, mirror_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGB8,
        TEXTURE_SIZE,
        TEXTURE_SIZE,
        0,
        GL.GL_RGB,
        GL.GL_UNSIGNED_BYTE,
        None,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

    depth_rb: int = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
    GL.glRenderbufferStorage(
        GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, TEXTURE_SIZE, TEXTURE_SIZE
    )

    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        mirror_tex,
        0,
    )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth_rb
    )
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # Water VAO
    GL.glUseProgram(prog_water)
    GL.glUniform1f(
        common.uniform(prog=prog_water, name="u_waterPlaneLength"),
        float(WATER_PLANE_LENGTH),
    )
    GL.glUniform1i(common.uniform(prog=prog_water, name="u_cubemap"), 0)
    GL.glUniform1i(common.uniform(prog=prog_water, name="u_waterTexture"), 1)

    vao_water: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_water)
    vbo_w: int = common.make_vbo(pts)
    vbo_wi: int = common.make_vbo(
        data=indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_water, vao=vao_water, vbo=vbo_w, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_wi)

    # Background VAO
    bg_sphere: common.Shape = common.make_sphere(
        radius=WATER_PLANE_LENGTH / 2.0 + 0.5, slices=32
    )
    GL.glUseProgram(prog_bg)
    GL.glUniform1i(common.uniform(prog=prog_bg, name="u_cubemap"), 0)
    vao_bg: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_bg)
    vbo_bv: int = common.make_vbo(bg_sphere.vertices)
    vbo_bn: int = common.make_vbo(bg_sphere.normals)
    vbo_bi: int = common.make_vbo(
        data=bg_sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_bg, vao=vao_bg, vbo=vbo_bv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_bg, vao=vao_bg, vbo=vbo_bn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_bi)

    # Water-texture pass plane
    plane: common.Shape = common.make_plane(TEXTURE_SIZE / 2.0)
    GL.glUseProgram(prog_wt)
    GL.glUniform1f(
        common.uniform(prog=prog_wt, name="u_waterPlaneLength"),
        float(WATER_PLANE_LENGTH),
    )
    view_wt: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    proj_wt: np.ndarray = common.ortho(
        left=-TEXTURE_SIZE / 2,
        right=TEXTURE_SIZE / 2,
        bottom=-TEXTURE_SIZE / 2,
        top=TEXTURE_SIZE / 2,
        near=1.0,
        far=100.0,
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog_wt, name="u_modelViewMatrix"), m=view_wt
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog_wt, name="u_projectionMatrix"), m=proj_wt
    )
    vao_wt: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_wt)
    vbo_wtv: int = common.make_vbo(plane.vertices)
    vbo_wtt: int = common.make_vbo(plane.texcoords)
    vbo_wti: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_wt, vao=vao_wt, vbo=vbo_wtv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_wt, vao=vao_wt, vbo=vbo_wtt, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_wti)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog_water=prog_water,
        prog_bg=prog_bg,
        prog_wt=prog_wt,
        vao_water=vao_water,
        vao_bg=vao_bg,
        vao_wt=vao_wt,
        n_water=len(indices),
        n_bg=len(bg_sphere.indices),
        n_wt=len(plane.indices),
        cubemap=cubemap,
        mirror_tex=mirror_tex,
        fbo=fbo,
        passed_time=0.0,
        angle=0.0,
        proj_loc=common.uniform(prog=prog_water, name="u_projectionMatrix"),
        view_loc=common.uniform(prog=prog_water, name="u_viewMatrix"),
        inv_view_n_loc=common.uniform(
            prog=prog_water, name="u_inverseViewNormalMatrix"
        ),
        time_loc=common.uniform(prog=prog_water, name="u_passedTime"),
        wp_loc=common.uniform(prog=prog_water, name="u_waveParameters"),
        wd_loc=common.uniform(prog=prog_water, name="u_waveDirections"),
        proj_loc_bg=common.uniform(prog=prog_bg, name="u_projectionMatrix"),
        mv_loc_bg=common.uniform(prog=prog_bg, name="u_modelViewMatrix"),
        time_loc_wt=common.uniform(prog=prog_wt, name="u_passedTime"),
        wp_loc_wt=common.uniform(prog=prog_wt, name="u_waveParameters"),
        wd_loc_wt=common.uniform(prog=prog_wt, name="u_waveDirections"),
    )
    return True


def _waves_water() -> tuple[np.ndarray, np.ndarray]:
    overall: float = 0.2
    params: np.ndarray = np.zeros((NUMBERWAVES, 4), dtype=np.float32)
    dirs: np.ndarray = np.zeros((NUMBERWAVES, 2), dtype=np.float32)
    cfg = [
        (1.0, 0.01, 4.0, +1.0, +1.0),
        (0.5, 0.02, 3.0, +1.0, 0.0),
        (0.1, 0.015, 2.0, -0.1, -0.2),
        (1.1, 0.008, 1.0, -0.2, -0.1),
    ]
    for i, (sp, am, wl, dx, dz) in enumerate(cfg):
        params[i] = (sp, am, wl, overall / (wl * am * NUMBERWAVES))
        dirs[i] = (dx, dz)
    return params, dirs


def _waves_texture() -> tuple[np.ndarray, np.ndarray]:
    overall = 0.2
    params: np.ndarray = np.zeros((NUMBERWAVES, 4), dtype=np.float32)
    dirs: np.ndarray = np.zeros((NUMBERWAVES, 2), dtype=np.float32)
    cfg = [
        (0.05, 0.02, 0.3, +1.0, +1.5),
        (0.1, 0.01, 0.4, +0.8, +0.2),
        (0.04, 0.035, 0.1, -0.2, -0.1),
        (0.05, 0.007, 0.2, -0.4, -0.3),
    ]
    for i, (sp, am, wl, dx, dz) in enumerate(cfg):
        params[i] = (sp, am, wl, overall / (wl * am * NUMBERWAVES))
        dirs[i] = (dx, dz)
    return params, dirs


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=1000.0
    )
    GL.glUseProgram(state["prog_water"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["proj_loc_bg"], m=proj)


def update(dt: float) -> bool:
    state["passed_time"] += dt
    state["angle"] += 2.0 * math.pi / 120.0 * dt
    a = state["angle"]
    view: np.ndarray = common.look_at(
        eye=[0, 1, 0],
        center=[0.5 * math.sin(a), 1.0, -0.5 * math.cos(a)],
        up=[0, 1, 0],
    )
    inv_view: np.ndarray = np.linalg.inv(view).astype(np.float32)
    inv_view_n: np.ndarray = inv_view[:3, :3]

    # Pass 1: render normal-map texture
    GL.glViewport(0, 0, TEXTURE_SIZE, TEXTURE_SIZE)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog_wt"])
    p, d = _waves_texture()
    GL.glUniform1f(state["time_loc_wt"], state["passed_time"])
    GL.glUniform4fv(state["wp_loc_wt"], NUMBERWAVES, p)
    GL.glUniform2fv(state["wd_loc_wt"], NUMBERWAVES, d)
    GL.glFrontFace(GL.GL_CCW)
    GL.glBindVertexArray(state["vao_wt"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_wt"], GL.GL_UNSIGNED_INT, None)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, state.get("w", 1), state.get("h", 1))

    # Pass 2: background sphere
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["cubemap"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["mirror_tex"])

    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["mv_loc_bg"], m=view)
    GL.glBindVertexArray(state["vao_bg"])
    # Seen from inside, the GLUS sphere's triangles wind clockwise: the C
    # keeps culling on and flips the front face for this draw.
    GL.glFrontFace(GL.GL_CW)
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_bg"], GL.GL_UNSIGNED_INT, None)

    # Pass 3: water surface
    GL.glUseProgram(state["prog_water"])
    common.upload_matrix(loc=state["view_loc"], m=view)
    common.upload_matrix3(loc=state["inv_view_n_loc"], m=inv_view_n)
    GL.glUniform1f(state["time_loc"], state["passed_time"])
    p, d = _waves_water()
    GL.glUniform4fv(state["wp_loc"], NUMBERWAVES, p)
    GL.glUniform2fv(state["wd_loc"], NUMBERWAVES, d)
    GL.glFrontFace(GL.GL_CCW)
    GL.glBindVertexArray(state["vao_water"])
    GL.glDrawElements(
        GL.GL_TRIANGLE_STRIP, state["n_water"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(f"water grid: {WATER_PLANE_LENGTH}x{WATER_PLANE_LENGTH}")
    imgui.text(f"normal-map RTT: {TEXTURE_SIZE}x{TEXTURE_SIZE}")
    imgui.text(f"t = {state['passed_time']:.2f}s")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 15 - Gerstner waves",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
