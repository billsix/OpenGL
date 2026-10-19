"""McNopper Example 31: Deferred shading with 16 instanced point lights.

Original: McNopper/OpenGL Example31. G-buffer pass writes diffuse / specular /
position / normal into 4 MRT attachments while drawing 16x16 instanced
chess pawns. The lighting pass renders one sphere per point-light using
glDrawElementsInstanced and accumulates additive contributions.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL
from OpenGL.constant import Constant

TEXTURE_WIDTH: int = 1024
TEXTURE_HEIGHT: int = 768
OBJECTS_COLUMNS: int = 16
POINT_LIGHT_COUNT: int = 16
POINT_LIGHT_RADIUS: float = 2.0
state: dict = {}


def _make_attachment(
    w: int,
    h: int,
    internal: int | Constant = GL.GL_RGBA8,
    fmt: int | Constant = GL.GL_RGBA,
    type_: int | Constant = GL.GL_UNSIGNED_BYTE,
) -> int:
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, internal, w, h, 0, fmt, type_, None)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
    )
    return tex


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("ChessPawn.obj"))

    light_sphere: common.Shape = common.make_sphere(
        radius=POINT_LIGHT_RADIUS, slices=32
    )

    prog_ds: int = common.compile_program_files(
        script_file=__file__,
        vert="deferred_shading.vert.glsl",
        frag="deferred_shading.frag.glsl",
    )
    prog_pl: int = common.compile_program_files(
        script_file=__file__,
        vert="point_light.vert.glsl",
        frag="point_light.frag.glsl",
    )

    diff_tex: int = _make_attachment(w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT)
    spec_tex: int = _make_attachment(w=TEXTURE_WIDTH, h=TEXTURE_HEIGHT)
    pos_tex: int = _make_attachment(
        w=TEXTURE_WIDTH,
        h=TEXTURE_HEIGHT,
        internal=GL.GL_RGBA32F,
        fmt=GL.GL_RGBA,
        type_=GL.GL_FLOAT,
    )
    norm_tex: int = _make_attachment(
        w=TEXTURE_WIDTH,
        h=TEXTURE_HEIGHT,
        internal=GL.GL_RGB16F,
        fmt=GL.GL_RGB,
        type_=GL.GL_FLOAT,
    )
    depth_tex: int = _make_attachment(
        w=TEXTURE_WIDTH,
        h=TEXTURE_HEIGHT,
        internal=GL.GL_DEPTH_COMPONENT24,
        fmt=GL.GL_DEPTH_COMPONENT,
        type_=GL.GL_UNSIGNED_INT,
    )
    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D,
        diff_tex,
        0,
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT1,
        GL.GL_TEXTURE_2D,
        spec_tex,
        0,
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT2, GL.GL_TEXTURE_2D, pos_tex, 0
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT3,
        GL.GL_TEXTURE_2D,
        norm_tex,
        0,
    )
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_TEXTURE_2D,
        depth_tex,
        0,
    )
    # The G-buffer has to be drawn-to with all 4 attachments active, so call
    # glDrawBuffers here too — completeness is queried with the *current*
    # draw-buffer state on some drivers.
    GL.glDrawBuffers(
        n=4,
        bufs=[
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_COLOR_ATTACHMENT3,
        ],
    )
    status: int = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[example31] G-buffer FBO incomplete: 0x{status:x}")  # noqa: T201
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    GL.glUseProgram(prog_ds)
    vao_ds: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_ds)
    vbo_v: int = common.make_vbo(mesh.positions)
    vbo_n: int = common.make_vbo(mesh.normals)
    vbo_t: int = common.make_vbo(mesh.texcoords)
    vbo_i: int = common.make_vbo(
        data=mesh.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_ds, vao=vao_ds, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_ds, vao=vao_ds, vbo=vbo_n, name="a_normal", size=3
    )
    common.attach_attrib(
        prog=prog_ds, vao=vao_ds, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    GL.glUniform1i(
        common.uniform(prog=prog_ds, name="u_material.diffuseTexture"), 0
    )
    # ChessPawn.obj.mtl: ChessPawn.tga on the pawn, ChessColumn.tga on its base.
    common.upload_material_diffuse_textures(mesh.materials)

    GL.glUseProgram(prog_pl)
    vao_pl: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_pl)
    vbo_lv: int = common.make_vbo(light_sphere.vertices)
    vbo_li: int = common.make_vbo(
        data=light_sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_pl, vao=vao_pl, vbo=vbo_lv, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_li)
    GL.glUniform1f(
        common.uniform(prog=prog_pl, name="u_lightRadius"), POINT_LIGHT_RADIUS
    )
    # Bind G-buffer samplers to texture units 0-3. Without this they all
    # default to unit 0 and read the same attachment, killing the lighting.
    GL.glUniform1i(common.uniform(prog=prog_pl, name="u_diffuse"), 0)
    GL.glUniform1i(common.uniform(prog=prog_pl, name="u_specular"), 1)
    GL.glUniform1i(common.uniform(prog=prog_pl, name="u_position"), 2)
    GL.glUniform1i(common.uniform(prog=prog_pl, name="u_normal"), 3)
    bias: np.ndarray = np.array(
        [
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog_pl, name="u_biasMatrix"), m=bias
    )

    rng: np.random.Generator = np.random.default_rng(13)
    pos_m: np.ndarray = rng.uniform(
        0, POINT_LIGHT_COUNT - 2.0, POINT_LIGHT_COUNT
    ).astype(np.float32)
    dir_m: np.ndarray = (rng.integers(0, 2, POINT_LIGHT_COUNT) * 2 - 1).astype(
        np.float32
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)

    state.update(
        prog_ds=prog_ds,
        prog_pl=prog_pl,
        vao_ds=vao_ds,
        vao_pl=vao_pl,
        groups=mesh.groups,
        diffuse_loc_ds=common.uniform(
            prog=prog_ds, name="u_material.diffuseColor"
        ),
        specular_loc_ds=common.uniform(
            prog=prog_ds, name="u_material.specularColor"
        ),
        exponent_loc_ds=common.uniform(
            prog=prog_ds, name="u_material.specularExponent"
        ),
        use_texture_loc_ds=common.uniform(prog=prog_ds, name="u_useTexture"),
        n_light=len(light_sphere.indices),
        diff_tex=diff_tex,
        spec_tex=spec_tex,
        pos_tex=pos_tex,
        norm_tex=norm_tex,
        fbo=fbo,
        pos_m=pos_m,
        dir_m=dir_m,
        angle=0.0,
        proj_loc_ds=common.uniform(prog=prog_ds, name="u_projectionMatrix"),
        view_loc_ds=common.uniform(prog=prog_ds, name="u_viewMatrix"),
        model_loc_ds=common.uniform(prog=prog_ds, name="u_modelMatrix"),
        nm_loc_ds=common.uniform(prog=prog_ds, name="u_normalMatrix"),
        proj_loc_pl=common.uniform(prog=prog_pl, name="u_projectionMatrix"),
        view_loc_pl=common.uniform(prog=prog_pl, name="u_viewMatrix"),
        model_loc_pl=common.uniform(prog=prog_pl, name="u_modelMatrix"),
        pos_m_loc=common.uniform(prog=prog_pl, name="u_positionMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    w, h = state.get("w", 1280), state.get("h", 720)
    # Use window aspect for BOTH passes. The bias-matrix screen-space sampling
    # only requires that the two passes share an identical projection — the
    # G-buffer texture size is independent of aspect (a 1024x768 texture can
    # equally hold a 16:9 image stretched to fit). Using window aspect lets
    # the lighting pass fill the whole window with no letterbox bars.
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    state["angle"] += 2.0 * math.pi * dt / 20.0
    a: float = state["angle"]
    view: np.ndarray = common.look_at(
        eye=[math.sin(a) * 10, 4, math.cos(a) * 10],
        center=[0, 0, 0],
        up=[0, 1, 0],
    )

    # G-buffer pass
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["fbo"])
    GL.glDrawBuffers(
        n=4,
        bufs=[
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_COLOR_ATTACHMENT1,
            GL.GL_COLOR_ATTACHMENT2,
            GL.GL_COLOR_ATTACHMENT3,
        ],
    )
    GL.glViewport(0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog_ds"])
    common.upload_matrix(loc=state["proj_loc_ds"], m=proj)
    common.upload_matrix(loc=state["view_loc_ds"], m=view)
    model: np.ndarray = common.translate(
        x=-OBJECTS_COLUMNS / 2 + 0.5, y=0, z=OBJECTS_COLUMNS / 2 - 0.5
    ) @ common.scale_m(x=10, y=10, z=10)
    common.upload_matrix(loc=state["model_loc_ds"], m=model)
    common.upload_matrix3(loc=state["nm_loc_ds"], m=model[:3, :3])
    GL.glBindVertexArray(state["vao_ds"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    # C: one instanced draw per group with its .mtl material and diffuse map.
    for group in state["groups"]:
        material: common.ObjMaterial = group.material
        GL.glUniform4f(state["diffuse_loc_ds"], *material.diffuse)
        GL.glUniform4f(state["specular_loc_ds"], *material.specular)
        GL.glUniform1f(state["exponent_loc_ds"], material.shininess)
        if material.diffuse_texture_name:
            GL.glUniform1i(state["use_texture_loc_ds"], 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, material.diffuse_texture_name)
        else:
            GL.glUniform1i(state["use_texture_loc_ds"], 0)
        common.draw_obj_group(
            group=group, instances=OBJECTS_COLUMNS * OBJECTS_COLUMNS
        )

    # Animate point-light positions
    state["pos_m"] += state["dir_m"] * dt * 5.0
    over: np.ndarray = state["pos_m"] > POINT_LIGHT_COUNT - 2.0
    state["pos_m"][over] = (POINT_LIGHT_COUNT - 2.0) - (
        state["pos_m"][over] - (POINT_LIGHT_COUNT - 2.0)
    )
    state["dir_m"][over] *= -1
    under: np.ndarray = state["pos_m"] < 0
    state["pos_m"][under] = -state["pos_m"][under]
    state["dir_m"][under] *= -1

    # Lighting pass — fill the whole window.
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    # Mesa is strict: GL_COLOR_ATTACHMENT0 is invalid for the default
    # framebuffer. Use GL_BACK (the default for double-buffered windows).
    GL.glDrawBuffer(GL.GL_BACK)
    GL.glViewport(0, 0, w, h)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog_pl"])
    common.upload_matrix(loc=state["proj_loc_pl"], m=proj)
    common.upload_matrix(loc=state["view_loc_pl"], m=view)
    light_model: np.ndarray = common.translate(x=-7, y=0, z=7)
    common.upload_matrix(loc=state["model_loc_pl"], m=light_model)
    GL.glUniformMatrix4fv(
        state["pos_m_loc"], 1, GL.GL_FALSE, state["pos_m"].astype(np.float32)
    )

    GL.glEnable(GL.GL_BLEND)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["diff_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["spec_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE2)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["pos_tex"])
    GL.glActiveTexture(GL.GL_TEXTURE3)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["norm_tex"])
    GL.glBindVertexArray(state["vao_pl"])
    GL.glDrawElementsInstanced(
        GL.GL_TRIANGLES,
        state["n_light"],
        GL.GL_UNSIGNED_INT,
        None,
        POINT_LIGHT_COUNT,
    )
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_BLEND)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(f"{OBJECTS_COLUMNS}x{OBJECTS_COLUMNS} instanced pawns")
    imgui.text(f"{POINT_LIGHT_COUNT} additive point lights")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 31 - deferred shading",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        gl_major=4,
        gl_minor=1,
    )
