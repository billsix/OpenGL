"""McNopper Example 36: Per-pixel linked-list OIT.

Original: McNopper/OpenGL Example36 (GL 4.3+). Each fragment atomically allocates
a node from a free-list via an atomic counter, links into a per-pixel
chain stored in a head-index image + SSBO, and the fullscreen pass
walks the lists in the FS to composite.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
MAX_NODES: int = 8 * SCREEN_WIDTH * SCREEN_HEIGHT
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("dragon.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="phong_linked_list.vert.glsl",
        frag="phong_linked_list.frag.glsl",
    )
    prog_blend: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen_blend.vert.glsl",
        frag="fullscreen_blend.frag.glsl",
    )

    free_idx: int = GL.glGenBuffers(1)
    GL.glBindBufferBase(GL.GL_ATOMIC_COUNTER_BUFFER, 0, free_idx)
    GL.glBufferData(
        target=GL.GL_ATOMIC_COUNTER_BUFFER,
        size=4,
        data=np.zeros(1, dtype=np.uint32),
        usage=GL.GL_DYNAMIC_DRAW,
    )

    head_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, head_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_R32UI,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        0,
        GL.GL_RED_INTEGER,
        GL.GL_UNSIGNED_INT,
        None,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    GL.glBindImageTexture(
        0, head_tex, 0, GL.GL_FALSE, 0, GL.GL_READ_WRITE, GL.GL_R32UI
    )

    clear_buffer: int = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, clear_buffer)
    clear_data: np.ndarray = np.full(
        SCREEN_WIDTH * SCREEN_HEIGHT, 0xFFFFFFFF, dtype=np.uint32
    )
    GL.glBufferData(
        target=GL.GL_PIXEL_UNPACK_BUFFER,
        size=clear_data.nbytes,
        data=clear_data,
        usage=GL.GL_STATIC_COPY,
    )

    ll_buffer: int = GL.glGenBuffers(1)
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, ll_buffer)
    GL.glBufferData(
        target=GL.GL_SHADER_STORAGE_BUFFER,
        # C: per node RGBA + depth (5 floats), next (1 uint), 2 paddings
        size=MAX_NODES * (4 * 5 + 4 * 1 + 4 * 2),
        data=None,
        usage=GL.GL_DYNAMIC_DRAW,
    )

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.ambientColor"),
        0.3,
        0.3,
        0.3,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.diffuseColor"), 1, 1, 1, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.specularColor"), 1, 1, 1, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.ambientColor"), 0, 1, 0, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"), 0, 1, 0, 1
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.specularColor"), 1, 1, 1, 1
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_material.specularExponent"), 20.0
    )
    GL.glUniform1f(common.uniform(prog=prog, name="u_material.alpha"), 0.5)
    GL.glUniform1ui(common.uniform(prog=prog, name="u_maxNodes"), MAX_NODES)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 3], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_light.direction"), *light_view
    )

    GL.glUseProgram(prog_blend)
    vao_blend: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_blend)

    GL.glDisable(GL.GL_DEPTH_TEST)

    state.update(
        prog=prog,
        prog_blend=prog_blend,
        vao=vao,
        vao_blend=vao_blend,
        count=len(pos),
        view=view,
        free_idx=free_idx,
        head_tex=head_tex,
        clear_buffer=clear_buffer,
        ll_buffer=ll_buffer,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    GL.glUseProgram(state["prog"])
    common.upload_matrix(
        loc=state["proj_loc"],
        m=common.perspective(
            fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
        ),
    )


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0

    GL.glUseProgram(state["prog"])
    GL.glBindVertexArray(state["vao"])
    mv: np.ndarray = state["view"] @ common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])

    GL.glBindBufferBase(GL.GL_ATOMIC_COUNTER_BUFFER, 0, state["free_idx"])
    GL.glBufferSubData(
        target=GL.GL_ATOMIC_COUNTER_BUFFER,
        offset=0,
        size=4,
        data=np.zeros(1, dtype=np.uint32),
    )

    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, state["clear_buffer"])
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["head_tex"])
    GL.glTexSubImage2D(
        GL.GL_TEXTURE_2D,
        0,
        0,
        0,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        GL.GL_RED_INTEGER,
        GL.GL_UNSIGNED_INT,
        None,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])
    GL.glMemoryBarrier(
        GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
    )

    GL.glUseProgram(state["prog_blend"])
    GL.glBindVertexArray(state["vao_blend"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["auto_rotate"] = imgui.checkbox(
        label="auto rotate", v=ui["auto_rotate"]
    )
    _, ui["angle"] = imgui.slider_float(
        label="angle", v=ui["angle"], v_min=0, v_max=360
    )
    imgui.text(f"max nodes: {MAX_NODES:,}")


if __name__ == "__main__":
    # Head-index image and per-pixel SSBO are sized to SCREEN_WIDTH x
    # SCREEN_HEIGHT. The window has to match or fragments past the
    # texture extent write/read out of bounds and you get garbage on
    # the side strips.
    common.run_demo(
        title="McNopper Example 36 - linked-list OIT",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=4,
        width=SCREEN_WIDTH,
        height=SCREEN_HEIGHT,
    )
