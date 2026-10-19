"""McNopper Example 45: 3D voxelization of an OBJ via geometry shader.

Original: McNopper/OpenGL Example45 (GL 4.4). Rotates the chess king and renders
into a 128^3 R32UI 3D image; the geom stage selects the dominant axis
and writes filled voxels through atomicOr. A fullscreen FS pass walks
the volume per-pixel to display the voxelization.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WINDOW_SIZE: int = 512
VOXEL_SIZE: int = 128
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("ChessKing.obj"))

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="voxelize.vert.glsl",
        geom="voxelize.geom.glsl",
        frag="voxelize.frag.glsl",
    )
    prog_full: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen.vert.glsl",
        frag="draw_voxels.frag.glsl",
    )

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(mesh.positions)
    vbo_t: int = common.make_vbo(mesh.texcoords)
    vbo_i: int = common.make_vbo(
        data=mesh.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_v)
    GL.glVertexAttribPointer(
        index=0,
        size=4,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(0)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo_t)
    GL.glVertexAttribPointer(
        index=1,
        size=2,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(1)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    # The voxel colour is each group's map_Kd (ChessKing.tga, ChessColumn.tga),
    # sampled on unit 1 by voxelize.frag; the C loads them GL_LINEAR, no mips.
    GL.glActiveTexture(GL.GL_TEXTURE1)
    common.upload_material_diffuse_textures(
        materials=mesh.materials, mipmaps=False
    )
    GL.glActiveTexture(GL.GL_TEXTURE0)

    voxel_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, voxel_tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D,
        0,
        GL.GL_R32UI,
        VOXEL_SIZE,
        VOXEL_SIZE,
        VOXEL_SIZE,
        0,
        GL.GL_RED_INTEGER,
        GL.GL_UNSIGNED_INT,
        None,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )

    eye = VOXEL_SIZE * 0.5
    proj: np.ndarray = common.ortho(
        left=-VOXEL_SIZE / 2,
        right=VOXEL_SIZE / 2,
        bottom=-VOXEL_SIZE / 2,
        top=VOXEL_SIZE / 2,
        near=0.0,
        far=VOXEL_SIZE,
    )
    view: np.ndarray = common.look_at(
        eye=[0, 0, eye], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_projectionMatrix"), m=proj
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )
    GL.glUniform2f(
        common.uniform(prog=prog, name="u_halfPixelSize"),
        1.0 / VOXEL_SIZE,
        1.0 / VOXEL_SIZE,
    )

    GL.glUseProgram(prog_full)
    vao_full: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_full)

    GL.glClearColor(0.2, 0.2, 0.2, 1)

    state.update(
        prog=prog,
        prog_full=prog_full,
        vao=vao,
        vao_full=vao_full,
        groups=mesh.groups,
        voxel_tex=voxel_tex,
        model_loc=common.uniform(prog=prog, name="u_modelMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 10.0 * dt) % 360.0
    GL.glClearTexImage(
        state["voxel_tex"],
        0,
        GL.GL_RED_INTEGER,
        GL.GL_UNSIGNED_INT,
        np.zeros(1, dtype=np.uint32),
    )
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)

    GL.glColorMask(False, False, False, False)
    GL.glUseProgram(state["prog"])
    model = (
        common.translate(x=0, y=-200.0 * VOXEL_SIZE / WINDOW_SIZE, z=0)
        @ common.rotate_y(ui["angle"])
        @ common.scale_m(
            x=3000.0 * VOXEL_SIZE / WINDOW_SIZE,
            y=3000.0 * VOXEL_SIZE / WINDOW_SIZE,
            z=3000.0 * VOXEL_SIZE / WINDOW_SIZE,
        )
    )
    common.upload_matrix(loc=state["model_loc"], m=model)
    GL.glBindImageTexture(
        0, state["voxel_tex"], 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_R32UI
    )
    GL.glViewport(0, 0, VOXEL_SIZE, VOXEL_SIZE)
    GL.glBindVertexArray(state["vao"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    for group in state["groups"]:
        GL.glBindTexture(GL.GL_TEXTURE_2D, group.material.diffuse_texture_name)
        common.draw_obj_group(group)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glMemoryBarrier(GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
    GL.glBindImageTexture(0, 0, 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_R32UI)

    GL.glColorMask(True, True, True, True)
    GL.glUseProgram(state["prog_full"])
    GL.glBindImageTexture(
        0, state["voxel_tex"], 0, GL.GL_TRUE, 0, GL.GL_READ_ONLY, GL.GL_R32UI
    )
    GL.glViewport(
        0, 0, state.get("w", WINDOW_SIZE), state.get("h", WINDOW_SIZE)
    )
    GL.glBindVertexArray(state["vao_full"])
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
    imgui.text(f"voxel grid: {VOXEL_SIZE}^3")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 45 - voxelization",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=4,
        width=WINDOW_SIZE,
        height=WINDOW_SIZE,
    )
