"""McNopper Example 23: Screen-space adaptive tessellation + displacement.

Original: McNopper/OpenGL Example23. A 16x16 rectangular grid is tessellated as
triangle patches whose tessellation level depends on screen-space size;
the eval shader displaces each generated vertex along the normal map.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

DISPLACEMENT_SCALE: float = 0.3
state: dict = {}
ui: dict = {
    "angle": 0.0,
    "auto_rotate": True,
    "displacement": True,
    "tessellation": True,
    "wireframe": False,
}


def init() -> bool:
    plane: common.Shape = common.calculate_tangents(
        common.make_rectangular_grid_plane(
            horizontal_extend=3.0, vertical_extend=3.0, rows=16, columns=16
        )
    )
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="tessellation.vert.glsl",
        tesc="tessellation.cont.glsl",
        tese="tessellation.eval.glsl",
        geom="tessellation.geom.glsl",
        frag="tessellation.frag.glsl",
    )
    GL.glUseProgram(prog)

    color: int = common.upload_tga_2d(
        filename=common.find_media("four_shapes_color.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
    )
    normal: int = common.upload_tga_2d(
        filename=common.find_media("four_shapes_normal.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
    )

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    bufs: list[tuple[int, int, str]] = [
        (common.make_vbo(plane.vertices), 4, "a_vertex"),
        (common.make_vbo(plane.tangents), 3, "a_tangent"),
        (common.make_vbo(plane.bitangents), 3, "a_bitangent"),
        (common.make_vbo(plane.normals), 3, "a_normal"),
        (common.make_vbo(plane.texcoords), 2, "a_texCoord"),
    ]
    for vbo, size, name in bufs:
        common.attach_attrib(prog=prog, vao=vao, vbo=vbo, name=name, size=size)
    vbo_i: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    view: np.ndarray = common.look_at(
        eye=[0, 4, 4], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_viewMatrix"), m=view
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_displacementScale"),
        DISPLACEMENT_SCALE,
    )
    GL.glUniform1f(common.uniform(prog=prog, name="u_screenDistance"), 8.0)
    GL.glUniform1i(common.uniform(prog=prog, name="u_doTessellate"), 1)
    GL.glUniform1i(common.uniform(prog=prog, name="u_colorTexture"), 0)
    GL.glUniform1i(common.uniform(prog=prog, name="u_normalTexture"), 1)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, color)
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, normal)

    GL.glClearColor(0, 0, 0, 0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 3)

    state.update(
        prog=prog,
        vao=vao,
        n=len(plane.indices),
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        disp_loc=common.uniform(prog=prog, name="u_displacementScale"),
        do_tess_loc=common.uniform(prog=prog, name="u_doTessellate"),
        w_loc=common.uniform(prog=prog, name="u_width"),
        h_loc=common.uniform(prog=prog, name="u_height"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    GL.glUseProgram(state["prog"])
    GL.glUniform1f(state["w_loc"], float(w))
    GL.glUniform1f(state["h_loc"], float(h))
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=0.1, far=100.0
    )
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 45.0 * dt) % 360.0
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog"])
    GL.glUniform1f(
        state["disp_loc"], DISPLACEMENT_SCALE if ui["displacement"] else 0.0
    )
    GL.glUniform1i(state["do_tess_loc"], int(ui["tessellation"]))
    GL.glPolygonMode(
        GL.GL_FRONT_AND_BACK, GL.GL_LINE if ui["wireframe"] else GL.GL_FILL
    )
    model: np.ndarray = common.rotate_y(ui["angle"]) @ common.rotate_x(-90.0)
    common.upload_matrix(loc=state["m_loc"], m=model)
    common.upload_matrix3(loc=state["nm_loc"], m=model[:3, :3])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(GL.GL_PATCHES, state["n"], GL.GL_UNSIGNED_INT, None)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["auto_rotate"] = imgui.checkbox(
        label="auto rotate", v=ui["auto_rotate"]
    )
    _, ui["displacement"] = imgui.checkbox(
        label="displacement", v=ui["displacement"]
    )
    _, ui["tessellation"] = imgui.checkbox(
        label="tessellation", v=ui["tessellation"]
    )
    _, ui["wireframe"] = imgui.checkbox(label="wireframe", v=ui["wireframe"])


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 23 - adaptive tessellation",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        gl_major=4,
        gl_minor=1,
    )
