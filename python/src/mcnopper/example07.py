"""McNopper Example 07: Tangent-space normal mapping on a plane.

Original: McNopper/OpenGL Example07 - rock_color.tga + rock_normal.tga, animated
directional light. Lighting evaluated in tangent space; the GLUS plane
generator already provides tangent/bitangent/normal vectors.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"animate_light": True, "phase": 0.0}


def init() -> bool:
    plane: common.Shape = common.calculate_tangents(
        common.make_rectangular_plane(half_x=1.5, half_y=1.5)
    )
    prog: int = common.compile_program_files(
        script_file=__file__, vert="normmap.vert.glsl", frag="normmap.frag.glsl"
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    bufs = {
        "v": (common.make_vbo(plane.vertices), 4, "a_vertex"),
        "tan": (common.make_vbo(plane.tangents), 3, "a_tangent"),
        "bit": (common.make_vbo(plane.bitangents), 3, "a_bitangent"),
        "n": (common.make_vbo(plane.normals), 3, "a_normal"),
        "uv": (common.make_vbo(plane.texcoords), 2, "a_texCoord"),
    }
    for vbo, size, name in bufs.values():
        common.attach_attrib(prog=prog, vao=vao, vbo=vbo, name=name, size=size)
    vbo_i: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    tex_color: int = common.upload_tga_2d(
        filename=common.find_media("rock_color.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
    )
    tex_normal: int = common.upload_tga_2d(
        filename=common.find_media("rock_normal.tga"),
        mipmaps=False,
        min_filter=GL.GL_LINEAR,
    )

    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_color)
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex_normal)
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)
    GL.glUniform1i(common.uniform(prog=prog, name="u_normalMap"), 1)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    common.upload_matrix(
        loc=common.uniform(prog=prog, name="u_modelViewMatrix"), m=view
    )
    common.upload_matrix3(
        loc=common.uniform(prog=prog, name="u_normalMatrix"), m=view[:3, :3]
    )

    GL.glClearColor(0, 0, 0, 0)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(plane.indices),
        view=view,
        tex_color=tex_color,
        tex_normal=tex_normal,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        light_loc=common.uniform(prog=prog, name="u_lightDirection"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    if ui["animate_light"]:
        ui["phase"] += dt
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    # Re-bind every frame: imgui's renderer touches the texture units when
    # it draws its panels, so init-time bindings don't survive.
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["tex_color"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["tex_normal"])
    ld: np.ndarray = np.array(
        [2.0 * math.cos(ui["phase"]), 1.0, 1.0], dtype=np.float32
    )
    n = np.linalg.norm(ld)
    if n > 1e-8:
        ld = ld / n
    view_ld: np.ndarray = state["view"][:3, :3] @ ld
    GL.glUniform3f(state["light_loc"], *view_ld)
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["animate_light"] = imgui.checkbox(
        label="animate light", v=ui["animate_light"]
    )
    _, ui["phase"] = imgui.slider_float(
        label="phase", v=ui["phase"], v_min=0, v_max=2 * math.pi
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 07 - normal mapping",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
