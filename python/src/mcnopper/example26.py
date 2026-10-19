"""McNopper Example 26: Geometry-shader fur on the Stanford bunny.

Original: McNopper/OpenGL Example26. The geometry shader emits N small fur shells
per triangle along the normal; each shell samples a Perlin-noise
strength map and the tiger color texture.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common, perlin
from mcnopper.common import GL

state: dict = {}


def _texgen_axes(positions: np.ndarray, s_x: float, t_y: float) -> np.ndarray:
    """``glusShapeTexGenByAxesf(shape, s_x, 0, t_y, 0, 0, 0)``: s = s_x*x,
    t = t_y*y."""
    return np.column_stack(
        [positions[:, 0] * s_x, positions[:, 1] * t_y]
    ).astype(np.float32)


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("bunny.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    uv = _texgen_axes(positions=pos, s_x=2.0, t_y=2.0)

    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="ambient_diffuse_texture.vert.glsl",
        frag="ambient_diffuse_texture.frag.glsl",
    )
    prog_fur: int = common.compile_program_files(
        script_file=__file__,
        vert="fur.vert.glsl",
        geom="fur.geom.glsl",
        frag="fur.frag.glsl",
    )

    color: int = common.upload_tga_2d(common.find_media("tiger.tga"))
    # C: glusPerlinCreateNoise2D(&image, 64, 64, 0, 100.0f, 255.0f, 0.5f, 1)
    strength_data: np.ndarray = perlin.create_noise_2d(
        width=64,
        height=64,
        seed=0,
        frequency=100.0,
        amplitude=255.0,
        persistence=0.5,
        octaves=1,
    )
    strength: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, strength)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_R8,
        64,
        64,
        0,
        GL.GL_RED,
        GL.GL_UNSIGNED_BYTE,
        strength_data,
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(pos)
    vbo_n: int = common.make_vbo(nor)
    vbo_t: int = common.make_vbo(uv)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )

    GL.glUseProgram(prog_fur)
    vao_fur: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_fur)
    common.attach_attrib(
        prog=prog_fur, vao=vao_fur, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_fur, vao=vao_fur, vbo=vbo_n, name="a_normal", size=3
    )
    common.attach_attrib(
        prog=prog_fur, vao=vao_fur, vbo=vbo_t, name="a_texCoord", size=2
    )

    view: np.ndarray = common.look_at(
        eye=[0, 0.75, 3], center=[0, 0.75, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light

    GL.glUseProgram(prog)
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_light.direction"), *light_view
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.ambientColor"),
        0.3,
        0.3,
        0.3,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.diffuseColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_light.specularColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform1i(common.uniform(prog=prog, name="u_textureFurColor"), 0)

    GL.glUseProgram(prog_fur)
    GL.glUniform3f(
        common.uniform(prog=prog_fur, name="u_lightDirection"), *light_view
    )
    GL.glUniform1i(common.uniform(prog=prog_fur, name="u_textureFurColor"), 0)
    GL.glUniform1i(
        common.uniform(prog=prog_fur, name="u_textureFurStrength"), 1
    )

    GL.glClearColor(0.8, 0.8, 0.8, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    state.update(
        prog=prog,
        prog_fur=prog_fur,
        vao=vao,
        vao_fur=vao_fur,
        count=len(pos),
        color=color,
        strength=strength,
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        proj_loc_fur=common.uniform(prog=prog_fur, name="u_projectionMatrix"),
        mv_loc_fur=common.uniform(prog=prog_fur, name="u_modelViewMatrix"),
        nm_loc_fur=common.uniform(prog=prog_fur, name="u_normalMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    GL.glUseProgram(state["prog_fur"])
    common.upload_matrix(loc=state["proj_loc_fur"], m=proj)


def update(dt: float) -> bool:
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["color"])
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["strength"])

    mv: np.ndarray = state["view"]
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])

    GL.glDepthMask(False)
    GL.glEnable(GL.GL_BLEND)
    GL.glUseProgram(state["prog_fur"])
    common.upload_matrix(loc=state["mv_loc_fur"], m=mv)
    common.upload_matrix3(loc=state["nm_loc_fur"], m=mv[:3, :3])
    GL.glBindVertexArray(state["vao_fur"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["count"])
    GL.glDisable(GL.GL_BLEND)
    GL.glDepthMask(True)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text("bunny.obj + tiger.tga + perlin strength map")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 26 - fur",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        samples=8,
    )
