"""McNopper Example 24: Perlin-noise erosion of a sphere.

Original: McNopper/OpenGL Example24. A 64^3 Perlin-noise volume is uploaded as a
GL_TEXTURE_3D and sampled in the FS to discard fragments whose noise
value is above a slowly-decreasing threshold; the sphere "erodes" away
and loops.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common, perlin
from mcnopper.common import GL

ERODE_START: float = 0.7
ERODE_END: float = -0.1
state: dict = {}
ui: dict = {"angle": 0.0, "erode": ERODE_START, "auto_erode": True}


def init() -> bool:
    sphere: common.Shape = common.make_sphere(radius=1.0, slices=64)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="erode.vert.glsl", frag="erode.frag.glsl"
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(sphere.vertices)
    vbo_n: int = common.make_vbo(sphere.normals)
    vbo_i: int = common.make_vbo(
        data=sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    # C: glusPerlinCreateNoise3D(&image, 64, 64, 64, 0, 8.0f, 192.0f, 0.5f, 4)
    noise: np.ndarray = perlin.create_noise_3d(
        width=64,
        height=64,
        depth=64,
        seed=0,
        frequency=8.0,
        amplitude=192.0,
        persistence=0.5,
        octaves=4,
    )
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_3D,
        0,
        GL.GL_R8,
        64,
        64,
        64,
        0,
        GL.GL_RED,
        GL.GL_UNSIGNED_BYTE,
        noise,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glActiveTexture(GL.GL_TEXTURE0)

    view: np.ndarray = common.look_at(
        eye=[0, 0, 5], center=[0, 0, 0], up=[0, 1, 0]
    )
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    light_view: np.ndarray = view[:3, :3] @ light

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
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.ambientColor"),
        0.75,
        0.75,
        0.75,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        0.75,
        0.75,
        0.75,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.specularColor"),
        1.0,
        1.0,
        1.0,
        1.0,
    )
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_material.specularExponent"), 20.0
    )
    GL.glUniform1i(common.uniform(prog=prog, name="u_texture"), 0)

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    state.update(
        prog=prog,
        vao=vao,
        n=len(sphere.indices),
        tex=tex,
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        erode_loc=common.uniform(prog=prog, name="u_erode"),
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
    ui["angle"] = (ui["angle"] + 45.0 * dt) % 360.0
    if ui["auto_erode"]:
        ui["erode"] -= 0.1 * dt
        if ui["erode"] < ERODE_END:
            ui["erode"] = ERODE_START

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    mv: np.ndarray = state["view"] @ common.rotate_y(ui["angle"])
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glUniform1f(state["erode_loc"], ui["erode"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, state["tex"])
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n"], GL.GL_UNSIGNED_INT, None)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["auto_erode"] = imgui.checkbox(label="auto erode", v=ui["auto_erode"])
    _, ui["erode"] = imgui.slider_float(
        label="erode", v=ui["erode"], v_min=ERODE_END, v_max=ERODE_START
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 24 - perlin erosion",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
