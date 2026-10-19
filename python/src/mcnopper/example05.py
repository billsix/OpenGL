"""McNopper Example 05: Phong-lit sphere.

Original: McNopper/OpenGL Example05 - blue sphere, white ambient/diffuse/specular
directional light evaluated in view space. Port keeps the C struct
layout (LightProperties / MaterialProperties) as plain dataclasses and
exposes them through an imgui panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcnopper import common
from mcnopper.common import GL


@dataclass
class LightProperties:
    direction: list
    ambient: list
    diffuse: list
    specular: list


@dataclass
class MaterialProperties:
    ambient: list
    diffuse: list
    specular: list
    exponent: float


light = LightProperties(
    direction=[1.0, 1.0, 1.0],
    ambient=[0.3, 0.3, 0.3, 1.0],
    diffuse=[1.0, 1.0, 1.0, 1.0],
    specular=[1.0, 1.0, 1.0, 1.0],
)
material = MaterialProperties(
    ambient=[0.0, 0.0, 1.0, 1.0],
    diffuse=[0.0, 0.0, 1.0, 1.0],
    specular=[1.0, 1.0, 1.0, 1.0],
    exponent=20.0,
)
state: dict = {}


def init() -> bool:
    sphere: common.Shape = common.make_sphere(radius=0.5, slices=32)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
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
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        num_indices=len(sphere.indices),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        light_dir_loc=common.uniform(prog=prog, name="u_light.direction"),
        light_amb_loc=common.uniform(prog=prog, name="u_light.ambientColor"),
        light_diff_loc=common.uniform(prog=prog, name="u_light.diffuseColor"),
        light_spec_loc=common.uniform(prog=prog, name="u_light.specularColor"),
        mat_amb_loc=common.uniform(prog=prog, name="u_material.ambientColor"),
        mat_diff_loc=common.uniform(prog=prog, name="u_material.diffuseColor"),
        mat_spec_loc=common.uniform(prog=prog, name="u_material.specularColor"),
        mat_exp_loc=common.uniform(
            prog=prog, name="u_material.specularExponent"
        ),
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
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])

    ld: np.ndarray = np.asarray(light.direction, dtype=np.float32)
    n = np.linalg.norm(ld)
    if n > 1e-8:
        ld = ld / n
    # transform light direction to view space (rotation only)
    view_ld: np.ndarray = state["view"][:3, :3] @ ld
    GL.glUniform3f(state["light_dir_loc"], *view_ld)
    GL.glUniform4f(state["light_amb_loc"], *light.ambient)
    GL.glUniform4f(state["light_diff_loc"], *light.diffuse)
    GL.glUniform4f(state["light_spec_loc"], *light.specular)

    GL.glUniform4f(state["mat_amb_loc"], *material.ambient)
    GL.glUniform4f(state["mat_diff_loc"], *material.diffuse)
    GL.glUniform4f(state["mat_spec_loc"], *material.specular)
    GL.glUniform1f(state["mat_exp_loc"], material.exponent)

    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["num_indices"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ld = imgui.input_float3(label="light dir", v=light.direction)
    light.direction = list(ld)
    _, c = imgui.color_edit3(label="light ambient", col=light.ambient[:3])
    light.ambient[:3] = list(c)
    _, c = imgui.color_edit3(label="light diffuse", col=light.diffuse[:3])
    light.diffuse[:3] = list(c)
    _, c = imgui.color_edit3(label="light specular", col=light.specular[:3])
    light.specular[:3] = list(c)
    imgui.separator()
    _, c = imgui.color_edit3(label="material diffuse", col=material.diffuse[:3])
    material.ambient[:3] = list(c)
    material.diffuse[:3] = list(c)
    _, c = imgui.color_edit3(
        label="material specular", col=material.specular[:3]
    )
    material.specular[:3] = list(c)
    _, material.exponent = imgui.slider_float(
        label="specular exponent", v=material.exponent, v_min=1.0, v_max=256.0
    )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 05 - phong sphere",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
