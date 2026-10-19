"""McNopper Example 25: Phong-textured OBJ with material groups (chess king).

Original: McNopper/OpenGL Example25 - loads ChessKing.obj plus its .mtl, draws
each material group with the right diffuse texture/color. Port simplifies
to a single-material draw using the bundled ChessKing.tga.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


def init() -> bool:
    mesh: common.ObjMesh = common.load_obj(common.find_media("ChessKing.obj"))
    prog: int = common.compile_program_files(
        script_file=__file__,
        vert="phong_textured.vert.glsl",
        frag="phong_textured.frag.glsl",
    )
    GL.glUseProgram(prog)

    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(mesh.positions)
    vbo_n: int = common.make_vbo(mesh.normals)
    vbo_t: int = common.make_vbo(mesh.texcoords)
    vbo_i: int = common.make_vbo(
        data=mesh.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_n, name="a_normal", size=3)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    # The king's two groups carry their own map_Kd (ChessKing.tga and
    # ChessColumn.tga) in ChessKing.obj.mtl.
    common.upload_material_diffuse_textures(mesh.materials)
    GL.glActiveTexture(GL.GL_TEXTURE0)

    view: np.ndarray = common.look_at(
        eye=[0, 0.75, 3], center=[0, 0.75, 0], up=[0, 1, 0]
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
    GL.glUniform1i(
        common.uniform(prog=prog, name="u_material.diffuseTexture"), 0
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog=prog,
        vao=vao,
        groups=mesh.groups,
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        ambient_loc=common.uniform(prog=prog, name="u_material.ambientColor"),
        diffuse_loc=common.uniform(prog=prog, name="u_material.diffuseColor"),
        specular_loc=common.uniform(prog=prog, name="u_material.specularColor"),
        exponent_loc=common.uniform(
            prog=prog, name="u_material.specularExponent"
        ),
        use_texture_loc=common.uniform(prog=prog, name="u_useTexture"),
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
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 30.0 * dt) % 360.0
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glUseProgram(state["prog"])
    mv: np.ndarray = (
        state["view"]
        @ common.rotate_y(ui["angle"])
        @ common.scale_m(x=10, y=10, z=10)
    )
    common.upload_matrix(loc=state["mv_loc"], m=mv)
    common.upload_matrix3(loc=state["nm_loc"], m=mv[:3, :3])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindVertexArray(state["vao"])
    # C: one draw per group with its .mtl material and diffuse map.
    for group in state["groups"]:
        material: common.ObjMaterial = group.material
        GL.glUniform4f(state["ambient_loc"], *material.ambient)
        GL.glUniform4f(state["diffuse_loc"], *material.diffuse)
        GL.glUniform4f(state["specular_loc"], *material.specular)
        GL.glUniform1f(state["exponent_loc"], material.shininess)
        if material.diffuse_texture_name:
            GL.glUniform1i(state["use_texture_loc"], 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, material.diffuse_texture_name)
        else:
            GL.glUniform1i(state["use_texture_loc"], 0)
        common.draw_obj_group(group)
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


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 25 - phong-textured OBJ",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        samples=8,
    )
