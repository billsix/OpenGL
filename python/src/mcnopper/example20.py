"""McNopper Example 20: Alpha-discarded checker plane in front of monkey.

Original: McNopper/OpenGL Example20. Phong-lit suzanne (monkey.obj) is drawn first;
a checker-discard plane is drawn afterwards with blending. The plane's
texcoord matrix is animated so the pattern rotates CW.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

state: dict = {}
ui: dict = {"angle": 0.0, "obj_angle": 0.0, "auto_rotate": True, "repeat": 16.0}


def _init_obj() -> dict:
    mesh: common.ObjMesh = common.load_obj(common.find_media("monkey.obj"))
    pos: np.ndarray = mesh.positions[mesh.indices].astype(np.float32)
    nor: np.ndarray = mesh.normals[mesh.indices].astype(np.float32)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="phong.vert.glsl", frag="phong.frag.glsl"
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
    # C wavefront.c: magenta material with white specular
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.ambientColor"),
        1.0,
        0.0,
        1.0,
        1.0,
    )
    GL.glUniform4f(
        common.uniform(prog=prog, name="u_material.diffuseColor"),
        1.0,
        0.0,
        1.0,
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
    return dict(
        prog=prog,
        vao=vao,
        count=len(pos),
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        light_loc=common.uniform(prog=prog, name="u_light.direction"),
    )


def init() -> bool:
    obj = _init_obj()
    plane: common.Shape = common.make_plane(2.0)
    prog: int = common.compile_program_files(
        script_file=__file__, vert="checker.vert.glsl", frag="checker.frag.glsl"
    )
    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    vbo_v: int = common.make_vbo(plane.vertices)
    vbo_t: int = common.make_vbo(plane.texcoords)
    vbo_i: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(prog=prog, vao=vao, vbo=vbo_v, name="a_vertex", size=4)
    common.attach_attrib(
        prog=prog, vao=vao, vbo=vbo_t, name="a_texCoord", size=2
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    GL.glUniform1f(common.uniform(prog=prog, name="u_repeat"), ui["repeat"])

    view: np.ndarray = common.look_at(
        eye=[0, 0, 10], center=[0, 0, 0], up=[0, 1, 0]
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    state.update(
        obj=obj,
        prog=prog,
        vao=vao,
        n_plane=len(plane.indices),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        mv_loc=common.uniform(prog=prog, name="u_modelViewMatrix"),
        tex_loc=common.uniform(prog=prog, name="u_textureMatrix"),
        repeat_loc=common.uniform(prog=prog, name="u_repeat"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    state["proj"] = proj
    GL.glUseProgram(state["obj"]["prog"])
    common.upload_matrix(loc=state["obj"]["proj_loc"], m=proj)
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 15.0 * dt) % 360.0
        ui["obj_angle"] = (ui["obj_angle"] + 30.0 * dt) % 360.0

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    obj = state["obj"]
    GL.glUseProgram(obj["prog"])
    model_obj: np.ndarray = common.translate(
        x=0, y=-1.5, z=0
    ) @ common.rotate_y(ui["obj_angle"])
    mv: np.ndarray = state["view"] @ model_obj
    common.upload_matrix(loc=obj["mv_loc"], m=mv)
    common.upload_matrix3(loc=obj["nm_loc"], m=mv[:3, :3])
    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    GL.glUniform3f(obj["light_loc"], *(state["view"][:3, :3] @ light))
    GL.glBindVertexArray(obj["vao"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, obj["count"])

    GL.glUseProgram(state["prog"])
    plane_mv: np.ndarray = state["view"] @ common.translate(x=0, y=0, z=4)
    common.upload_matrix(loc=state["mv_loc"], m=plane_mv)
    a = np.radians(ui["angle"])
    rot2: np.ndarray = np.array(
        [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32
    )
    GL.glUniformMatrix2fv(state["tex_loc"], 1, GL.GL_TRUE, rot2)
    GL.glBindVertexArray(state["vao"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_plane"], GL.GL_UNSIGNED_INT, None
    )
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["auto_rotate"] = imgui.checkbox(
        label="auto rotate", v=ui["auto_rotate"]
    )
    _, ui["angle"] = imgui.slider_float(
        label="plane uv angle", v=ui["angle"], v_min=0, v_max=360
    )
    _, ui["repeat"] = imgui.slider_float(
        label="checker repeat", v=ui["repeat"], v_min=1.0, v_max=64.0
    )
    GL.glUseProgram(state["prog"])
    GL.glUniform1f(state["repeat_loc"], ui["repeat"])


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 20 - alpha-discard checker",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
    )
