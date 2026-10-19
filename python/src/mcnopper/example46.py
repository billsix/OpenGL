"""McNopper Example 46: Voxel Cone Tracing global illumination.

Original: McNopper/OpenGL Example46 (Crassin et al. 2011 PG). The Sponza atrium
plus a moving emissive orange sphere are voxelized into a 256^3
RGBA16F radiance volume each frame, mip-mapped, then cone-traced for
indirect diffuse + specular + AO in the FS.

This is the most complex example in the set. The Python port preserves
the algorithm shape - voxelize pass, mipmap, then VCT render - while
treating the Sponza OBJ as a single material draw (the C version walks
per-group materials with diffuse textures).
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

WINDOW_WIDTH: int = 1024
WINDOW_HEIGHT: int = 768
VCT_GRID: int = 256
VCT_MIPLEVELS: int = 9
VCT_WORLD_SIZE: float = 2.0
SPONZA_SCALE: float = 0.005375
SPONZA_T: tuple[float, float, float] = (6.05, -65.15, 3.87)
SPHERE_RADIUS: float = 0.02
SPHERE_COLOR: tuple[float, float, float] = (1.0, 0.8, 0.2)
SPHERE_ORBIT_RADIUS: float = 0.243
SPHERE_ORBIT_Y: float = -0.26
SPHERE_ORBIT_PERIOD: float = 7.5
state: dict = {}
ui: dict = {
    "orbit_paused": True,  # the C starts with the sphere orbit paused
    "yaw": 90.0,
    "pitch": 0.0,
    "eye": [0.0, -0.30, 0.0],
}


def init() -> bool:
    print("[Example 46] Loading Sponza OBJ (this may take a few seconds)...")  # noqa: T201
    mesh: common.ObjMesh = common.load_obj(common.find_media("sponza.obj"))
    sphere: common.Shape = common.make_sphere(radius=SPHERE_RADIUS, slices=24)

    prog_vox: int = common.compile_program_files(
        script_file=__file__,
        vert="voxelize.vert.glsl",
        geom="voxelize.geom.glsl",
        frag="voxelize.frag.glsl",
    )
    prog_vct: int = common.compile_program_files(
        script_file=__file__, vert="vct.vert.glsl", frag="vct.frag.glsl"
    )

    GL.glUseProgram(prog_vox)
    GL.glUniform3f(common.uniform(prog=prog_vox, name="u_lightPos"), 0, 0.38, 0)
    GL.glUniform3f(common.uniform(prog=prog_vox, name="u_lightColor"), 0, 0, 0)
    GL.glUniform1i(
        common.uniform(prog=prog_vox, name="u_voxelGridSize"), VCT_GRID
    )
    GL.glUniform2f(
        common.uniform(prog=prog_vox, name="u_halfPixelSize"),
        1.0 / VCT_GRID,
        1.0 / VCT_GRID,
    )
    GL.glUniform1i(common.uniform(prog=prog_vox, name="u_isEmissive"), 0)
    GL.glUniform1i(common.uniform(prog=prog_vox, name="u_hasDiffuseTexture"), 0)

    GL.glUseProgram(prog_vct)
    GL.glUniform3f(common.uniform(prog=prog_vct, name="u_lightPos"), 0, 0.38, 0)
    GL.glUniform3f(common.uniform(prog=prog_vct, name="u_lightColor"), 0, 0, 0)
    GL.glUniform1f(
        common.uniform(prog=prog_vct, name="u_voxelGridWorldSize"),
        VCT_WORLD_SIZE,
    )
    GL.glUniform1i(
        common.uniform(prog=prog_vct, name="u_voxelDimensions"), VCT_GRID
    )
    GL.glUniform1i(common.uniform(prog=prog_vct, name="u_hasDiffuseTexture"), 0)

    voxel_grid: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_3D, voxel_grid)
    GL.glTexStorage3D(
        GL.GL_TEXTURE_3D,
        VCT_MIPLEVELS,
        GL.GL_RGBA16F,
        VCT_GRID,
        VCT_GRID,
        VCT_GRID,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
    )
    GL.glTexParameteri(GL.GL_TEXTURE_3D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_3D, GL.GL_TEXTURE_WRAP_R, GL.GL_CLAMP_TO_BORDER
    )
    GL.glTexParameterfv(
        GL.GL_TEXTURE_3D,
        GL.GL_TEXTURE_BORDER_COLOR,
        np.zeros(4, dtype=np.float32),
    )

    def _make_obj_vao(prog: int, vp: int, vn: int, vt: int) -> int:
        vao: int = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vp)
        GL.glVertexAttribPointer(
            index=0,
            size=4,
            type=GL.GL_FLOAT,
            normalized=GL.GL_FALSE,
            stride=0,
            pointer=None,
        )
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vn)
        GL.glVertexAttribPointer(
            index=1,
            size=3,
            type=GL.GL_FLOAT,
            normalized=GL.GL_FALSE,
            stride=0,
            pointer=None,
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vt)
        GL.glVertexAttribPointer(
            index=2,
            size=2,
            type=GL.GL_FLOAT,
            normalized=GL.GL_FALSE,
            stride=0,
            pointer=None,
        )
        GL.glEnableVertexAttribArray(2)
        return vao

    vbo_v: int = common.make_vbo(mesh.positions)
    vbo_n: int = common.make_vbo(mesh.normals)
    vbo_t: int = common.make_vbo(mesh.texcoords)
    vbo_i: int = common.make_vbo(
        data=mesh.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    GL.glUseProgram(prog_vct)
    vao: int = _make_obj_vao(prog=prog_vct, vp=vbo_v, vn=vbo_n, vt=vbo_t)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)
    # sponza.mtl: 24 of the 25 materials carry a map_Kd; the C loads them
    # trilinear + anisotropic (GL_TEXTURE_MAX_ANISOTROPY is core in 4.6).
    common.upload_material_diffuse_textures(mesh.materials)
    max_aniso: float = GL.glGetFloatv(GL.GL_MAX_TEXTURE_MAX_ANISOTROPY)
    for material in mesh.materials.values():
        if material.diffuse_texture_name:
            GL.glBindTexture(GL.GL_TEXTURE_2D, material.diffuse_texture_name)
            GL.glTexParameterf(
                GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_ANISOTROPY, max_aniso
            )
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    s_pos: np.ndarray = (
        sphere.vertices[sphere.indices].astype(np.float32)
        if len(sphere.indices)
        else sphere.vertices
    )
    s_nor: np.ndarray = (
        sphere.normals[sphere.indices].astype(np.float32)
        if len(sphere.indices)
        else sphere.normals
    )
    s_uv: np.ndarray = (
        sphere.texcoords[sphere.indices].astype(np.float32)
        if len(sphere.indices)
        else sphere.texcoords
    )
    vbo_sv: int = common.make_vbo(s_pos)
    vbo_sn: int = common.make_vbo(s_nor)
    vbo_st: int = common.make_vbo(s_uv)
    vao_s: int = _make_obj_vao(prog=prog_vct, vp=vbo_sv, vn=vbo_sn, vt=vbo_st)

    model: np.ndarray = common.scale_m(
        x=SPONZA_SCALE, y=SPONZA_SCALE, z=SPONZA_SCALE
    ) @ common.translate(*SPONZA_T)

    GL.glClearColor(0, 0, 0, 1)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)

    state.update(
        prog_vox=prog_vox,
        prog_vct=prog_vct,
        vao=vao,
        vao_s=vao_s,
        groups=mesh.groups,
        s_count=len(s_pos),
        voxel_grid=voxel_grid,
        model=model,
        total_time=0.0,
        vox_model_loc=common.uniform(prog=prog_vox, name="u_modelMatrix"),
        vox_diff_loc=common.uniform(prog=prog_vox, name="u_diffuseColor"),
        vox_emiss_loc=common.uniform(prog=prog_vox, name="u_isEmissive"),
        vox_hastex_loc=common.uniform(
            prog=prog_vox, name="u_hasDiffuseTexture"
        ),
        vct_model_loc=common.uniform(prog=prog_vct, name="u_modelMatrix"),
        vct_mvp_loc=common.uniform(prog=prog_vct, name="u_mvpMatrix"),
        vct_cam_loc=common.uniform(prog=prog_vct, name="u_cameraPos"),
        vct_diff_loc=common.uniform(prog=prog_vct, name="u_diffuseColor"),
        vct_spec_loc=common.uniform(prog=prog_vct, name="u_specularColor"),
        vct_shin_loc=common.uniform(prog=prog_vct, name="u_shininess"),
        vct_hastex_loc=common.uniform(
            prog=prog_vct, name="u_hasDiffuseTexture"
        ),
    )
    return True


def _bind_group_material(
    group: common.ObjGroup,
    diffuse_loc: int,
    specular_loc: int,
    shininess_loc: int,
    has_texture_loc: int,
) -> None:
    """C ``bindGroupMaterial``: the group's .mtl colours and map_Kd on
    unit 1, or the 0.8 grey no-material fallback. ``specular_loc`` is -1
    for the voxelize program, which has no specular term."""
    material: common.ObjMaterial | None = group.material
    if material is not None:
        GL.glUniform4f(diffuse_loc, *material.diffuse)
        if specular_loc >= 0:
            GL.glUniform4f(specular_loc, *material.specular)
            GL.glUniform1f(shininess_loc, material.shininess)
        GL.glBindTexture(GL.GL_TEXTURE_2D, material.diffuse_texture_name)
        GL.glUniform1i(
            has_texture_loc, 1 if material.diffuse_texture_name else 0
        )
    else:
        GL.glUniform4f(diffuse_loc, 0.8, 0.8, 0.8, 1.0)
        if specular_loc >= 0:
            GL.glUniform4f(specular_loc, 0.0, 0.0, 0.0, 1.0)
            GL.glUniform1f(shininess_loc, 10.0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUniform1i(has_texture_loc, 0)


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    if not ui["orbit_paused"]:
        state["total_time"] += dt
    yaw_r: float = math.radians(ui["yaw"])
    pitch_r: float = math.radians(ui["pitch"])
    fwd: tuple[float, float, float] = (
        math.sin(yaw_r) * math.cos(pitch_r),
        math.sin(pitch_r),
        -math.cos(yaw_r) * math.cos(pitch_r),
    )
    eye: list[float] = ui["eye"]
    view: np.ndarray = common.look_at(
        eye=eye,
        center=[eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2]],
        up=[0, 1, 0],
    )
    w, h = state.get("w", WINDOW_WIDTH), state.get("h", WINDOW_HEIGHT)
    proj: np.ndarray = common.perspective(
        fovy_deg=60.0, aspect=w / max(h, 1), near=0.001, far=10.0
    )
    vp: np.ndarray = proj @ view
    mvp: np.ndarray = vp @ state["model"]

    a: float = state["total_time"] * 2.0 * math.pi / SPHERE_ORBIT_PERIOD
    sphere_model: np.ndarray = common.translate(
        x=SPHERE_ORBIT_RADIUS * math.cos(a),
        y=SPHERE_ORBIT_Y,
        z=SPHERE_ORBIT_RADIUS * math.sin(a),
    )
    sphere_mvp: np.ndarray = vp @ sphere_model

    # Voxelize
    GL.glMemoryBarrier(
        GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT  # ty: ignore[unsupported-operator]
        | GL.GL_TEXTURE_UPDATE_BARRIER_BIT
        | GL.GL_TEXTURE_FETCH_BARRIER_BIT
    )
    GL.glBindTexture(GL.GL_TEXTURE_3D, state["voxel_grid"])
    GL.glClearTexImage(
        state["voxel_grid"],
        0,
        GL.GL_RGBA,
        GL.GL_FLOAT,
        np.zeros(4, dtype=np.float32),
    )

    GL.glColorMask(False, False, False, False)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glViewport(0, 0, VCT_GRID, VCT_GRID)
    GL.glUseProgram(state["prog_vox"])
    GL.glUniform1i(state["vox_emiss_loc"], 0)
    common.upload_matrix(loc=state["vox_model_loc"], m=state["model"])
    GL.glBindImageTexture(
        0,
        state["voxel_grid"],
        0,
        GL.GL_TRUE,
        0,
        GL.GL_WRITE_ONLY,
        GL.GL_RGBA16F,
    )
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindVertexArray(state["vao"])
    for group in state["groups"]:
        _bind_group_material(
            group=group,
            diffuse_loc=state["vox_diff_loc"],
            specular_loc=-1,
            shininess_loc=-1,
            has_texture_loc=state["vox_hastex_loc"],
        )
        common.draw_obj_group(group)

    GL.glUniform1i(state["vox_emiss_loc"], 1)
    GL.glUniform1i(state["vox_hastex_loc"], 0)
    common.upload_matrix(loc=state["vox_model_loc"], m=sphere_model)
    GL.glUniform4f(state["vox_diff_loc"], *SPHERE_COLOR, 1.0)
    GL.glBindVertexArray(state["vao_s"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["s_count"])

    GL.glBindImageTexture(
        0, 0, 0, GL.GL_TRUE, 0, GL.GL_WRITE_ONLY, GL.GL_RGBA16F
    )
    GL.glMemoryBarrier(
        GL.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL.GL_TEXTURE_FETCH_BARRIER_BIT  # ty: ignore[unsupported-operator]
    )
    # C: bind the grid on the active unit before building its mip chain (the
    # material loop above left unit 1 active, which holds no 3D texture).
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, state["voxel_grid"])
    GL.glGenerateMipmap(GL.GL_TEXTURE_3D)

    # Render
    GL.glColorMask(True, True, True, True)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glViewport(0, 0, w, h)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]
    GL.glUseProgram(state["prog_vct"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_3D, state["voxel_grid"])
    GL.glUniform3f(state["vct_cam_loc"], *eye)

    common.upload_matrix(loc=state["vct_model_loc"], m=state["model"])
    common.upload_matrix(loc=state["vct_mvp_loc"], m=mvp)
    GL.glActiveTexture(GL.GL_TEXTURE1)
    GL.glBindVertexArray(state["vao"])
    for group in state["groups"]:
        _bind_group_material(
            group=group,
            diffuse_loc=state["vct_diff_loc"],
            specular_loc=state["vct_spec_loc"],
            shininess_loc=state["vct_shin_loc"],
            has_texture_loc=state["vct_hastex_loc"],
        )
        common.draw_obj_group(group)

    GL.glUniform1i(state["vct_hastex_loc"], 0)
    common.upload_matrix(loc=state["vct_model_loc"], m=sphere_model)
    common.upload_matrix(loc=state["vct_mvp_loc"], m=sphere_mvp)
    GL.glUniform4f(state["vct_diff_loc"], *SPHERE_COLOR, 1.0)
    GL.glUniform4f(state["vct_spec_loc"], 1.0, 0.9, 0.5, 1.0)
    GL.glUniform1f(state["vct_shin_loc"], 128.0)
    GL.glBindVertexArray(state["vao_s"])
    GL.glDrawArrays(GL.GL_TRIANGLES, 0, state["s_count"])
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["orbit_paused"] = imgui.checkbox(
        label="pause sphere orbit", v=ui["orbit_paused"]
    )
    _, ui["yaw"] = imgui.slider_float(
        label="yaw", v=ui["yaw"], v_min=0, v_max=360
    )
    _, ui["pitch"] = imgui.slider_float(
        label="pitch", v=ui["pitch"], v_min=-89, v_max=89
    )
    _, e = imgui.input_float3(label="eye", v=ui["eye"])
    ui["eye"] = list(e)
    imgui.text(f"voxel grid: {VCT_GRID}^3 RGBA16F, {VCT_MIPLEVELS} mips")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 46 - VCT GI",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=6,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
