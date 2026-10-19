"""McNopper Example 40: GPU cloth simulation via compute shader.

Original: McNopper/OpenGL Example40. A 32x32 cloth grid is simulated by a compute
shader using Verlet-style integration over previous/current/output
SSBOs. A static sphere collides with the cloth.
"""

from __future__ import annotations

import math

import numpy as np

from mcnopper import common
from mcnopper.common import GL

ROWS: int = 31
SPHERE_CENTER = (0.0, 0.0, -0.01, 1.0)
SPHERE_RADIUS: float = 1.0
state: dict = {}


def init() -> bool:
    grid: common.Shape = common.make_rectangular_grid_plane(
        horizontal_extend=2.0, vertical_extend=2.0, rows=ROWS, columns=ROWS
    )
    # Rotate so the grid lies in xz-plane and lift by 1.1
    rot: np.ndarray = common.translate(x=0, y=1.1, z=0) @ common.rotate_x(-90.0)
    rotated = (rot @ grid.vertices.T).T

    distance_rest: float = float(
        np.linalg.norm(rotated[1, :3] - rotated[0, :3])
    )
    distance_diag = math.sqrt(2.0) * distance_rest

    n_verts: int = len(grid.vertices)
    # Pad normals to vec4 for std430.
    normals_padded: np.ndarray = np.zeros((n_verts, 4), dtype=np.float32)
    normals_padded[:, :3] = grid.normals

    prog_comp: int = common.compile_program_files(
        script_file=__file__, comp="cloth.comp.glsl"
    )
    prog: int = common.compile_program_files(
        script_file=__file__, vert="cloth.vert.glsl", frag="cloth.frag.glsl"
    )
    prog_sphere: int = common.compile_program_files(
        script_file=__file__, vert="sphere.vert.glsl", frag="sphere.frag.glsl"
    )

    GL.glUseProgram(prog_comp)
    GL.glUniform1i(
        common.uniform(prog=prog_comp, name="u_verticesPerRow"), ROWS + 1
    )
    GL.glUniform1f(
        common.uniform(prog=prog_comp, name="u_distanceRest"), distance_rest
    )
    GL.glUniform1f(
        common.uniform(prog=prog_comp, name="u_distanceDiagonalRest"),
        distance_diag,
    )
    GL.glUniform4f(
        common.uniform(prog=prog_comp, name="u_sphereCenter"), *SPHERE_CENTER
    )
    GL.glUniform1f(
        common.uniform(prog=prog_comp, name="u_sphereRadius"), SPHERE_RADIUS
    )

    vertices_buffers = []
    for i in range(3):
        b: int = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, b)
        if i < 2:
            GL.glBufferData(
                target=GL.GL_SHADER_STORAGE_BUFFER,
                size=rotated.nbytes,
                data=np.ascontiguousarray(rotated, dtype=np.float32),
                usage=GL.GL_DYNAMIC_DRAW,
            )
        else:
            GL.glBufferData(
                target=GL.GL_SHADER_STORAGE_BUFFER,
                size=rotated.nbytes,
                data=None,
                usage=GL.GL_DYNAMIC_DRAW,
            )
        vertices_buffers.append(b)
    normals_buffer: int = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, normals_buffer)
    GL.glBufferData(
        target=GL.GL_SHADER_STORAGE_BUFFER,
        size=normals_padded.nbytes,
        data=normals_padded,
        usage=GL.GL_DYNAMIC_DRAW,
    )

    GL.glUseProgram(prog)
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, normals_buffer)
    n_loc: int = GL.glGetAttribLocation(program=prog, name="a_normal")
    GL.glVertexAttribPointer(
        index=n_loc,
        size=4,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(n_loc)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertices_buffers[2])
    v_loc: int = GL.glGetAttribLocation(program=prog, name="a_vertex")
    GL.glVertexAttribPointer(
        index=v_loc,
        size=4,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glEnableVertexAttribArray(v_loc)
    vbo_i: int = common.make_vbo(
        data=grid.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_i)

    light: np.ndarray = np.array([1, 1, 1], dtype=np.float32)
    light /= np.linalg.norm(light)
    GL.glUniform3f(common.uniform(prog=prog, name="u_lightDirection"), *light)
    GL.glUniform4f(common.uniform(prog=prog, name="u_color"), 0, 1, 1, 1)

    # Sphere VAO (visualization)
    sphere: common.Shape = common.make_sphere(radius=SPHERE_RADIUS, slices=64)
    GL.glUseProgram(prog_sphere)
    vao_s: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_s)
    vbo_sv: int = common.make_vbo(sphere.vertices)
    vbo_sn: int = common.make_vbo(sphere.normals)
    vbo_si: int = common.make_vbo(
        data=sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    common.attach_attrib(
        prog=prog_sphere, vao=vao_s, vbo=vbo_sv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog_sphere, vao=vao_s, vbo=vbo_sn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_si)
    GL.glUniform3f(
        common.uniform(prog=prog_sphere, name="u_lightDirection"), *light
    )
    # The C sets the sphere's colour once in initSphere (render_sphere.c:
    # color = {1, 1, 0, 1}). The port set u_color only on the CLOTH program
    # (above), so the sphere program's u_color stayed at its zero default and
    # sphere.frag.glsl's fragColor = u_color * lambert drew the sphere black on
    # black. Found by the 2026-09-09 C-vs-Python frame comparison: the C's
    # yellow sphere, nothing in the port.
    GL.glUniform4f(
        common.uniform(prog=prog_sphere, name="u_color"), 1.0, 1.0, 0.0, 1.0
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glEnable(GL.GL_DEPTH_TEST)

    state.update(
        prog=prog,
        prog_comp=prog_comp,
        prog_sphere=prog_sphere,
        vao=vao,
        vao_s=vao_s,
        n_indices=len(grid.indices),
        n_sphere=len(sphere.indices),
        v_buffers=vertices_buffers,
        n_buffer=normals_buffer,
        v_loc=v_loc,
        prev=0,
        cur=1,
        out=2,
        total_time=0.0,
        grid_verts=rotated.copy().astype(np.float32),
        mvp_loc=common.uniform(prog=prog, name="u_modelViewProjectionMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        dt_loc=common.uniform(prog=prog_comp, name="u_deltaTime"),
        mvp_loc_s=common.uniform(
            prog=prog_sphere, name="u_modelViewProjectionMatrix"
        ),
        nm_loc_s=common.uniform(prog=prog_sphere, name="u_normalMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    view: np.ndarray = common.look_at(
        eye=[0, 4, 4], center=[0, 0, 0], up=[0, 1, 0]
    )
    state["vp"] = proj @ view


def update(dt: float) -> bool:
    state["total_time"] += dt
    if state["total_time"] >= 10.0:
        state["total_time"] = 0.0
        for i in (state["prev"], state["cur"]):
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, state["v_buffers"][i])
            GL.glBufferSubData(
                target=GL.GL_SHADER_STORAGE_BUFFER,
                offset=0,
                size=state["grid_verts"].nbytes,
                data=state["grid_verts"],
            )

    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

    # Draw sphere
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glUseProgram(state["prog_sphere"])
    sphere_m: np.ndarray = common.translate(*SPHERE_CENTER[:3])
    mvp: np.ndarray = state["vp"] @ sphere_m
    common.upload_matrix(loc=state["mvp_loc_s"], m=mvp)
    common.upload_matrix3(loc=state["nm_loc_s"], m=sphere_m[:3, :3])
    GL.glBindVertexArray(state["vao_s"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_sphere"], GL.GL_UNSIGNED_INT, None
    )
    GL.glDisable(GL.GL_CULL_FACE)

    # Cloth simulate
    GL.glUseProgram(state["prog_comp"])
    GL.glUniform1f(state["dt_loc"], dt)
    GL.glBindBufferBase(
        GL.GL_SHADER_STORAGE_BUFFER, 1, state["v_buffers"][state["prev"]]
    )
    GL.glBindBufferBase(
        GL.GL_SHADER_STORAGE_BUFFER, 2, state["v_buffers"][state["cur"]]
    )
    GL.glBindBufferBase(
        GL.GL_SHADER_STORAGE_BUFFER, 3, state["v_buffers"][state["out"]]
    )
    GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 4, state["n_buffer"])
    GL.glDispatchCompute(1, 1, 1)
    GL.glMemoryBarrier(GL.GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT)

    # Cloth draw
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["mvp_loc"], m=state["vp"])
    common.upload_matrix3(loc=state["nm_loc"], m=np.eye(3, dtype=np.float32))
    GL.glBindVertexArray(state["vao"])
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, state["v_buffers"][state["out"]])
    GL.glVertexAttribPointer(
        index=state["v_loc"],
        size=4,
        type=GL.GL_FLOAT,
        normalized=GL.GL_FALSE,
        stride=0,
        pointer=None,
    )
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_indices"], GL.GL_UNSIGNED_INT, None
    )

    state["prev"] = (state["prev"] + 1) % 3
    state["cur"] = (state["cur"] + 1) % 3
    state["out"] = (state["out"] + 1) % 3
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    imgui.text(f"{ROWS + 1}x{ROWS + 1} cloth, sphere collision")
    imgui.text(f"t = {state['total_time']:.1f} s (resets at 10s)")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 40 - GPU cloth",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=1024,
        height=768,
        gl_major=4,
        gl_minor=3,
        samples=4,
    )
