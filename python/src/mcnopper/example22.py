"""McNopper Example 22: Stencil shadow volumes (z-fail).

Original: McNopper/OpenGL Example22. Builds GL_TRIANGLES_ADJACENCY indices for the
torus, runs a geometry shader that extrudes silhouette edges to infinity,
then uses the z-fail algorithm with stencil INCR/DECR to mask shadowed
fragments. A blended fullscreen plane darkens those pixels.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common
from mcnopper.common import GL

CAMERA_POS: list[float] = [-5.0, 5.0, 10.0]
LIGHT_DIR: list[float] = [0.0, 0.0, 10.0]
state: dict = {}
ui: dict = {"angle": 0.0, "auto_rotate": True}


POINT_TOLERANCE: float = 0.001  # GLUS_POINT_TOLERANCE


def _build_adjacency(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """GLUS's ``glusShapeCreateAdjacencyIndicesf``: a TRIANGLES_ADJACENCY
    index buffer for the shadow-volume geometry shader.

    For each triangle (a, b, c) emit (a, ab_adj, b, bc_adj, c, ca_adj),
    where each ``_adj`` is the third vertex of the first other triangle
    sharing that edge, found by index, or else by position within
    ``POINT_TOLERANCE`` (which is what closes the torus's seams, where the
    duplicated boundary vertices have different indices but the same
    position). An edge with no neighbour keeps GLUS's degenerate default,
    the edge's own first vertex; degenerate triangles are left that way.
    """
    tris: np.ndarray = indices.reshape(-1, 3).astype(np.int64)
    out: np.ndarray = np.empty((len(tris), 6), dtype=np.uint32)
    out[:, 0::2] = tris
    out[:, 1::2] = tris  # degenerate default: (a, a, b, b, c, c)
    degenerate: np.ndarray = (
        (tris[:, 0] == tris[:, 1])
        | (tris[:, 0] == tris[:, 2])
        | (tris[:, 1] == tris[:, 2])
    )
    # Position keys: vertices within the tolerance share a key.
    quantised: np.ndarray = np.round(positions[:, :3] / POINT_TOLERANCE)
    keys: list[tuple[float, ...]] = [tuple(q + 0.0) for q in quantised]
    # edge (unordered) -> [(triangle, third vertex)] in triangle order.
    by_index: dict[tuple[int, int], list[tuple[int, int]]] = {}
    by_vertex: dict[tuple[tuple[float, ...], ...], list[tuple[int, int]]] = {}
    for i, (a, b, c) in enumerate(tris):
        if degenerate[i]:
            continue
        for u, v, w in ((a, b, c), (b, c, a), (c, a, b)):
            by_index.setdefault((min(u, v), max(u, v)), []).append((i, w))
            key = tuple(sorted((keys[u], keys[v])))
            by_vertex.setdefault(key, []).append((i, w))
    for i, (a, b, c) in enumerate(tris):
        if degenerate[i]:
            continue
        for edge, (u, v) in enumerate(((a, b), (b, c), (c, a))):
            found: list[tuple[int, int]] = [
                hit
                for hit in by_index.get((min(u, v), max(u, v)), [])
                if hit[0] != i
            ]
            if not found:
                key = tuple(sorted((keys[u], keys[v])))
                found = [hit for hit in by_vertex.get(key, []) if hit[0] != i]
            if found:
                out[i, 2 * edge + 1] = found[0][1]
    return out.reshape(-1)


def init() -> bool:
    torus: common.Shape = common.make_torus(
        inner_radius=0.5, outer_radius=1.0, slices=32, stacks=32
    )
    plane: common.Shape = common.make_plane(10.0)
    shadow_plane: common.Shape = common.make_plane(
        1.0
    )  # NDC fullscreen is its job here

    adj_indices = _build_adjacency(
        positions=torus.vertices, indices=torus.indices
    )

    prog: int = common.compile_program_files(
        script_file=__file__, vert="color.vert.glsl", frag="color.frag.glsl"
    )
    prog_sv: int = common.compile_program_files(
        script_file=__file__,
        vert="shadowvolume.vert.glsl",
        geom="shadowvolume.geom.glsl",
        frag="shadowvolume.frag.glsl",
    )
    prog_sp: int = common.compile_program_files(
        script_file=__file__, vert="shadow.vert.glsl", frag="shadow.frag.glsl"
    )

    vbo_v: int = common.make_vbo(torus.vertices)
    vbo_n: int = common.make_vbo(torus.normals)
    vbo_adj: int = common.make_vbo(
        data=adj_indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    vbo_pv: int = common.make_vbo(plane.vertices)
    vbo_pn: int = common.make_vbo(plane.normals)
    vbo_pi: int = common.make_vbo(
        data=plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )
    vbo_spv: int = common.make_vbo(shadow_plane.vertices)
    vbo_spi: int = common.make_vbo(
        data=shadow_plane.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
    )

    GL.glUseProgram(prog)
    vao_t: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_t)
    common.attach_attrib(
        prog=prog, vao=vao_t, vbo=vbo_v, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_t, vbo=vbo_n, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_adj)

    vao_p: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_p)
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pv, name="a_vertex", size=4
    )
    common.attach_attrib(
        prog=prog, vao=vao_p, vbo=vbo_pn, name="a_normal", size=3
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_pi)

    GL.glUseProgram(prog_sv)
    vao_sv: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_sv)
    common.attach_attrib(
        prog=prog_sv, vao=vao_sv, vbo=vbo_v, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_adj)

    GL.glUseProgram(prog_sp)
    vao_sp: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_sp)
    common.attach_attrib(
        prog=prog_sp, vao=vao_sp, vbo=vbo_spv, name="a_vertex", size=4
    )
    GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, vbo_spi)

    light: np.ndarray = np.array(LIGHT_DIR, dtype=np.float32)
    light /= np.linalg.norm(light)
    view: np.ndarray = common.look_at(
        eye=CAMERA_POS, center=[0, 0, 0], up=[0, 1, 0]
    )
    light_view: np.ndarray = view[:3, :3] @ light

    GL.glUseProgram(prog)
    GL.glUniform3f(
        common.uniform(prog=prog, name="u_lightDirection"), *light_view
    )
    GL.glUseProgram(prog_sv)
    GL.glUniform3f(
        common.uniform(prog=prog_sv, name="u_lightDirection"), *light_view
    )

    GL.glClearColor(0, 0, 0, 0)
    GL.glClearDepth(1.0)
    GL.glClearStencil(0)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glPolygonOffset(0.0, 100.0)

    state.update(
        prog=prog,
        prog_sv=prog_sv,
        prog_sp=prog_sp,
        vao_t=vao_t,
        vao_p=vao_p,
        vao_sv=vao_sv,
        vao_sp=vao_sp,
        n_torus=len(adj_indices),
        n_plane=len(plane.indices),
        n_shadow=len(shadow_plane.indices),
        view=view,
        proj_loc=common.uniform(prog=prog, name="u_projectionMatrix"),
        view_loc=common.uniform(prog=prog, name="u_viewMatrix"),
        m_loc=common.uniform(prog=prog, name="u_modelMatrix"),
        nm_loc=common.uniform(prog=prog, name="u_normalMatrix"),
        color_loc=common.uniform(prog=prog, name="u_shapeColor"),
        proj_loc_sv=common.uniform(prog=prog_sv, name="u_projectionMatrix"),
        view_loc_sv=common.uniform(prog=prog_sv, name="u_viewMatrix"),
        m_loc_sv=common.uniform(prog=prog_sv, name="u_modelMatrix"),
    )
    return True


def reshape(w: int, h: int) -> None:
    GL.glViewport(0, 0, w, h)
    proj: np.ndarray = common.perspective(
        fovy_deg=40.0, aspect=w / max(h, 1), near=1.0, far=100.0
    )
    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["proj_loc"], m=proj)
    GL.glUseProgram(state["prog_sv"])
    common.upload_matrix(loc=state["proj_loc_sv"], m=proj)


def update(dt: float) -> bool:
    if ui["auto_rotate"]:
        ui["angle"] = (ui["angle"] + 20.0 * dt) % 360.0

    GL.glClear(
        GL.GL_COLOR_BUFFER_BIT
        | GL.GL_DEPTH_BUFFER_BIT
        | GL.GL_STENCIL_BUFFER_BIT
    )

    GL.glUseProgram(state["prog"])
    common.upload_matrix(loc=state["view_loc"], m=state["view"])
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glEnable(GL.GL_DEPTH_TEST)

    # Plane (floor)
    plane_m: np.ndarray = common.translate(x=0, y=0, z=-5)
    plane_mv: np.ndarray = state["view"] @ plane_m
    common.upload_matrix(loc=state["m_loc"], m=plane_m)
    common.upload_matrix3(loc=state["nm_loc"], m=plane_mv[:3, :3])
    GL.glUniform4f(state["color_loc"], 0.0, 0.5, 0.0, 1.0)
    GL.glBindVertexArray(state["vao_p"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_plane"], GL.GL_UNSIGNED_INT, None
    )

    # Torus (occluder)
    torus_m: np.ndarray = common.rotate_y(ui["angle"])
    torus_mv: np.ndarray = state["view"] @ torus_m
    common.upload_matrix(loc=state["m_loc"], m=torus_m)
    common.upload_matrix3(loc=state["nm_loc"], m=torus_mv[:3, :3])
    GL.glUniform4f(state["color_loc"], 0.33, 0.0, 0.5, 1.0)
    GL.glBindVertexArray(state["vao_t"])
    GL.glDrawElements(
        GL.GL_TRIANGLES_ADJACENCY, state["n_torus"], GL.GL_UNSIGNED_INT, None
    )

    # Shadow volume (z-fail stencil)
    GL.glEnable(GL.GL_STENCIL_TEST)
    GL.glUseProgram(state["prog_sv"])
    common.upload_matrix(loc=state["view_loc_sv"], m=state["view"])
    common.upload_matrix(loc=state["m_loc_sv"], m=torus_m)
    GL.glColorMask(False, False, False, False)
    GL.glDepthMask(False)
    GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glEnable(GL.GL_DEPTH_CLAMP)
    GL.glBindVertexArray(state["vao_sv"])

    GL.glCullFace(GL.GL_FRONT)
    GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
    GL.glStencilOp(GL.GL_KEEP, GL.GL_INCR, GL.GL_KEEP)
    GL.glDrawElements(
        GL.GL_TRIANGLES_ADJACENCY, state["n_torus"], GL.GL_UNSIGNED_INT, None
    )

    GL.glCullFace(GL.GL_BACK)
    GL.glStencilFunc(GL.GL_ALWAYS, 0, 0xFF)
    GL.glStencilOp(GL.GL_KEEP, GL.GL_DECR, GL.GL_KEEP)
    GL.glDrawElements(
        GL.GL_TRIANGLES_ADJACENCY, state["n_torus"], GL.GL_UNSIGNED_INT, None
    )

    GL.glColorMask(True, True, True, True)
    GL.glDepthMask(True)
    GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
    GL.glDisable(GL.GL_DEPTH_CLAMP)

    # Darken shadowed fragments
    GL.glUseProgram(state["prog_sp"])
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_BLEND)
    GL.glStencilFunc(GL.GL_NOTEQUAL, 0, 0xFF)
    GL.glStencilOp(GL.GL_REPLACE, GL.GL_REPLACE, GL.GL_REPLACE)
    GL.glBindVertexArray(state["vao_sp"])
    GL.glDrawElements(
        GL.GL_TRIANGLES, state["n_shadow"], GL.GL_UNSIGNED_INT, None
    )
    GL.glDisable(GL.GL_BLEND)
    GL.glDisable(GL.GL_STENCIL_TEST)
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
    imgui.text("z-fail / GL_TRIANGLES_ADJACENCY / geometry-shader extrude")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 22 - shadow volumes",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        width=640,
        height=480,
        stencil_bits=8,
    )
