"""McNopper Example 50: glTF PBR + IBL + skinning in a Rec.2020 working space.

Port of the C Example50 (upstream github.com/McNopper/OpenGL, v3.2.7). Compared
to Example49 the C does three things: (1) it works in linear Rec.2020 -- the
HDR panorama is converted Rec.709 -> Rec.2020 on the CPU before the IBL
build, so every IBL texture is in the working space; (2) it opens its window
through McNopper/EGL, probes the EGL colour-space extensions, picks the best
HDR output (BT.2020 PQ -> HLG -> linear, then scRGB, Display P3, sRGB) and
the resolve shader encodes for that choice; (3) the camera orbits the other
way. The C window backend is Windows-only (the Linux ``main`` exits with a
message), so there is no C run to compare against on Linux.

This port runs in the GLFW window every other demo uses, which is the C's
final fallback: an 8-bit sRGB surface, output colour space ``OUT_SRGB``. The
resolve shader is still the verbatim Example50 one, so the other eight
encodings can be selected live from the panel to see what the C would send
to an HDR surface (they will look wrong on an SDR window -- that is the
encoding, not a bug). Reference white and peak nits are the C defaults.

Note: the C header comment says the PBR fragment shader converts base-colour
and emissive texture samples to Rec.2020 as well, but the C loads GLUS's
stock ``glus_gltf_pbr.frag.glsl``, which does not; the port follows the
code, so textures stay Rec.709 while the IBL is Rec.2020, as in the C.

Every
shader is a verbatim copy of the one the C build runs -- the example's own
``background``/``fullscreen`` pair here, and GLUS's ``glus_gltf_pbr``
(unskinned and skinned vertex shaders, one fragment shader) and
``glus_ibl_*`` shaders in ``../glus_shaders/`` -- with only the ``#version``
line changed to 410 (the port runs a 4.1 core context). Copied by
``tools/mcnopper/copy_upstream_shaders.py``.

Scene loading, the node hierarchy, skinning and T/R/S animation are the
shared ``gltf`` module (= ``glusGltfLoadScene`` / ``glusGltfUpdateAnimation``);
VAO/texture upload and material binding are ``gltf_gl``; the IBL prefilter
passes are ``ibl`` (= ``glus_ibl.c``). This driver owns the camera, the MSAA
resolve pass and, per primitive, the choice of the skinned or unskinned PBR
program -- as the C example does.

Assets (set ``$MCNOPPER_MEDIA`` or drop them under ``mcnopper/media/``):
``phoenix/scene.gltf`` (+ ``textures/``) and ``sunny_rose_garden_4k.hdr``.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common, gltf, ibl
from mcnopper import gltf_gl as gl
from mcnopper.common import GL

# Sizes from the C Example50 (main.c) and GLUS (glus_ibl.c).
NUMBER_ROUGHNESS: int = 6
SCREEN_WIDTH: int = 1280
SCREEN_HEIGHT: int = 720
MSAA: int = 4
BG_SIZE: int = 512
SPEC_SIZE: int = 256
DIFF_SIZE: int = 128
LUT_SIZE: int = 512

GLTF_MODEL: str = "phoenix/scene.gltf"
PANORAMA: str = "sunny_rose_garden_4k.hdr"

# Camera constants from the C Example50 (main.c). Distances are in units of
# the scene's bounding-sphere radius, so the framing is model-size independent.
# Example50 orbits in the opposite direction to Example49.
CAMERA_ORBIT_RADIUS_FACTOR: float = 1.8
CAMERA_HEIGHT_OFFSET: float = 0.4
CAMERA_ORBIT_SPEED: float = 18.0  # degrees per second
CAMERA_FOV_DEG: float = 45.0
CAMERA_FAR_RADIUS_FACTOR: float = 3.0
CAMERA_FAR_EXTRA: float = 600.0  # clears the radius-500 background sphere

# Linear Rec.709 (sRGB primaries) -> linear Rec.2020 (BT.2020 primaries), the
# matrix the C applies to the panorama on the CPU (main.c). Row-major.
REC709_TO_REC2020: np.ndarray = np.array(
    [
        [0.62740389896, 0.32928303716, 0.04331306388],
        [0.06909728935, 0.91954039507, 0.01136231558],
        [0.01639143887, 0.08801330909, 0.89559525204],
    ],
    dtype=np.float32,
)

# Output colour-space ids; must match shader/fullscreen.frag.glsl and main.c.
OUTPUT_COLORSPACES: tuple[str, ...] = (
    "linear (Rec.709)",
    "sRGB",
    "Display P3",
    "Display P3 linear",
    "scRGB linear (fp16)",
    "scRGB",
    "BT.2020 linear",
    "BT.2020 HLG",
    "BT.2020 PQ (HDR10)",
)
OUT_SRGB: int = 1

state: dict = {}
# The C's resolve defaults: gamma 2.2, scene 1.0 = 200 nits, 1000-nit panel,
# and the output encoding its EGL probe would settle on for an SDR window.
# The widgets are port extras for exploring the other encodings.
ui: dict[str, float] = {
    "gamma": 2.2,
    "output_colorspace": OUT_SRGB,
    "reference_white_nits": 200.0,
    "peak_nits": 1000.0,
}


# --- scene ------------------------------------------------------------------
def _fit_matrix(scene: gltf.Scene) -> tuple[np.ndarray, float]:
    """Center the model at the origin and scale it into a unit box.

    Returns the fit matrix and the model's bounding-sphere radius (half the
    bounding-box diagonal, as GLUS computes ``sceneRadius``) in fit units, so
    the camera can be placed relative to the model the way the C build does.
    Bounds are the bind pose in world space (positions through each node's
    world transform), the same box GLUS measures at load.
    """
    allpos: np.ndarray = np.vstack(
        [
            p.positions @ p.world[:3, :3].T + p.world[:3, 3]
            for p in scene.primitives
        ]
    )
    lo: np.ndarray = allpos.min(axis=0)
    hi: np.ndarray = allpos.max(axis=0)
    center: np.ndarray = (lo + hi) * 0.5
    span: float = float((hi - lo).max()) or 1.0
    scale: np.ndarray = np.diag([2.0 / span] * 3 + [1.0]).astype(np.float32)
    translate: np.ndarray = np.eye(4, dtype=np.float32)
    translate[:3, 3] = -center
    radius: float = float(np.linalg.norm(hi - lo)) * 0.5 * (2.0 / span)
    return scale @ translate, radius


def init() -> bool:
    print("[Example 50] Loading glTF + building IBL (~5s)...")  # noqa: T201
    # Convert the panorama from linear Rec.709 into linear Rec.2020 so all
    # derived IBL textures end up in the working colour space already. The
    # .hdr file is float32 RGB, so this conversion is lossless (as the C).
    w_px, h_px, pixels = common.load_hdr(common.find_media(PANORAMA))
    rec2020: np.ndarray = (pixels @ REC709_TO_REC2020.T).astype(np.float32)
    panorama: int = common.upload_hdr_pixels(w=w_px, h=h_px, px=rec2020)
    GL.glBindTexture(GL.GL_TEXTURE_2D, panorama)
    # As the C example: mipmapped panorama for the cubemap conversions.
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR_MIPMAP_LINEAR,
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    ibl_set: dict = ibl.build(
        panorama=panorama,
        number_roughness=NUMBER_ROUGHNESS,
        specular_size=SPEC_SIZE,
        diffuse_size=DIFF_SIZE,
        lut_size=LUT_SIZE,
        background_size=BG_SIZE,
    )
    GL.glDeleteTextures(size=1, array=[panorama])

    prog_pbr: int = gl.compile_pbr_program(skinned=False)
    prog_skinned: int = gl.compile_pbr_program(skinned=True)
    prog_bg: int = common.compile_program_files(
        script_file=__file__,
        vert="background.vert.glsl",
        frag="background.frag.glsl",
    )
    prog_full: int = common.compile_program_files(
        script_file=__file__,
        vert="fullscreen.vert.glsl",
        frag="fullscreen.frag.glsl",
    )
    u_pbr: dict[str, int] = gl.setup_pbr_program(
        prog=prog_pbr, number_roughness=NUMBER_ROUGHNESS
    )
    u_skinned: dict[str, int] = gl.setup_pbr_program(
        prog=prog_skinned, number_roughness=NUMBER_ROUGHNESS
    )

    scene: gltf.Scene = gltf.load_scene(common.find_media(GLTF_MODEL))
    print(  # noqa: T201
        f"[Example 50] {len(scene.primitives)} primitives, "
        f"{len(scene.skins)} skins, {len(scene.animations)} animations."
    )
    state["scene"] = scene
    state["fit"], state["radius"] = _fit_matrix(scene)

    fallbacks: gl.Fallbacks = gl.make_fallbacks()
    cache: gl.TextureCache = gl.TextureCache(scene)
    # The two vertex shaders share attribute locations, so one VAO per
    # primitive, built against the program that will draw it, serves both.
    prims: list[dict] = []
    for prim in scene.primitives:
        skinned: bool = prim.skin is not None
        mat: gltf.Material = (
            scene.materials[prim.material]
            if prim.material is not None
            else gltf.Material()
        )
        prims.append(
            {
                "draw": gl.upload_primitive(
                    prog=prog_skinned if skinned else prog_pbr, prim=prim
                ),
                "prim": prim,
                "material": mat,
                "textures": gl.material_textures(
                    mat=mat,
                    cache=cache,
                    fallbacks=fallbacks,
                    has_tangents=prim.tangents is not None,
                ),
                "skinned": skinned,
            }
        )

    bg_sphere: common.Shape = common.make_sphere(radius=500.0, slices=32)
    GL.glUseProgram(prog_bg)
    vao_bg: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_bg)
    common.attach_attrib(
        prog=prog_bg,
        vao=vao_bg,
        vbo=common.make_vbo(bg_sphere.vertices),
        name="a_vertex",
        size=4,
    )
    GL.glBindBuffer(
        GL.GL_ELEMENT_ARRAY_BUFFER,
        common.make_vbo(
            data=bg_sphere.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
        ),
    )
    GL.glUniform1i(common.uniform(prog=prog_bg, name="u_texture"), 0)

    GL.glUseProgram(prog_full)
    vao_full: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao_full)
    GL.glUniform1i(
        common.uniform(prog=prog_full, name="u_framebufferTexture"), 0
    )
    GL.glUniform1i(common.uniform(prog=prog_full, name="u_msaaSamples"), MSAA)

    # MSAA HDR framebuffer: GL_RGBA16F colour + 24-bit depth, as the C example.
    full_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, full_tex)
    GL.glTexImage2DMultisample(
        GL.GL_TEXTURE_2D_MULTISAMPLE,
        MSAA,
        GL.GL_RGBA16F,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        GL.GL_TRUE,
    )
    full_depth: int = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, full_depth)
    GL.glRenderbufferStorageMultisample(
        GL.GL_RENDERBUFFER,
        MSAA,
        GL.GL_DEPTH_COMPONENT24,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
    )
    full_fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, full_fbo)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER,
        GL.GL_COLOR_ATTACHMENT0,
        GL.GL_TEXTURE_2D_MULTISAMPLE,
        full_tex,
        0,
    )
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER,
        GL.GL_DEPTH_ATTACHMENT,
        GL.GL_RENDERBUFFER,
        full_depth,
    )
    status: int = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[Example 50] MSAA framebuffer incomplete: 0x{status:x}")  # noqa: T201
        return False
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    # Initial GL state, as the C example's init.
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glEnable(GL.GL_CULL_FACE)
    GL.glCullFace(GL.GL_BACK)
    GL.glFrontFace(GL.GL_CCW)
    GL.glEnable(GL.GL_TEXTURE_CUBE_MAP_SEAMLESS)

    state.update(
        prog_pbr=prog_pbr,
        prog_skinned=prog_skinned,
        prog_bg=prog_bg,
        prog_full=prog_full,
        u_pbr=u_pbr,
        u_skinned=u_skinned,
        prims=prims,
        vao_bg=vao_bg,
        n_bg=len(bg_sphere.indices),
        vao_full=vao_full,
        full_tex=full_tex,
        full_fbo=full_fbo,
        ibl=ibl_set,
        vp_bg=common.uniform(prog=prog_bg, name="u_viewProjectionMatrix"),
        gam_f=common.uniform(prog=prog_full, name="u_gamma"),
        cs_f=common.uniform(prog=prog_full, name="u_outputColorspace"),
        white_f=common.uniform(prog=prog_full, name="u_referenceWhiteNits"),
        peak_f=common.uniform(prog=prog_full, name="u_peakNits"),
    )
    print("[Example 50] Ready.")  # noqa: T201
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def _draw_primitive(p: dict, fit: np.ndarray) -> None:
    """= bindPrimitiveMaterial + glusGltfDrawPrimitive for one primitive.

    A skinned primitive is positioned entirely by its joint matrices (the
    skinned vertex shader has no model matrix), so the fit matrix is folded
    into them; an unskinned one gets fit @ node world as its model matrix.
    """
    mat: gltf.Material = p["material"]
    if mat.double_sided:
        GL.glDisable(GL.GL_CULL_FACE)
    else:
        GL.glEnable(GL.GL_CULL_FACE)
    prim: gltf.Primitive = p["prim"]
    scene: gltf.Scene = state["scene"]
    if p["skinned"]:
        u: dict[str, int] = state["u_skinned"]
        GL.glUseProgram(state["prog_skinned"])
        skin_idx: int | None = prim.skin
        assert skin_idx is not None  # skin index is set when p["skinned"]
        joints: np.ndarray = scene.skins[skin_idx].joint_matrices
        gl.upload_joint_matrices(
            loc=u["u_jointMatrices"], matrices=fit @ joints
        )
    else:
        u = state["u_pbr"]
        GL.glUseProgram(state["prog_pbr"])
        model: np.ndarray = fit @ prim.world
        common.upload_matrix(loc=u["u_modelMatrix"], m=model)
        common.upload_matrix3(
            loc=u["u_normalMatrix"], m=gl.normal_matrix(model)
        )
    gl.bind_material(u=u, mat=mat, textures=p["textures"])
    gl.draw(p["draw"])


def update(dt: float) -> bool:
    w, h = state.get("w", 1), state.get("h", 1)
    t: float = state.get("t", 0.0) + dt
    state["t"] = t
    scene: gltf.Scene = state["scene"]
    fit: np.ndarray = state["fit"]

    # --- Advance skinning / TRS animation (= glusGltfUpdateAnimation) ---
    gltf.update_animation(scene=scene, dt=dt)

    # The fit matrix centres the model at the origin, so the C build's
    # sceneCenter is the origin here and its sceneRadius is state["radius"].
    radius: float = state["radius"]
    orbit: float = radius * CAMERA_ORBIT_RADIUS_FACTOR
    # Near sits well inside the orbit-minus-radius gap to the model's closest
    # point (0.8 radii) while keeping the depth range narrow enough that the
    # 24-bit buffer does not z-fight; far must clear the background sphere.
    near: float = radius * 0.25
    far: float = orbit + radius * CAMERA_FAR_RADIUS_FACTOR + CAMERA_FAR_EXTRA
    proj: np.ndarray = common.perspective(
        fovy_deg=CAMERA_FOV_DEG,
        aspect=SCREEN_WIDTH / SCREEN_HEIGHT,
        near=near,
        far=far,
    )
    angle: float = np.radians((t * CAMERA_ORBIT_SPEED) % 360.0)
    eye: list[float] = [
        orbit * np.sin(angle),
        radius * CAMERA_HEIGHT_OFFSET,
        orbit * np.cos(angle),
    ]
    view: np.ndarray = common.look_at(eye=eye, center=[0, 0, 0], up=[0, 1, 0])
    vp: np.ndarray = proj @ view
    # Background: view rotation only, so the sphere is centred on the eye.
    bg_view: np.ndarray = view.copy()
    bg_view[:3, 3] = 0.0
    bg_vp: np.ndarray = proj @ bg_view

    # --- Render into the MSAA FBO ---
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, state["full_fbo"])
    GL.glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    GL.glEnable(GL.GL_MULTISAMPLE)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glDisable(GL.GL_BLEND)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)  # ty: ignore[unsupported-operator]

    # --- Background sphere (inside-facing, no depth write) ---
    GL.glFrontFace(GL.GL_CW)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glDepthMask(GL.GL_FALSE)
    GL.glDepthFunc(GL.GL_LEQUAL)
    GL.glUseProgram(state["prog_bg"])
    common.upload_matrix(loc=state["vp_bg"], m=bg_vp)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["ibl"]["bg"])
    GL.glBindVertexArray(state["vao_bg"])
    GL.glDrawElements(GL.GL_TRIANGLES, state["n_bg"], GL.GL_UNSIGNED_INT, None)
    GL.glDepthFunc(GL.GL_LESS)
    GL.glFrontFace(GL.GL_CCW)
    GL.glDepthMask(GL.GL_TRUE)

    # --- IBL textures on units 5-7 ---
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_SPECULAR_ENV)  # ty: ignore[unsupported-operator]
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, state["ibl"]["spec"])
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_DIFFUSE_ENV)  # ty: ignore[unsupported-operator]
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["ibl"]["diff"])
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_BRDF_LUT)  # ty: ignore[unsupported-operator]
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["ibl"]["lut"])

    # VP + eye on both PBR programs.
    for prog, u in (
        (state["prog_pbr"], state["u_pbr"]),
        (state["prog_skinned"], state["u_skinned"]),
    ):
        GL.glUseProgram(prog)
        common.upload_matrix(loc=u["u_viewProjectionMatrix"], m=vp)
        GL.glUniform4f(u["u_eye"], *eye, 1.0)

    # --- Pass A: opaque + mask ---
    GL.glEnable(GL.GL_DEPTH_TEST)
    for p in state["prims"]:
        if p["material"].alpha_mode != "BLEND":
            _draw_primitive(p=p, fit=fit)

    # --- Pass B: blend (no depth write) ---
    GL.glEnable(GL.GL_BLEND)
    GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    GL.glDepthMask(GL.GL_FALSE)
    for p in state["prims"]:
        if p["material"].alpha_mode == "BLEND":
            _draw_primitive(p=p, fit=fit)
    GL.glDisable(GL.GL_BLEND)
    GL.glDepthMask(GL.GL_TRUE)
    GL.glEnable(GL.GL_CULL_FACE)

    # --- Resolve: MSAA average, tone-map, gamma ---
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glViewport(0, 0, w, h)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glDisable(GL.GL_MULTISAMPLE)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(state["prog_full"])
    GL.glUniform1f(state["gam_f"], ui["gamma"])
    GL.glUniform1i(state["cs_f"], int(ui["output_colorspace"]))
    GL.glUniform1f(state["white_f"], ui["reference_white_nits"])
    GL.glUniform1f(state["peak_f"], ui["peak_nits"])
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, state["full_tex"])
    GL.glBindVertexArray(state["vao_full"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    GL.glEnable(GL.GL_DEPTH_TEST)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    scene: gltf.Scene = state["scene"]
    _, cs = imgui.combo(
        label="output colorspace",
        current_item=int(ui["output_colorspace"]),
        items=list(OUTPUT_COLORSPACES),
    )
    ui["output_colorspace"] = cs
    _, ui["gamma"] = imgui.slider_float(
        label="gamma", v=ui["gamma"], v_min=1.0, v_max=3.0
    )
    _, ui["reference_white_nits"] = imgui.slider_float(
        label="reference white (nits)",
        v=ui["reference_white_nits"],
        v_min=80.0,
        v_max=400.0,
    )
    _, ui["peak_nits"] = imgui.slider_float(
        label="peak (nits)", v=ui["peak_nits"], v_min=100.0, v_max=10000.0
    )
    imgui.text(f"primitives: {len(state.get('prims', []))}")
    if scene.animations:
        anim: gltf.Animation = scene.animations[scene.active_animation]
        imgui.text(
            f"animation: {anim.name!r} "
            f"{scene.animation_time - anim.start:5.2f}/{anim.duration:.2f}s"
        )


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 50 - Rec.2020 PBR IBL HDR EGL",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=1,
    )
