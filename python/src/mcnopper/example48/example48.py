"""McNopper Example 48: glTF 2.0 model with PBR + split-sum IBL.

Port of the C Example48 (upstream github.com/McNopper/OpenGL, v3.2.7). Every
shader is a verbatim copy of the one the C build runs -- the example's own
``background``/``fullscreen`` pair here, and GLUS's ``glus_gltf_pbr`` and
``glus_ibl_*`` shaders in ``../glus_shaders/`` -- with only the ``#version``
line changed to 410 (the port runs a 4.1 core context). Copied by
``tools/mcnopper/copy_upstream_shaders.py``. Parity work
therefore lives on the Python side: texture formats and units, GL state, the
camera.

The IBL prefilter passes are the shared ``ibl`` module (= GLUS's
``glus_ibl.c``); the scene is loaded with ``gltf`` and uploaded/bound with
``gltf_gl``. Static model only (no skinning/animation) -- that is example
49 -- and the untextured material path: like GLUS, a 1x1 fallback texture
(white; flat normal) is bound for every texture slot a material lacks, so the
full PBR shader runs unchanged on einstein's factor-only material.

Assets (set ``$MCNOPPER_MEDIA`` or drop them under ``mcnopper/media/``):
``einstein/scene.gltf`` and ``sunny_rose_garden_4k.hdr``.
"""

from __future__ import annotations

import numpy as np

from mcnopper import common, gltf, ibl
from mcnopper import gltf_gl as gl
from mcnopper.common import GL

# Sizes from the C Example48 (main.c) and GLUS (glus_ibl.c).
NUMBER_ROUGHNESS: int = 6
SCREEN_WIDTH: int = 1024
SCREEN_HEIGHT: int = 768
MSAA: int = 4
BG_SIZE: int = 512
SPEC_SIZE: int = 256
DIFF_SIZE: int = 128
LUT_SIZE: int = 512

GLTF_MODEL: str = "einstein/scene.gltf"
PANORAMA: str = "sunny_rose_garden_4k.hdr"

# Camera constants from the C Example48 (main.c). Distances are in units of
# the scene's bounding-sphere radius, so the framing is model-size independent.
CAMERA_ORBIT_RADIUS_FACTOR: float = 1.8
CAMERA_HEIGHT_OFFSET: float = 0.4
CAMERA_ORBIT_SPEED: float = 18.0  # degrees per second
CAMERA_FOV_DEG: float = 45.0
CAMERA_FAR_RADIUS_FACTOR: float = 3.0
CAMERA_FAR_EXTRA: float = 600.0  # clears the radius-500 background sphere

state: dict = {}
# The C tone-map hard-codes gamma 2.2 and has no exposure; the slider is a
# port extra for A/B tuning and defaults to the C value.
ui: dict[str, float] = {"gamma": 2.2}


# --- scene ------------------------------------------------------------------
def _fit_matrix(scene: gltf.Scene) -> tuple[np.ndarray, float]:
    """Center the model at the origin and scale it into a unit box.

    Returns the fit matrix and the model's bounding-sphere radius (half the
    bounding-box diagonal, as GLUS computes ``sceneRadius``) in fit units, so
    the camera can be placed relative to the model the way the C build does.

    Bounds are measured in world space -- each primitive's positions pushed
    through its node ``world`` transform -- because the draw applies
    ``fit @ prim.world``. A model whose nodes carry a transform (e.g. a
    Sketchfab Z-up -> Y-up root rotation) is centered on the right point
    only when the fit and the draw agree on the space.
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
    print("[Example 48] Loading glTF + building IBL (~5s)...")
    panorama: int = common.upload_hdr_2d(common.find_media(PANORAMA))
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

    scene: gltf.Scene = gltf.load_scene(common.find_media(GLTF_MODEL))
    print(f"[Example 48] {len(scene.primitives)} primitives loaded.")
    state["fit"], state["radius"] = _fit_matrix(scene)
    u_pbr: dict[str, int] = gl.setup_pbr_program(
        prog=prog_pbr, number_roughness=NUMBER_ROUGHNESS
    )
    fallbacks: gl.Fallbacks = gl.make_fallbacks()
    cache: gl.TextureCache = gl.TextureCache(scene)
    prims: list[dict] = []
    for prim in scene.primitives:
        mat: gltf.Material = (
            scene.materials[prim.material]
            if prim.material is not None
            else gltf.Material()
        )
        model: np.ndarray = state["fit"] @ prim.world
        prims.append(
            {
                "draw": gl.upload_primitive(prog=prog_pbr, prim=prim),
                "model": model,
                "normal": gl.normal_matrix(model),
                "material": mat,
                "textures": gl.material_textures(
                    mat=mat,
                    cache=cache,
                    fallbacks=fallbacks,
                    has_tangents=prim.tangents is not None,
                ),
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
        print(f"[Example 48] MSAA framebuffer incomplete: 0x{status:x}")
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
        prog_bg=prog_bg,
        prog_full=prog_full,
        prims=prims,
        vao_bg=vao_bg,
        n_bg=len(bg_sphere.indices),
        vao_full=vao_full,
        full_tex=full_tex,
        full_fbo=full_fbo,
        ibl=ibl_set,
        pbr=u_pbr,
        vp_bg=common.uniform(prog=prog_bg, name="u_viewProjectionMatrix"),
        gam_f=common.uniform(prog=prog_full, name="u_gamma"),
    )
    print("[Example 48] Ready.")
    return True


def reshape(w: int, h: int) -> None:
    state["w"], state["h"] = w, h


def update(dt: float) -> bool:
    w, h = state.get("w", 1), state.get("h", 1)
    t: float = state.get("t", 0.0) + dt
    state["t"] = t
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
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

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

    # --- PBR IBL textures on units 5-7 ---
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_SPECULAR_ENV)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, state["ibl"]["spec"])
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_DIFFUSE_ENV)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, state["ibl"]["diff"])
    GL.glActiveTexture(GL.GL_TEXTURE0 + gl.UNIT_BRDF_LUT)
    GL.glBindTexture(GL.GL_TEXTURE_2D, state["ibl"]["lut"])

    u: dict[str, int] = state["pbr"]
    GL.glUseProgram(state["prog_pbr"])
    common.upload_matrix(loc=u["u_viewProjectionMatrix"], m=vp)
    GL.glUniform4f(u["u_eye"], *eye, 1.0)

    # --- Opaque + mask primitives (the C pass A; einstein has no blend) ---
    for p in state["prims"]:
        mat: gltf.Material = p["material"]
        if mat.alpha_mode == "BLEND":
            continue
        if mat.double_sided:
            GL.glDisable(GL.GL_CULL_FACE)
        else:
            GL.glEnable(GL.GL_CULL_FACE)
        gl.bind_material(u=u, mat=mat, textures=p["textures"])
        common.upload_matrix(loc=u["u_modelMatrix"], m=p["model"])
        common.upload_matrix3(loc=u["u_normalMatrix"], m=p["normal"])
        gl.draw(p["draw"])

    # --- Resolve: MSAA average, tone-map, gamma ---
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDisable(GL.GL_MULTISAMPLE)
    GL.glViewport(0, 0, w, h)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, state["full_tex"])
    GL.glUseProgram(state["prog_full"])
    GL.glUniform1f(state["gam_f"], ui["gamma"])
    GL.glBindVertexArray(state["vao_full"])
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
    return True


def panel() -> None:
    if not common.HAS_IMGUI:
        return
    from imgui_bundle import imgui

    _, ui["gamma"] = imgui.slider_float(
        label="gamma", v=ui["gamma"], v_min=1.0, v_max=3.0
    )
    imgui.text(f"primitives: {len(state.get('prims', []))}")


if __name__ == "__main__":
    common.run_demo(
        title="McNopper Example 48 - glTF PBR IBL",
        callbacks=common.DemoCallbacks(
            init=init, reshape=reshape, update=update, imgui_panel=panel
        ),
        gl_major=4,
        gl_minor=1,
    )
