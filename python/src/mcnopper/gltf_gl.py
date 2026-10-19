"""GL side of the glTF slice: upload primitives and textures, bind materials.

Python counterpart of the GL-touching half of GLUS's glTF loader (the VAO,
texture and fallback-texture creation in ``glusGltfLoadScene``) plus the C
examples' ``bindPrimitiveMaterial``, written for the verbatim
``glus_gltf_pbr`` shaders: their attribute names, texture units 0-7, the 1x1
fallback textures for empty material slots, and the full uniform set. Needs a
GL context; ``_gltf`` is the context-free half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import common, gltf, ibl
from .common import GL

# Texture units, as the C examples assign them: material slots 0-4, IBL 5-7.
UNIT_BASE_COLOR: int = 0
UNIT_METALLIC_ROUGHNESS: int = 1
UNIT_NORMAL: int = 2
UNIT_OCCLUSION: int = 3
UNIT_EMISSIVE: int = 4
UNIT_SPECULAR_ENV: int = 5
UNIT_DIFFUSE_ENV: int = 6
UNIT_BRDF_LUT: int = 7

# glTF alphaMode as the PBR shader's u_alphaMode integer.
ALPHA_MODES: dict[str, int] = {"OPAQUE": 0, "MASK": 1, "BLEND": 2}

# Every uniform glus_gltf_pbr.frag.glsl + either vertex shader declares,
# except the samplers (set once at init) and u_roughnessScale.
PBR_UNIFORM_NAMES: tuple[str, ...] = (
    "u_modelMatrix",
    "u_viewProjectionMatrix",
    "u_normalMatrix",
    "u_jointMatrices",
    "u_eye",
    "u_baseColorFactor",
    "u_metallicFactor",
    "u_roughnessFactor",
    "u_emissiveFactor",
    "u_occlusionStrength",
    "u_alphaCutoff",
    "u_alphaMode",
    "u_hasNormalMap",
    "u_normalScale",
    "u_baseColorTexCoordSet",
    "u_metallicRoughnessTexCoordSet",
    "u_normalTexCoordSet",
    "u_occlusionTexCoordSet",
    "u_emissiveTexCoordSet",
)

SAMPLER_UNITS: tuple[tuple[str, int], ...] = (
    ("u_baseColorTexture", UNIT_BASE_COLOR),
    ("u_metallicRoughnessTexture", UNIT_METALLIC_ROUGHNESS),
    ("u_normalTexture", UNIT_NORMAL),
    ("u_occlusionTexture", UNIT_OCCLUSION),
    ("u_emissiveTexture", UNIT_EMISSIVE),
    ("u_specularEnvMap", UNIT_SPECULAR_ENV),
    ("u_diffuseEnvMap", UNIT_DIFFUSE_ENV),
    ("u_brdfLUT", UNIT_BRDF_LUT),
)


def compile_pbr_program(skinned: bool) -> int:
    """GLUS's glTF PBR program: ``glus_gltf_pbr[_skinned].vert`` + its frag."""
    return ibl.compile_glus_program(
        vert="glus_gltf_pbr_skinned.vert.glsl"
        if skinned
        else "glus_gltf_pbr.vert.glsl",
        frag="glus_gltf_pbr.frag.glsl",
    )


def setup_pbr_program(prog: int, number_roughness: int) -> dict[str, int]:
    """Set a PBR program's sampler units + roughness scale; return locations.

    The returned dict maps every name in :data:`PBR_UNIFORM_NAMES` to its
    location (-1 for one the vertex shader variant lacks).
    """
    GL.glUseProgram(prog)
    for sampler, unit in SAMPLER_UNITS:
        GL.glUniform1i(common.uniform(prog=prog, name=sampler), unit)
    GL.glUniform1f(
        common.uniform(prog=prog, name="u_roughnessScale"),
        float(number_roughness - 1),
    )
    return {
        name: common.uniform(prog=prog, name=name) for name in PBR_UNIFORM_NAMES
    }


# --- textures ------------------------------------------------------------------
def make_texture_1x1(r: int, g: int, b: int, a: int) -> int:
    """A 1x1 RGBA8 texture: GLUS's fallback for an absent material texture."""
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RGBA8,
        1,
        1,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        np.array([r, g, b, a], dtype=np.uint8),
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
    )
    return tex


@dataclass
class Fallbacks:
    """The two GLUS fallback textures: white, and a flat (+Z) normal."""

    white: int
    normal: int


def make_fallbacks() -> Fallbacks:
    return Fallbacks(
        white=make_texture_1x1(r=255, g=255, b=255, a=255),
        normal=make_texture_1x1(r=128, g=128, b=255, a=255),
    )


def upload_image_texture(path: str, srgb: bool) -> int:
    """Decode an image file to RGBA8 and upload it mipmapped, as GLUS does.

    Rows are uploaded top-first without flipping: glTF's UV origin is the
    image's top-left, so texture row 0 is the image's top row.
    """
    img: Image.Image = Image.open(path).convert("RGBA")
    pixels: np.ndarray = np.asarray(img, dtype=np.uint8)
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_SRGB8_ALPHA8 if srgb else GL.GL_RGBA8,
        img.width,
        img.height,
        0,
        GL.GL_RGBA,
        GL.GL_UNSIGNED_BYTE,
        np.ascontiguousarray(pixels),
    )
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    GL.glTexParameteri(
        GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    return tex


def apply_sampler(tex: int, sampler: gltf.Sampler) -> None:
    """Override the texture's filter/wrap state with the glTF sampler's."""
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    for param, value in (
        (GL.GL_TEXTURE_MIN_FILTER, sampler.min_filter),
        (GL.GL_TEXTURE_MAG_FILTER, sampler.mag_filter),
        (GL.GL_TEXTURE_WRAP_S, sampler.wrap_s),
        (GL.GL_TEXTURE_WRAP_T, sampler.wrap_t),
    ):
        if value is not None:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, param, value)


@dataclass
class MaterialTextures:
    """The five GL textures a material binds on units 0-4, plus its flags."""

    units: dict[int, int]  # unit -> texture id
    has_normal_map: bool


class TextureCache:
    """Image index -> GL texture, so shared images upload once (as GLUS)."""

    def __init__(self, scene: gltf.Scene) -> None:
        self.scene: gltf.Scene = scene
        self.by_image: dict[tuple[int, bool], int] = {}

    def get(self, ref: gltf.TextureRef, srgb: bool) -> int:
        key: tuple[int, bool] = (ref.image, srgb)
        if key not in self.by_image:
            path: str | None = self.scene.image_paths[ref.image]
            if path is None:
                raise NotImplementedError(
                    "embedded (bufferView) images are not ported"
                )
            self.by_image[key] = upload_image_texture(path=path, srgb=srgb)
        tex: int = self.by_image[key]
        apply_sampler(tex=tex, sampler=ref.sampler)
        return tex


def material_textures(
    mat: gltf.Material,
    cache: TextureCache,
    fallbacks: Fallbacks,
    has_tangents: bool,
) -> MaterialTextures:
    """Resolve a material's texture slots to GL textures (= gltfFillMaterial).

    Base colour and emissive are sRGB-decoded; the data textures are not. A
    normal map only counts when the primitive carries tangents, as in GLUS.
    """
    units: dict[int, int] = {
        UNIT_BASE_COLOR: fallbacks.white,
        UNIT_METALLIC_ROUGHNESS: fallbacks.white,
        UNIT_NORMAL: fallbacks.normal,
        UNIT_OCCLUSION: fallbacks.white,
        UNIT_EMISSIVE: fallbacks.white,
    }
    if mat.base_color_texture is not None:
        units[UNIT_BASE_COLOR] = cache.get(mat.base_color_texture, True)
    if mat.metallic_roughness_texture is not None:
        units[UNIT_METALLIC_ROUGHNESS] = cache.get(
            mat.metallic_roughness_texture, False
        )
    if mat.normal_texture is not None:
        units[UNIT_NORMAL] = cache.get(mat.normal_texture, False)
    if mat.occlusion_texture is not None:
        units[UNIT_OCCLUSION] = cache.get(mat.occlusion_texture, False)
    if mat.emissive_texture is not None:
        units[UNIT_EMISSIVE] = cache.get(mat.emissive_texture, True)
    return MaterialTextures(
        units=units,
        has_normal_map=mat.normal_texture is not None and has_tangents,
    )


def bind_material(
    u: dict[str, int], mat: gltf.Material, textures: MaterialTextures
) -> None:
    """Bind a material's textures on units 0-4 and set its factor uniforms.

    = the C examples' ``bindPrimitiveMaterial`` minus the matrices, which
    the caller sets (model + normal, or the joint matrices).
    """
    for unit, tex in textures.units.items():
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glUniform4f(u["u_baseColorFactor"], *mat.base_color_factor)
    GL.glUniform1f(u["u_metallicFactor"], mat.metallic_factor)
    GL.glUniform1f(u["u_roughnessFactor"], mat.roughness_factor)
    GL.glUniform3f(u["u_emissiveFactor"], *mat.emissive_factor)
    GL.glUniform1f(u["u_occlusionStrength"], mat.occlusion_strength)
    GL.glUniform1f(u["u_alphaCutoff"], mat.alpha_cutoff)
    GL.glUniform1i(u["u_alphaMode"], ALPHA_MODES.get(mat.alpha_mode, 0))
    GL.glUniform1i(u["u_hasNormalMap"], int(textures.has_normal_map))
    GL.glUniform1f(u["u_normalScale"], mat.normal_scale)
    for name, ref in (
        ("u_baseColorTexCoordSet", mat.base_color_texture),
        ("u_metallicRoughnessTexCoordSet", mat.metallic_roughness_texture),
        ("u_normalTexCoordSet", mat.normal_texture),
        ("u_occlusionTexCoordSet", mat.occlusion_texture),
        ("u_emissiveTexCoordSet", mat.emissive_texture),
    ):
        GL.glUniform1i(u[name], ref.texcoord if ref is not None else 0)


# --- geometry ----------------------------------------------------------------
def _stream(
    prog: int,
    vao: int,
    name: str,
    data: np.ndarray | None,
    count: int,
    width: int,
) -> None:
    """Attach a vertex stream, or GLUS's all-zero stream when it is absent."""
    arr: np.ndarray = (
        np.ascontiguousarray(data, dtype=np.float32)
        if data is not None
        else np.zeros((count, width), dtype=np.float32)
    )
    common.attach_attrib(
        prog=prog, vao=vao, vbo=common.make_vbo(arr), name=name, size=width
    )


@dataclass
class DrawState:
    """A primitive's VAO and draw call parameters."""

    vao: int
    count: int
    indexed: bool


def upload_primitive(prog: int, prim: gltf.Primitive) -> DrawState:
    """Upload one glTF primitive into a VAO built against program ``prog``.

    Attribute names are the GLUS PBR vertex shaders' (``a_position`` ...
    ``a_color0``). Like GLUS, absent normal/tangent/texcoord/joint/weight
    streams become zeros. An absent COLOR_0 is a constant white in GLUS
    (``glVertexAttrib4f``); that is context state, not VAO state, so a white
    stream is uploaded instead -- same shader input.
    """
    n: int = int(prim.positions.shape[0])
    vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vao)
    _stream(
        prog=prog,
        vao=vao,
        name="a_position",
        data=prim.positions,
        count=n,
        width=3,
    )
    _stream(
        prog=prog, vao=vao, name="a_normal", data=prim.normals, count=n, width=3
    )
    _stream(
        prog=prog,
        vao=vao,
        name="a_tangent",
        data=prim.tangents,
        count=n,
        width=4,
    )
    _stream(
        prog=prog,
        vao=vao,
        name="a_texCoord0",
        data=prim.texcoords,
        count=n,
        width=2,
    )
    _stream(
        prog=prog, vao=vao, name="a_joints0", data=prim.joints, count=n, width=4
    )
    _stream(
        prog=prog,
        vao=vao,
        name="a_weights0",
        data=prim.weights,
        count=n,
        width=4,
    )
    _stream(
        prog=prog,
        vao=vao,
        name="a_texCoord1",
        data=prim.texcoords1,
        count=n,
        width=2,
    )
    colors: np.ndarray = (
        prim.colors
        if prim.colors is not None
        else np.ones((n, 4), dtype=np.float32)
    )
    if colors.shape[1] == 3:
        colors = np.hstack([colors, np.ones((n, 1), dtype=np.float32)])
    _stream(prog=prog, vao=vao, name="a_color0", data=colors, count=n, width=4)
    indexed: bool = prim.indices is not None
    if indexed:
        GL.glBindBuffer(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            common.make_vbo(
                data=prim.indices, target=GL.GL_ELEMENT_ARRAY_BUFFER
            ),
        )
    GL.glBindVertexArray(0)
    return DrawState(
        vao=vao,
        count=int(prim.indices.shape[0]) if indexed else n,
        indexed=indexed,
    )


def draw(ds: DrawState) -> None:
    """= ``glusGltfDrawPrimitive``."""
    GL.glBindVertexArray(ds.vao)
    if ds.indexed:
        GL.glDrawElements(GL.GL_TRIANGLES, ds.count, GL.GL_UNSIGNED_INT, None)
    else:
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, ds.count)


def normal_matrix(model: np.ndarray) -> np.ndarray:
    """GLUS's normal matrix: inverse-transpose of the model matrix's 3x3."""
    return np.linalg.inv(model[:3, :3]).T


def upload_joint_matrices(loc: int, matrices: np.ndarray) -> None:
    """Set ``u_jointMatrices`` from a row-major ``(J, 4, 4)`` array."""
    GL.glUniformMatrix4fv(
        loc,
        matrices.shape[0],
        GL.GL_TRUE,
        np.ascontiguousarray(matrices, dtype=np.float32),
    )
