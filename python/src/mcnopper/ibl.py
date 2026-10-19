"""Image-based-lighting prefilter passes = GLUS's ``glus_ibl.c``.

Builds, on the GPU and at startup, the four textures the glTF PBR shader
samples from an equirectangular HDR panorama: the roughness-layered
prefiltered specular cubemap array, the diffuse irradiance cubemap, the
split-sum BRDF lookup table, and the unfiltered background cubemap. The
shaders are GLUS's own, verbatim, in ``glus_shaders/`` (the port's
``GLUS_SHADER_DIR``). Needs a GL context.
"""

from __future__ import annotations

from pathlib import Path

from . import common
from .common import GL

# The port's GLUS_SHADER_DIR: GLUS's shaders, verbatim, under their own names.
GLUS_SHADER_DIR: Path = Path(__file__).resolve().parent / "glus_shaders"
# find_shader() takes a script path and looks beside it; any path in the dir.
_GLUS_ANCHOR: str = str(GLUS_SHADER_DIR / "glus_ibl.vert.glsl")

# GLUS's intermediate source cubemap size (IBL_SOURCE_CUBEMAP_SIZE): the
# panorama is converted to this mipmapped cubemap and the prefilter passes
# sample it with per-sample mip selection, so the sun is averaged rather
# than speckling.
SOURCE_CUBEMAP_SIZE: int = 512


def compile_glus_program(vert: str, frag: str) -> int:
    """Compile a program from two shaders in ``GLUS_SHADER_DIR``."""
    return common.compile_program_files(
        script_file=_GLUS_ANCHOR, vert=vert, frag=frag
    )


def make_cubemap(size: int, fmt=GL.GL_RGBA32F, mipmap: bool = False) -> int:
    """Allocate an empty float cubemap; ``mipmap`` selects trilinear sampling.

    GL_RGBA32F because, as GLUS notes, it is required to be colour-renderable
    while GL_RGB32F is not.
    """
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, tex)
    for face in range(6):
        GL.glTexImage2D(
            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            0,
            fmt,
            size,
            size,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            None,
        )
    for wrap in (
        GL.GL_TEXTURE_WRAP_S,
        GL.GL_TEXTURE_WRAP_T,
        GL.GL_TEXTURE_WRAP_R,
    ):
        GL.glTexParameteri(GL.GL_TEXTURE_CUBE_MAP, wrap, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP,
        GL.GL_TEXTURE_MIN_FILTER,
        GL.GL_LINEAR_MIPMAP_LINEAR if mipmap else GL.GL_LINEAR,
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
    )
    return tex


def make_cubemap_array(size: int, layers: int) -> int:
    tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP_ARRAY, tex)
    GL.glTexImage3D(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY,
        0,
        GL.GL_RGBA32F,
        size,
        size,
        layers,
        0,
        GL.GL_RGBA,
        GL.GL_FLOAT,
        None,
    )
    for wrap in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP_ARRAY, wrap, GL.GL_CLAMP_TO_EDGE
        )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
    )
    GL.glTexParameteri(
        GL.GL_TEXTURE_CUBE_MAP_ARRAY, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
    )
    return tex


def _render_cube_faces(prog: int, tex: int, size: int) -> None:
    """Draw the fullscreen quad into each face of ``tex`` (level 0)."""
    GL.glViewport(0, 0, size, size)
    for face in range(6):
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            tex,
            0,
        )
        GL.glUniform1i(common.uniform(prog=prog, name="u_face"), face)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)


def build(
    panorama: int,
    *,
    number_roughness: int,
    specular_size: int,
    diffuse_size: int,
    lut_size: int,
    background_size: int,
) -> dict:
    """Run the four GPU IBL passes as GLUS does; return the IBL texture ids.

    = ``glusIblBuildSpecularEnvironmentMap`` + ``...DiffuseEnvironmentMap`` +
    ``...BrdfLookupTable`` + ``...BackgroundCubemap``. The panorama (a bound
    GL_TEXTURE_2D, float) is first converted to a mipmapped source cubemap;
    the specular and diffuse prefilter passes sample that rather than the
    panorama. The background cubemap is a second, unmipmapped conversion.
    The quad comes from ``gl_VertexID`` in ``glus_ibl.vert.glsl``, so only
    an empty VAO is bound. Returns ``{"spec", "diff", "lut", "bg"}``.
    """
    prog_spec: int = compile_glus_program(
        vert="glus_ibl.vert.glsl", frag="glus_ibl_specular.frag.glsl"
    )
    prog_diff: int = compile_glus_program(
        vert="glus_ibl.vert.glsl", frag="glus_ibl_diffuse.frag.glsl"
    )
    prog_lut: int = compile_glus_program(
        vert="glus_ibl.vert.glsl", frag="glus_ibl_brdf_lut.frag.glsl"
    )
    prog_p2c: int = compile_glus_program(
        vert="glus_ibl.vert.glsl", frag="glus_ibl_background.frag.glsl"
    )
    spec_tex: int = make_cubemap_array(
        size=specular_size, layers=6 * number_roughness
    )
    diff_tex: int = make_cubemap(diffuse_size)
    bg_tex: int = make_cubemap(background_size)
    src_tex: int = make_cubemap(size=SOURCE_CUBEMAP_SIZE, mipmap=True)
    lut_tex: int = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, lut_tex)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D,
        0,
        GL.GL_RG32F,
        lut_size,
        lut_size,
        0,
        GL.GL_RG,
        GL.GL_FLOAT,
        None,
    )
    for wrap in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, wrap, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    quad_vao: int = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(quad_vao)

    fbo: int = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glDisable(GL.GL_CULL_FACE)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, panorama)

    # Panorama -> cubemap, twice: the mipmapped prefilter source and the
    # background. Same shader, same face mapping.
    GL.glUseProgram(prog_p2c)
    GL.glUniform1i(common.uniform(prog=prog_p2c, name="u_panoramaTexture"), 0)
    _render_cube_faces(prog=prog_p2c, tex=src_tex, size=SOURCE_CUBEMAP_SIZE)
    _render_cube_faces(prog=prog_p2c, tex=bg_tex, size=background_size)
    GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, src_tex)
    GL.glGenerateMipmap(GL.GL_TEXTURE_CUBE_MAP)

    GL.glUseProgram(prog_spec)
    GL.glUniform1i(common.uniform(prog=prog_spec, name="u_cubemapTexture"), 0)
    GL.glUniform1i(
        common.uniform(prog=prog_spec, name="u_width"), SOURCE_CUBEMAP_SIZE
    )
    GL.glViewport(0, 0, specular_size, specular_size)
    for k in range(number_roughness):
        GL.glUniform1f(
            common.uniform(prog=prog_spec, name="u_roughness"),
            k / max(number_roughness - 1, 1),
        )
        for face in range(6):
            GL.glFramebufferTextureLayer(
                GL.GL_FRAMEBUFFER,
                GL.GL_COLOR_ATTACHMENT0,
                spec_tex,
                0,
                k * 6 + face,
            )
            GL.glUniform1i(common.uniform(prog=prog_spec, name="u_face"), face)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    GL.glUseProgram(prog_diff)
    GL.glUniform1i(common.uniform(prog=prog_diff, name="u_cubemapTexture"), 0)
    GL.glUniform1i(
        common.uniform(prog=prog_diff, name="u_width"), SOURCE_CUBEMAP_SIZE
    )
    _render_cube_faces(prog=prog_diff, tex=diff_tex, size=diffuse_size)

    GL.glUseProgram(prog_lut)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, lut_tex, 0
    )
    GL.glViewport(0, 0, lut_size, lut_size)
    GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
    GL.glDeleteFramebuffers(1, [fbo])
    GL.glDeleteVertexArrays(1, [quad_vao])
    GL.glDeleteTextures(size=1, array=[src_tex])
    for prog in (prog_spec, prog_diff, prog_lut, prog_p2c):
        GL.glDeleteProgram(prog)
    return {"spec": spec_tex, "diff": diff_tex, "lut": lut_tex, "bg": bg_tex}
