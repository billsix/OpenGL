# Modern-GL traps on Mesa + PyOpenGL (Python port) (L2)

Up: `README.md`. The failure modes met porting shader-era and compute-era C to PyOpenGL on
a **Fedora / Mesa radeonsi / Wayland** box. Each cost a debugging round; several are
invisible on NVIDIA or AMD proprietary drivers, which is why the C originals never needed
them. The imgui/GLFW integration traps and the planar-shadow `w` sign are the sibling
`gl-and-imgui-gotchas.md`. Code paths are under `python/src/mcnopper/`.

## 1. PyOpenGL "no valid context" → patch `contextdata`

`imgui_bundle`'s `GlfwRenderer.__init__` dies in its first `glVertexAttribPointer(…,
c_void_p(offset))` because PyOpenGL's pointer-retention bookkeeping asks
`OpenGL.platform.GetCurrentContext()` and Mesa returns 0. The fix, applied in `common.py`
*before the first `import OpenGL.GL`*, makes the per-context bucket a constant:

```python
import OpenGL.contextdata as contextdata
contextdata.getContext = lambda context=None: 0   # constant id
import OpenGL.GL as GL   # AFTER the patch
```

Retention degenerates to one global table, harmless because the port keeps its buffers
alive on instance attributes / `state`. Two companions: issue **one GL call after
`make_context_current` and before imgui init** (a `glGetString` print is load-bearing, not
decoration), and never call `glBindVertexArray(0)` — a default VAO stays bound for the
process lifetime (`gl-and-imgui-gotchas.md` §3).

## 2. Mesa needs the `glMemoryBarrier` the proprietary drivers insert for you

Two programs sharing an image-load-store target (or SSBO, or atomic counter) across
draws/dispatches show **artifacts that look like z-fighting** — the reader sees a
partially-flushed image. Insert the barrier between them:

| After … then … | Barrier bit |
|---|---|
| `imageStore` in one program, read in another | `GL_SHADER_IMAGE_ACCESS_BARRIER_BIT` |
| SSBO write, SSBO read | `GL_SHADER_STORAGE_BARRIER_BIT` |
| atomic counter write, read | `GL_ATOMIC_COUNTER_BARRIER_BIT` |
| compute `imageStore`, then `texture()` sampling | `GL_TEXTURE_FETCH_BARRIER_BIT` |
| `glClearTexImage`, then shader access | `GL_TEXTURE_UPDATE_BARRIER_BIT` |
| compute → CPU readback (`glGetBufferSubData`) | `GL_BUFFER_UPDATE_BARRIER_BIT`, and prefer `glFinish()` |

## 3. Overlapping fragments in ONE draw race on `imageStore` → fragment-shader interlock

Barriers between draws do not order fragments *within* a draw. Proprietary drivers happen
to raster-order; Mesa does not, so the same pixel gets a different "last writer" each frame
(shimmer). Fix in GLSL: `#extension GL_ARB_fragment_shader_interlock`,
`layout(pixel_interlock_ordered) in;`, and wrap the `imageStore`s in
`beginInvocationInterlockARB()` / `endInvocationInterlockARB()`. **Do not add it to shaders
that already use `imageAtomic*`** — atomics serialize on their own. **Make it `: enable` +
`#ifdef`, not `: require`** — radeonsi supports the extension but **llvmpipe does not**, so
a `require` turns the shader into a compile error there; guarded, it degrades to the
upstream's unordered behaviour.

## 4. Clear a texture with `glClearTexImage`, not a zero PBO + `glTexSubImage`

The C clears by uploading from a zero-filled `GL_PIXEL_UNPACK_BUFFER` sized at
`sizeof(GLuint)` per pixel — **undersized for RGBA32F** (16 bytes/pixel). Proprietary
drivers tolerate the short read; Mesa rejects with `INVALID_OPERATION`.
`glClearTexImage(tex, 0, GL_RGBA, GL_FLOAT, None)` (4.4+) is one call with no size to get
wrong.

## 5. `glUniform*(0, …)` — location 0 is not reliably yours

Some C demos write to location 0 for a uniform with no `layout(location=0)`. NVIDIA-style
drivers hand the first scalar uniform location 0; Mesa does not and rejects with
`INVALID_OPERATION`. **Always `glGetUniformLocation` after linking**, even where the C
hard-codes 0. Uniform *blocks* with `layout(binding=N)` are fine.

## 6. Two GLSL rewrites applied to shaders at load (the `.glsl` files stay verbatim)

The shaders are byte-for-byte copies of the C build's (`c-to-python-rules.md` §6); these
rewrites happen in the *loader*, not the files:

- **`layout(binding = N)` on samplers needs `#version 420`.** A shader that says `410 core`
  and uses it compiles on permissive drivers and fails on Mesa ("unrecognized layout
  identifier 'binding'"). `common._maybe_bump_version` bumps the `#version` line.
- **Integer interface-block members need `flat`.** An `int material_id;` inside an `out
  VS_OUT {…}` without an interpolation qualifier compiles on NVIDIA and fails on Mesa; if a
  port shader hits this, inject `flat` (with a negative lookahead so it never emits
  `flat flat`).

## 7. `glTransformFeedbackVaryings` from PyOpenGL

PyOpenGL type-checks the third argument as `LP_LP_c_char`; no list of `str`/`bytes`
satisfies it directly. The port handles this inside `common.compile_program(tf_varyings=…,
tf_mode=…)` — it builds the `create_string_buffer`s and the `POINTER(c_char)` array and
keeps them referenced until `glLinkProgram` returns. Request transform feedback through
that keyword; never call the raw function.

## 8. Upstream shader files are not all UTF-8

A GLUS/upstream shader can carry a Latin-1 byte (e.g. `©`, 0xA9) in a copyright comment;
the C handed the compiler raw bytes, Python's default decode raises `UnicodeDecodeError`.
Read shaders with `encoding="utf-8", errors="replace"`. When grepping for such bytes use
**`LC_ALL=C grep -P`** — under a UTF-8 locale `grep -P "[\x80-\xff]"` skips invalid bytes
and reports nothing.

## 9. A ported primitive must match the original's winding, not just its shape

The port's `make_sphere` once produced the right sphere with the wrong handedness (x/z
swapped in the parametrization and a different triangle order). Every exterior view
rendered, so a crash/blank sweep was green; the one inside-out use — a panoramic background
drawn with GLUS's `glFrontFace(GL_CW)` + culling (example32) — vanished, so the statue drew
on black where the C shows the Venice courtyard. The test that pins this class of bug: flip
`glFrontFace` in a scratch copy and see if the missing geometry appears. The fix reproduces
GLUS's primitive vertex-for-vertex and index-for-index, so demos keep the C's cull idiom
verbatim. Eleven demos use the primitive.

## 10. Image loaders must reproduce the original's row order

GLUS's HDR loader stores the file's first scanline in the *last* row (`y = height - 1`
downward): bottom-up, GL's convention. A port that fills rows top-down draws every panorama
upside down — and image-based lighting sampled from it is lit from the wrong directions,
which reads as "flat" rather than "flipped". Check the C loader's row index before writing a
Python one; `np.flipud` is the whole fix. `load_hdr` does this; example32/33 depend on it.

## Where each rule is enforced in code

| Rule | Code |
|---|---|
| §1 contextdata patch, default VAO | `common.py` (top-of-module patch), `common.run_demo` |
| §6 version bump | `common._maybe_bump_version` |
| §7 transform-feedback varyings | `common.compile_program(tf_varyings=…)` |
| §9 sphere winding | `common.make_sphere` (GLUS parametrization + index order) |
| §10 HDR row order | `common.load_hdr` |
