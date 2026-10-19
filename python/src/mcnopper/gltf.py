"""Minimal glTF 2.0 loader for the McNopper PBR demos (examples 48-52).

Python reimplementation of the slice of GLUS's glTF loader these demos use
(``glusGltfLoadScene`` / ``glusGltfUpdateAnimation`` /
``glusGltfUpdateTransforms``): parse with ``pygltflib``, decode accessors into
numpy arrays, keep the node hierarchy (local TRS or matrix -> world matrices),
skins (joint node lists + inverse-bind matrices -> per-frame joint matrices)
and animations (T/R/S keyframe channels, STEP / LINEAR / CUBICSPLINE), and
expose per-primitive geometry + metallic-roughness material so a driver can
upload and draw them. GL upload lives in ``_gltf_gl`` (needs a context); this
module is context-free and unit-testable (``python _gltf.py <model>``).

Not supported, and never silently mis-rendered: sparse accessors (raise),
morph targets and ``weights`` animation channels (skipped with a warning),
KHR_* material extensions (ignored; the core factors are used).

All matrices are row-major numpy ``(4, 4)`` float32; glTF's column-major
16-tuples are transposed on the way in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pygltflib import GLTF2

# glTF componentType -> numpy dtype.
_COMPONENT_DTYPE: dict[int, np.dtype] = {
    5120: np.dtype("<i1"),  # BYTE
    5121: np.dtype("<u1"),  # UNSIGNED_BYTE
    5122: np.dtype("<i2"),  # SHORT
    5123: np.dtype("<u2"),  # UNSIGNED_SHORT
    5125: np.dtype("<u4"),  # UNSIGNED_INT
    5126: np.dtype("<f4"),  # FLOAT
}

# glTF accessor type -> component count.
_TYPE_COUNT: dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

# Animation channel target path -> component count of one keyframe value.
_PATH_COMPONENTS: dict[str, int] = {
    "translation": 3,
    "rotation": 4,
    "scale": 3,
}


@dataclass
class Sampler:
    """glTF texture sampler state; ``None`` leaves the GL default in place."""

    min_filter: int | None = None
    mag_filter: int | None = None
    wrap_s: int | None = None
    wrap_t: int | None = None


@dataclass
class TextureRef:
    """A material's reference to a texture: image, UV set and sampler."""

    image: int  # index into Scene.image_paths
    texcoord: int = 0  # which TEXCOORD_n set the texture is read with
    sampler: Sampler = field(default_factory=Sampler)


@dataclass
class Material:
    """glTF metallic-roughness material (factors + texture references).

    Defaults are the glTF spec's, which are also what GLUS uploads for a
    primitive without a material.
    """

    base_color_factor: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    emissive_factor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha_mode: str = "OPAQUE"  # OPAQUE | MASK | BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    normal_scale: float = 1.0
    occlusion_strength: float = 1.0
    base_color_texture: TextureRef | None = None
    metallic_roughness_texture: TextureRef | None = None
    normal_texture: TextureRef | None = None
    occlusion_texture: TextureRef | None = None
    emissive_texture: TextureRef | None = None


@dataclass
class Primitive:
    """One drawable: decoded vertex streams, indices, material, node, skin."""

    positions: np.ndarray  # (n, 3) float32
    normals: np.ndarray | None  # (n, 3) float32
    texcoords: np.ndarray | None  # (n, 2) float32 TEXCOORD_0
    texcoords1: np.ndarray | None  # (n, 2) float32 TEXCOORD_1
    tangents: np.ndarray | None  # (n, 4) float32 (xyz + handedness)
    colors: np.ndarray | None  # (n, 3|4) float32 COLOR_0 vertex colors, linear
    joints: np.ndarray | None  # (n, 4) float32 JOINTS_0 (indices as floats)
    weights: np.ndarray | None  # (n, 4) float32 WEIGHTS_0
    indices: np.ndarray | None  # (m,) uint32, or None for non-indexed
    material: int | None
    node: int  # index into Scene.nodes
    skin: int | None  # index into Scene.skins, for a skinned mesh
    world: np.ndarray  # (4, 4) float32 node world transform, row-major


@dataclass
class Node:
    """A scene-graph node: local transform (matrix or T/R/S) + world matrix."""

    children: list[int] = field(default_factory=list)
    mesh: int | None = None
    skin: int | None = None
    matrix: np.ndarray | None = None  # (4, 4) row-major; wins over T/R/S
    translation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rotation: np.ndarray = field(  # quaternion (x, y, z, w)
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )
    scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    world: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )


@dataclass
class Skin:
    """A skin: joint node indices, inverse-bind and per-frame joint matrices."""

    joints: list[int]
    inverse_bind: np.ndarray  # (J, 4, 4) row-major
    joint_matrices: np.ndarray  # (J, 4, 4) row-major, = world[j] @ ibm[j]


@dataclass
class Channel:
    """One animation channel: keyframe times and values for a node property.

    ``values`` is ``(K, C)`` for STEP/LINEAR and ``(K, 3, C)`` for CUBICSPLINE
    (in-tangent, value, out-tangent per keyframe).
    """

    node: int
    path: str  # translation | rotation | scale
    interpolation: str  # STEP | LINEAR | CUBICSPLINE
    times: np.ndarray  # (K,) float32
    values: np.ndarray


@dataclass
class Animation:
    name: str
    channels: list[Channel]
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Scene:
    """A loaded glTF: primitives, materials, images, node graph, skins, anims."""

    primitives: list[Primitive] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    image_paths: list[str | None] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    roots: list[int] = field(default_factory=list)
    skins: list[Skin] = field(default_factory=list)
    animations: list[Animation] = field(default_factory=list)
    active_animation: int = -1
    animation_time: float = 0.0


# --- accessor decode -------------------------------------------------------
def _buffer_bytes(gltf: GLTF2, base_dir: Path) -> list[bytes]:
    """Raw bytes of every glTF buffer (glb blob or external/data-URI .bin)."""
    blobs: list[bytes] = []
    for i, buf in enumerate(gltf.buffers):
        if buf.uri is None:  # glb-embedded
            blob: bytes | None = gltf.binary_blob()
            if blob is None:
                raise ValueError(
                    f"buffer {i} is glb-embedded but no blob present"
                )
            blobs.append(blob)
        elif buf.uri.startswith("data:"):
            blobs.append(gltf.get_data_from_buffer_uri(buf.uri))
        else:
            blobs.append((base_dir / buf.uri).read_bytes())
    return blobs


def _decode_accessor(
    gltf: GLTF2, blobs: list[bytes], accessor_index: int
) -> np.ndarray:
    """Decode accessor ``accessor_index`` to a ``(count, ncomp)`` numpy array.

    Honours the bufferView byteStride (interleaved data) and accessor byteOffset;
    normalizes integer streams flagged ``normalized``. Sparse accessors raise.
    """
    acc: Any = gltf.accessors[accessor_index]
    if acc.sparse is not None:
        raise NotImplementedError("sparse accessors not supported yet")
    dtype: np.dtype = _COMPONENT_DTYPE[acc.componentType]
    ncomp: int = _TYPE_COUNT[acc.type]
    view: Any = gltf.bufferViews[acc.bufferView]
    blob: bytes = blobs[view.buffer]
    start: int = (view.byteOffset or 0) + (acc.byteOffset or 0)
    stride: int = view.byteStride or (dtype.itemsize * ncomp)

    if stride == dtype.itemsize * ncomp:  # tightly packed
        flat: np.ndarray = np.frombuffer(
            blob, dtype=dtype, count=acc.count * ncomp, offset=start
        )
        out: np.ndarray = flat.reshape(acc.count, ncomp).astype(np.float32)
    else:  # interleaved -- gather one element per stride step
        rows: list[np.ndarray] = []
        for i in range(acc.count):
            off: int = start + i * stride
            rows.append(
                np.frombuffer(blob, dtype=dtype, count=ncomp, offset=off)
            )
        out = np.vstack(rows).astype(np.float32)

    if acc.normalized:
        maxv: float = float(np.iinfo(dtype).max)
        out = out / maxv
    return out


# --- transforms --------------------------------------------------------------
def _quat_matrix(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (4x4, row-major) of unit quaternion ``(x, y, z, w)``."""
    x, y, z, w = (float(c) for c in r)
    return np.array(
        [
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                0,
            ],
            [
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                0,
            ],
            [
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
                0,
            ],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _node_local(node: Node) -> np.ndarray:
    """Local transform of a node: its ``matrix``, or composed T * R * S."""
    if node.matrix is not None:
        return node.matrix
    translate: np.ndarray = np.eye(4, dtype=np.float32)
    translate[:3, 3] = node.translation
    scale: np.ndarray = np.diag(
        [node.scale[0], node.scale[1], node.scale[2], 1.0]
    ).astype(np.float32)
    return translate @ _quat_matrix(node.rotation) @ scale


def update_transforms(scene: Scene) -> None:
    """Recompute every node's world matrix, the joint matrices, prim worlds.

    = ``glusGltfUpdateTransforms``. Walks from the scene roots; a node that
    lists itself or an ancestor as a child is visited once (the walk is
    bounded by the node count, as in GLUS).
    """
    nodes: list[Node] = scene.nodes
    identity: np.ndarray = np.eye(4, dtype=np.float32)
    stack: list[tuple[int, np.ndarray, int]] = [
        (r, identity, 0) for r in reversed(scene.roots)
    ]
    while stack:
        index, parent_world, depth = stack.pop()
        if not 0 <= index < len(nodes) or depth >= len(nodes):
            continue
        node: Node = nodes[index]
        node.world = parent_world @ _node_local(node)
        for child in reversed(node.children):
            stack.append((child, node.world, depth + 1))
    for skin in scene.skins:
        for j, joint in enumerate(skin.joints):
            if 0 <= joint < len(nodes):
                skin.joint_matrices[j] = (
                    nodes[joint].world @ skin.inverse_bind[j]
                )
    for prim in scene.primitives:
        prim.world = nodes[prim.node].world


# --- animation sampling (= glus_animation.c) ---------------------------------
def _find_segment(times: np.ndarray, t: float) -> int:
    """Index i with ``times[i] <= t < times[i + 1]`` (t inside the range)."""
    i: int = int(np.searchsorted(times, t, side="right")) - 1
    return min(max(i, 0), len(times) - 2)


def _cubic_hermite(
    p0: np.ndarray, m0: np.ndarray, p1: np.ndarray, m1: np.ndarray, u: float
) -> np.ndarray:
    u2: float = u * u
    u3: float = u2 * u
    return (
        (2 * u3 - 3 * u2 + 1) * p0
        + (u3 - 2 * u2 + u) * m0
        + (-2 * u3 + 3 * u2) * p1
        + (u3 - u2) * m1
    )


def _sample(channel: Channel, t: float) -> np.ndarray:
    """Sample one channel at time ``t`` (clamped to its keyframe range).

    STEP holds the previous key, LINEAR mixes, CUBICSPLINE evaluates the
    Hermite spline with the glTF tangent scaling. For ``rotation`` LINEAR is a
    normalized lerp along the shortest arc and CUBICSPLINE is per-component
    then normalized, exactly as ``glusAnimationSampleQuaternionf``.
    """
    times: np.ndarray = channel.times
    cubic: bool = channel.interpolation == "CUBICSPLINE"
    n: int = len(times)

    def value(k: int) -> np.ndarray:
        return channel.values[k, 1] if cubic else channel.values[k]

    if n == 1 or t <= times[0]:
        return value(0).copy()
    if t >= times[-1]:
        return value(n - 1).copy()
    i: int = _find_segment(times=times, t=t)
    v0: np.ndarray = value(i)
    v1: np.ndarray = value(i + 1)
    if channel.interpolation == "STEP":
        return v0.copy()
    delta: float = float(times[i + 1] - times[i])
    if delta <= 0.0:  # two keys share a timestamp
        return v0.copy()
    u: float = float(t - times[i]) / delta
    quaternion: bool = channel.path == "rotation"
    if cubic:
        m0: np.ndarray = channel.values[i, 2]  # out-tangent of key i
        m1: np.ndarray = channel.values[i + 1, 0]  # in-tangent of key i+1
        out: np.ndarray = _cubic_hermite(
            p0=v0, m0=delta * m0, p1=v1, m1=delta * m1, u=u
        )
        if quaternion:
            out = out / np.linalg.norm(out)
        return out.astype(np.float32)
    if quaternion:
        if float(np.dot(v0, v1)) < 0.0:  # shortest arc
            v1 = -v1
        q: np.ndarray = v0 + (v1 - v0) * u
        return (q / np.linalg.norm(q)).astype(np.float32)
    return (v0 + (v1 - v0) * u).astype(np.float32)


def apply_animation(scene: Scene) -> None:
    """Write the active animation's sampled T/R/S into the target nodes."""
    if not 0 <= scene.active_animation < len(scene.animations):
        return
    for ch in scene.animations[scene.active_animation].channels:
        if not 0 <= ch.node < len(scene.nodes):
            continue
        node: Node = scene.nodes[ch.node]
        sampled: np.ndarray = _sample(channel=ch, t=scene.animation_time)
        if ch.path == "translation":
            node.translation = sampled
        elif ch.path == "rotation":
            node.rotation = sampled
        elif ch.path == "scale":
            node.scale = sampled
        else:  # unreachable: load_scene only keeps the three paths above
            raise ValueError(f"unsupported animation path {ch.path!r}")


def set_active_animation(scene: Scene, index: int) -> None:
    """Select animation ``index`` (-1 = none) and rewind to its start."""
    scene.active_animation = index
    scene.animation_time = (
        scene.animations[index].start
        if 0 <= index < len(scene.animations)
        else 0.0
    )


def update_animation(scene: Scene, dt: float) -> None:
    """Advance the active animation by ``dt`` seconds, looping, then update.

    = ``glusGltfUpdateAnimation``: the time wraps into ``[start, end)`` and
    the node, joint and primitive transforms are recomputed.
    """
    if 0 <= scene.active_animation < len(scene.animations):
        anim: Animation = scene.animations[scene.active_animation]
        scene.animation_time += dt
        if anim.duration > 0.0 and scene.animation_time > anim.end:
            scene.animation_time = anim.start + (
                (scene.animation_time - anim.start) % anim.duration
            )
        apply_animation(scene)
    update_transforms(scene)


# --- parsing -------------------------------------------------------------------
def _texture_ref(gltf: GLTF2, info: Any) -> TextureRef:
    """A material textureInfo -> :class:`TextureRef` (image + UV set + sampler)."""
    tex: Any = gltf.textures[info.index]
    sampler: Sampler = Sampler()
    if tex.sampler is not None:
        gs: Any = gltf.samplers[tex.sampler]
        sampler = Sampler(
            min_filter=gs.minFilter,
            mag_filter=gs.magFilter,
            wrap_s=gs.wrapS,
            wrap_t=gs.wrapT,
        )
    return TextureRef(
        image=tex.source, texcoord=info.texCoord or 0, sampler=sampler
    )


def _material(gltf: GLTF2, index: int) -> Material:
    """Convert glTF material ``index`` to our :class:`Material`."""
    gm: Any = gltf.materials[index]
    out: Material = Material()
    pr: Any = gm.pbrMetallicRoughness
    if pr is not None:
        if pr.baseColorFactor is not None:
            out.base_color_factor = tuple(pr.baseColorFactor)
        if pr.metallicFactor is not None:
            out.metallic_factor = pr.metallicFactor
        if pr.roughnessFactor is not None:
            out.roughness_factor = pr.roughnessFactor
        if pr.baseColorTexture is not None:
            out.base_color_texture = _texture_ref(
                gltf=gltf, info=pr.baseColorTexture
            )
        if pr.metallicRoughnessTexture is not None:
            out.metallic_roughness_texture = _texture_ref(
                gltf=gltf, info=pr.metallicRoughnessTexture
            )
    if gm.alphaMode is not None:
        out.alpha_mode = gm.alphaMode
    if gm.alphaCutoff is not None:
        out.alpha_cutoff = gm.alphaCutoff
    if gm.doubleSided is not None:
        out.double_sided = bool(gm.doubleSided)
    if gm.normalTexture is not None:
        out.normal_texture = _texture_ref(gltf=gltf, info=gm.normalTexture)
        if gm.normalTexture.scale is not None:
            out.normal_scale = gm.normalTexture.scale
    if gm.occlusionTexture is not None:
        out.occlusion_texture = _texture_ref(
            gltf=gltf, info=gm.occlusionTexture
        )
        if gm.occlusionTexture.strength is not None:
            out.occlusion_strength = gm.occlusionTexture.strength
    if gm.emissiveTexture is not None:
        out.emissive_texture = _texture_ref(gltf=gltf, info=gm.emissiveTexture)
    if gm.emissiveFactor is not None:
        out.emissive_factor = tuple(gm.emissiveFactor)
    return out


def _node(gn: Any) -> Node:
    node: Node = Node(
        children=list(gn.children or []), mesh=gn.mesh, skin=gn.skin
    )
    if gn.matrix:
        # glTF matrices are column-major 16-tuples; store row-major.
        node.matrix = np.array(gn.matrix, dtype=np.float32).reshape(
            4, 4, order="F"
        )
    if gn.translation:
        node.translation = np.array(gn.translation, dtype=np.float32)
    if gn.rotation:
        node.rotation = np.array(gn.rotation, dtype=np.float32)
    if gn.scale:
        node.scale = np.array(gn.scale, dtype=np.float32)
    return node


def _skin(gltf: GLTF2, blobs: list[bytes], gs: Any) -> Skin:
    joints: list[int] = list(gs.joints)
    count: int = len(joints)
    ibm: np.ndarray = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    if gs.inverseBindMatrices is not None:
        raw: np.ndarray = _decode_accessor(
            gltf=gltf, blobs=blobs, accessor_index=gs.inverseBindMatrices
        )
        n: int = min(count, raw.shape[0])
        # Each MAT4 is column-major: reshape then transpose every matrix.
        ibm[:n] = raw[:n].reshape(n, 4, 4).transpose(0, 2, 1)
    return Skin(joints=joints, inverse_bind=ibm, joint_matrices=ibm.copy())


def _animation(gltf: GLTF2, blobs: list[bytes], ga: Any) -> Animation:
    channels: list[Channel] = []
    start: float = float("inf")
    end: float = float("-inf")
    for ch in ga.channels:
        path: str = ch.target.path
        if ch.target.node is None or ch.sampler is None:
            continue
        if path not in _PATH_COMPONENTS:
            print(  # noqa: T201
                f"[_gltf] animation {ga.name!r}: skipping unsupported "
                f"channel path {path!r} (morph targets are not ported)"
            )
            continue
        sampler: Any = ga.samplers[ch.sampler]
        interp: str = sampler.interpolation or "LINEAR"
        times: np.ndarray = _decode_accessor(
            gltf=gltf, blobs=blobs, accessor_index=sampler.input
        ).ravel()
        values: np.ndarray = _decode_accessor(
            gltf=gltf, blobs=blobs, accessor_index=sampler.output
        )
        comps: int = _PATH_COMPONENTS[path]
        per_key: int = 3 if interp == "CUBICSPLINE" else 1
        if len(times) == 0 or values.shape[0] < len(times) * per_key:
            print(  # noqa: T201
                f"[_gltf] animation {ga.name!r}: channel for node "
                f"{ch.target.node} has too few sampler outputs; skipping"
            )
            continue
        values = values[: len(times) * per_key, :comps]
        if interp == "CUBICSPLINE":
            values = values.reshape(len(times), 3, comps)
        channels.append(
            Channel(
                node=ch.target.node,
                path=path,
                interpolation=interp,
                times=times.astype(np.float32),
                values=np.ascontiguousarray(values, dtype=np.float32),
            )
        )
        start = min(start, float(times[0]))
        end = max(end, float(times[-1]))
    if end < start:
        start, end = 0.0, 0.0
    return Animation(
        name=ga.name or "", channels=channels, start=start, end=end
    )


def _primitive(
    gltf: GLTF2,
    blobs: list[bytes],
    gp: Any,
    node_index: int,
    skin: int | None,
) -> Primitive:
    """Decode one glTF mesh primitive into a :class:`Primitive`."""
    attrs: Any = gp.attributes

    def stream(name: str) -> np.ndarray | None:
        idx: Any = getattr(attrs, name, None)
        return (
            _decode_accessor(gltf=gltf, blobs=blobs, accessor_index=idx)
            if idx is not None
            else None
        )

    positions: np.ndarray | None = stream("POSITION")
    if positions is None:
        raise ValueError("primitive without POSITION")
    if gp.targets:
        print("[_gltf] primitive has morph targets; they are not ported")  # noqa: T201
    indices: np.ndarray | None = (
        _decode_accessor(gltf=gltf, blobs=blobs, accessor_index=gp.indices)
        .astype(np.uint32)
        .ravel()
        if gp.indices is not None
        else None
    )
    return Primitive(
        positions=positions,
        normals=stream("NORMAL"),
        texcoords=stream("TEXCOORD_0"),
        texcoords1=stream("TEXCOORD_1"),
        tangents=stream("TANGENT"),
        colors=stream("COLOR_0"),
        joints=stream("JOINTS_0"),
        weights=stream("WEIGHTS_0"),
        indices=indices,
        material=gp.material,
        node=node_index,
        skin=skin,
        world=np.eye(4, dtype=np.float32),
    )


def load_document(path: str) -> tuple[GLTF2, list[bytes]]:
    """Parse ``path`` and read its buffers; no decoding (= cgltf_load)."""
    gltf: GLTF2 = GLTF2().load(path)
    blobs: list[bytes] = _buffer_bytes(
        gltf=gltf, base_dir=Path(path).resolve().parent
    )
    return gltf, blobs


def accessor_view_bytes(
    gltf: GLTF2, blobs: list[bytes], accessor_index: int
) -> tuple[bytes, int]:
    """The accessor's buffer view from the accessor's offset to the view's end.

    Returns ``(bytes, byte_stride)`` -- what the C Example51 uploads straight
    into its splat SSBO (``buffer + view.offset + accessor.offset``, length
    ``view.size``). ``byte_stride`` is 0 when the view declares none.
    """
    acc: Any = gltf.accessors[accessor_index]
    view: Any = gltf.bufferViews[acc.bufferView]
    start: int = (view.byteOffset or 0) + (acc.byteOffset or 0)
    end: int = (view.byteOffset or 0) + view.byteLength
    return blobs[view.buffer][start:end], view.byteStride or 0


def load_node_graph(gltf: GLTF2) -> Scene:
    """A :class:`Scene` with nodes, roots and resolved world matrices only."""
    scene: Scene = Scene()
    scene.nodes = [_node(gn) for gn in gltf.nodes]
    scene.roots = (
        list(gltf.scenes[gltf.scene or 0].nodes or [])
        if gltf.scenes
        else list(range(len(gltf.nodes)))
    )
    update_transforms(scene)
    return scene


def load_scene(path: str) -> Scene:
    """Load ``path`` (.gltf or .glb) into a :class:`Scene`, transforms resolved.

    The first animation, if any, is made active (as GLUS does), so a driver
    only has to call :func:`update_animation` each frame.
    """
    gltf: GLTF2 = GLTF2().load(path)
    base_dir: Path = Path(path).resolve().parent
    blobs: list[bytes] = _buffer_bytes(gltf=gltf, base_dir=base_dir)

    scene: Scene = Scene()
    scene.materials = [
        _material(gltf=gltf, index=i) for i in range(len(gltf.materials))
    ]
    scene.image_paths = [
        str(base_dir / img.uri) if img.uri else None for img in gltf.images
    ]
    scene.nodes = [_node(gn) for gn in gltf.nodes]
    scene.roots = (
        list(gltf.scenes[gltf.scene or 0].nodes or [])
        if gltf.scenes
        else list(range(len(gltf.nodes)))
    )
    scene.skins = [_skin(gltf=gltf, blobs=blobs, gs=gs) for gs in gltf.skins]
    scene.animations = [
        _animation(gltf=gltf, blobs=blobs, ga=ga) for ga in gltf.animations
    ]

    # One Primitive per (node with a mesh, mesh primitive), in node order.
    for node_index, node in enumerate(scene.nodes):
        if node.mesh is None:
            continue
        for gp in gltf.meshes[node.mesh].primitives:
            scene.primitives.append(
                _primitive(
                    gltf=gltf,
                    blobs=blobs,
                    gp=gp,
                    node_index=node_index,
                    skin=node.skin,
                )
            )

    if scene.animations:
        set_active_animation(scene=scene, index=0)
    update_transforms(scene)
    return scene


def _summarize(path: str) -> None:
    """Smoke test: decode a model and print its stats (no GL needed)."""
    scene: Scene = load_scene(path)
    total_v: int = sum(p.positions.shape[0] for p in scene.primitives)
    total_i: int = sum(
        0 if p.indices is None else p.indices.shape[0] for p in scene.primitives
    )
    allpos: np.ndarray = np.vstack(
        [
            p.positions @ p.world[:3, :3].T + p.world[:3, 3]
            for p in scene.primitives
        ]
    )
    print(f"{path}")  # noqa: T201
    print(  # noqa: T201
        f"  primitives: {len(scene.primitives)}  materials: "
        f"{len(scene.materials)}  nodes: {len(scene.nodes)}  skins: "
        f"{len(scene.skins)}  animations: {len(scene.animations)}"
    )
    print(f"  vertices: {total_v}  indices: {total_i}")  # noqa: T201
    print(f"  world bbox min: {allpos.min(axis=0)}  max: {allpos.max(axis=0)}")  # noqa: T201
    for s in scene.skins:
        print(f"  skin: {len(s.joints)} joints")  # noqa: T201
    for a in scene.animations:
        print(  # noqa: T201
            f"  animation {a.name!r}: {len(a.channels)} channels, "
            f"{a.start:.3f}..{a.end:.3f}s"
        )
    if scene.animations:
        update_animation(scene=scene, dt=0.5)
        print("  stepped animation 0.5s: ok")  # noqa: T201


if __name__ == "__main__":
    import sys

    _summarize(
        sys.argv[1] if len(sys.argv) > 1 else "media/einstein/scene.gltf"
    )
