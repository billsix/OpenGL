"""GLUS's value-noise generators (``glus_perlin.c``), bit for bit.

``glusPerlinCreateNoise{1,2,3}D`` fill a single-channel image by summing
``octaves`` layers of cosine-interpolated lattice noise. Each octave ``i``
uses wavelength ``size // int(frequency * 2**i)`` (clamped to 1 texel) and
amplitude ``amplitude * persistence**(i + 1)`` truncated to an int; the
lattice values are ``rand() % amplitude`` drawn from a ``srand(seed)``
table the size of the image, looked up at the lattice corner's texel index
(wrapping at the image edge). The per-texel sum is truncated to a byte.

The C's ``rand()`` is glibc's, so :func:`glibc_rand` reproduces it (the
TYPE_3 additive-feedback generator): the port's noise is then identical to
the Linux C build's, texel for texel. Demos: 24 (3-D erosion field) and 26
(2-D fur-strength map).
"""

from __future__ import annotations

import numpy as np

GLIBC_RAND_MAX: int = 2147483647


def glibc_rand(seed: int, count: int) -> np.ndarray:
    """The first ``count`` values of glibc ``rand()`` after ``srand(seed)``.

    glibc's default generator (``random_r`` TYPE_3, degree 31): the state is
    seeded with a Park-Miller LCG, the first 344 outputs are discarded, and
    each output is ``(r[i-31] + r[i-3]) mod 2**32`` shifted right by one.
    ``srand(0)`` seeds with 1, as glibc does.
    """
    if seed == 0:
        seed = 1
    n: int = count + 344
    r: list[int] = [0] * (n + 34)
    r[0] = seed & 0xFFFFFFFF
    for i in range(1, 31):
        # Schrage's method, as glibc computes 16807 * r mod (2**31 - 1).
        hi: int = r[i - 1] // 127773
        lo: int = r[i - 1] % 127773
        word: int = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        r[i] = word
    for i in range(31, 34):
        r[i] = r[i - 31]
    for i in range(34, n + 34):
        r[i] = (r[i - 31] + r[i - 3]) & 0xFFFFFFFF
    out: np.ndarray = np.fromiter(
        (r[i] >> 1 for i in range(344, 344 + count)),
        dtype=np.int64,
        count=count,
    )
    return out


def _interp(v0: np.ndarray, v1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """``glusPerlinGetInterpolatedValue``: cosine interpolation."""
    ft: np.ndarray = (1.0 - np.cos(t * np.float32(np.pi))) / np.float32(2.0)
    return v0 * (np.float32(1.0) - ft) + v1 * ft


def _octaves(
    amplitude: float, persistence: float, octaves: int, frequency: float
) -> list[tuple[int, int]]:
    """``(wavelength divisor, int amplitude)`` per octave until it dies out."""
    out: list[tuple[int, int]] = []
    frequency_factor: float = 1.0
    amplitude_factor: float = 1.0 / persistence
    for _ in range(octaves):
        current_frequency: float = frequency * frequency_factor
        current_amplitude: int = int(amplitude / (1.0 * amplitude_factor))
        if current_amplitude < 1:
            break
        current_frequency = max(1.0, current_frequency)
        out.append((int(current_frequency), current_amplitude))
        frequency_factor *= 2.0
        amplitude_factor *= 1.0 / persistence
    return out


def _axis(size: int, divisor: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lattice corner indices and the in-cell parameter along one axis."""
    wavelength: int = max(size // divisor, 1)
    coords: np.ndarray = np.arange(size)
    cell: np.ndarray = coords // wavelength
    t: np.ndarray = ((coords % wavelength) / np.float32(wavelength)).astype(
        np.float32
    )
    return cell, (cell + 1) % size, t


def _to_bytes(data: np.ndarray) -> np.ndarray:
    """``(GLUSubyte)data[i]``: truncate toward zero, keep the low byte."""
    return (np.trunc(data).astype(np.int64) & 0xFF).astype(np.uint8)


def create_noise_1d(
    width: int,
    seed: int,
    frequency: float,
    amplitude: float,
    persistence: float,
    octaves: int,
) -> np.ndarray:
    """``glusPerlinCreateNoise1D``: a ``(width,)`` uint8 array."""
    table: np.ndarray = glibc_rand(seed=seed, count=width)
    data: np.ndarray = np.zeros(width, dtype=np.float32)
    for divisor, amp in _octaves(
        amplitude=amplitude,
        persistence=persistence,
        octaves=octaves,
        frequency=frequency,
    ):
        x0, x1, tx = _axis(size=width, divisor=divisor)
        p0: np.ndarray = (table[x0] % amp).astype(np.float32)
        p1: np.ndarray = (table[x1] % amp).astype(np.float32)
        data += _interp(v0=p0, v1=p1, t=tx)
    return _to_bytes(data)


def create_noise_2d(
    width: int,
    height: int,
    seed: int,
    frequency: float,
    amplitude: float,
    persistence: float,
    octaves: int,
) -> np.ndarray:
    """``glusPerlinCreateNoise2D``: a ``(height, width)`` uint8 array, row 0
    first as the C lays the image out (``glTexImage2D`` takes it as is)."""
    table: np.ndarray = glibc_rand(seed=seed, count=width * height).reshape(
        height, width
    )
    data: np.ndarray = np.zeros((height, width), dtype=np.float32)
    for divisor, amp in _octaves(
        amplitude=amplitude,
        persistence=persistence,
        octaves=octaves,
        frequency=frequency,
    ):
        x0, x1, tx = _axis(size=width, divisor=divisor)
        y0, y1, ty = _axis(size=height, divisor=divisor)
        p: list[np.ndarray] = [
            (table[np.ix_(yy, xx)] % amp).astype(np.float32)
            for yy, xx in ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
        ]
        row0: np.ndarray = _interp(v0=p[0], v1=p[1], t=tx[None, :])
        row1: np.ndarray = _interp(v0=p[2], v1=p[3], t=tx[None, :])
        data += _interp(v0=row0, v1=row1, t=ty[:, None])
    return _to_bytes(data)


def create_noise_3d(
    width: int,
    height: int,
    depth: int,
    seed: int,
    frequency: float,
    amplitude: float,
    persistence: float,
    octaves: int,
) -> np.ndarray:
    """``glusPerlinCreateNoise3D``: a ``(depth, height, width)`` uint8 array,
    slice 0 first as the C lays the image out (``glTexImage3D`` takes it
    as is)."""
    table: np.ndarray = glibc_rand(
        seed=seed, count=width * height * depth
    ).reshape(depth, height, width)
    data: np.ndarray = np.zeros((depth, height, width), dtype=np.float32)
    for divisor, amp in _octaves(
        amplitude=amplitude,
        persistence=persistence,
        octaves=octaves,
        frequency=frequency,
    ):
        x0, x1, tx = _axis(size=width, divisor=divisor)
        y0, y1, ty = _axis(size=height, divisor=divisor)
        z0, z1, tz = _axis(size=depth, divisor=divisor)
        p: list[np.ndarray] = [
            (table[np.ix_(zz, yy, xx)] % amp).astype(np.float32)
            for zz, yy, xx in (
                (z0, y0, x0),
                (z0, y0, x1),
                (z0, y1, x0),
                (z0, y1, x1),
                (z1, y0, x0),
                (z1, y0, x1),
                (z1, y1, x0),
                (z1, y1, x1),
            )
        ]
        tx3: np.ndarray = tx[None, None, :]
        ty3: np.ndarray = ty[None, :, None]
        tz3: np.ndarray = tz[:, None, None]
        x0v: np.ndarray = _interp(v0=p[0], v1=p[1], t=tx3)
        x1v: np.ndarray = _interp(v0=p[2], v1=p[3], t=tx3)
        x2v: np.ndarray = _interp(v0=p[4], v1=p[5], t=tx3)
        x3v: np.ndarray = _interp(v0=p[6], v1=p[7], t=tx3)
        y0v: np.ndarray = _interp(v0=x0v, v1=x1v, t=ty3)
        y1v: np.ndarray = _interp(v0=x2v, v1=x3v, t=ty3)
        data += _interp(v0=y0v, v1=y1v, t=tz3)
    return _to_bytes(data)
