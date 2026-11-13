"""OpenPI-compatible NumPy MessagePack encoding.

Adapted from packages/openpi-client/src/openpi_client/msgpack_numpy.py in this
repository, Copyright Physical Intelligence, licensed under Apache-2.0 (see
the repository LICENSE). Kept local to isolate NumPy2/Torch from the pi05 env.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def unpack_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        shape = tuple(value[b"shape"])
        if dtype.kind in ("V", "O", "c") or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Unsupported array dtype or shape")
        return np.ndarray(buffer=value[b"data"], dtype=dtype, shape=shape)
    if b"__npgeneric__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in ("V", "O", "c"):
            raise ValueError("Unsupported scalar dtype")
        return dtype.type(value[b"data"])
    return value


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
