"""Minimal, dependency-free safetensors reader and writer.

Why this exists
---------------
A BrainPatch runtime artifact is a handful of vectors. Reading them should not
force a user to install numpy, let alone torch: ``pip install brainpatch`` has
to stay light enough that inspecting, validating and installing a patch works
on any Python 3.10+ with nothing else present.

The safetensors container is simple enough to implement directly:

===============  =========================================================
8 bytes          little-endian ``uint64`` header length ``N``
``N`` bytes      UTF-8 JSON header
remainder        raw little-endian tensor data
===============  =========================================================

Each header entry is ``{"dtype": ..., "shape": [...], "data_offsets": [a, b]}``
with offsets relative to the end of the header. ``__metadata__`` is an optional
reserved key holding a flat ``str -> str`` map.

This module deliberately supports **only** the float dtypes a patch vector can
use. It is not a general safetensors implementation, and it never executes
anything from the file -- the whole point of the format choice is that a patch
is inert data.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

#: dtype name -> (bytes per element, struct format or None for bf16)
_DTYPES: dict[str, tuple[int, str | None]] = {
    "F64": (8, "<d"),
    "F32": (4, "<f"),
    "F16": (2, "<e"),
    "BF16": (2, None),  # handled specially: upper 16 bits of an f32
}

#: Refuse absurd headers before allocating anything.
MAX_HEADER_BYTES = 16 * 1024 * 1024


class SafetensorsError(ValueError):
    """The byte stream is not a well-formed safetensors container."""


@dataclass(frozen=True)
class Tensor:
    """A dense float tensor as plain Python floats.

    Attributes
    ----------
    dtype:
        The dtype it was *stored* as. Values are always decoded to Python
        floats; this records the on-disk precision so a round trip can preserve
        it and so a caller can report the real artifact size.
    """

    dtype: str
    shape: tuple[int, ...]
    data: list[float]

    def __post_init__(self) -> None:
        expected = 1
        for dim in self.shape:
            expected *= dim
        if len(self.data) != expected:
            raise SafetensorsError(
                f"shape {self.shape} implies {expected} elements but got {len(self.data)}"
            )

    @property
    def numel(self) -> int:
        return len(self.data)

    @property
    def nbytes(self) -> int:
        return self.numel * _DTYPES[self.dtype][0]


def _decode_bf16(raw: bytes, count: int) -> list[float]:
    """bfloat16 is the top 16 bits of an IEEE-754 float32."""
    out: list[float] = []
    for i in range(count):
        bits = raw[2 * i] | (raw[2 * i + 1] << 8)
        out.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    return out


def _encode_bf16(values: Sequence[float]) -> bytes:
    """Round-to-nearest-even truncation of float32 to its top 16 bits."""
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        # round-to-nearest-even on the discarded low 16 bits
        rounding = 0x7FFF + ((bits >> 16) & 1)
        bits = (bits + rounding) >> 16
        out += struct.pack("<H", bits & 0xFFFF)
    return bytes(out)


def load(blob: bytes) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Parse a safetensors byte string.

    Returns
    -------
    (tensors, metadata)

    Raises
    ------
    SafetensorsError
        On any structural problem. Offsets are validated against the actual
        payload length, so a truncated or overlapping file is rejected rather
        than silently yielding garbage.
    """
    if len(blob) < 8:
        raise SafetensorsError("file is too short to contain a header length")

    (header_len,) = struct.unpack("<Q", blob[:8])
    if header_len > MAX_HEADER_BYTES:
        raise SafetensorsError(f"header claims {header_len} bytes, refusing to read")
    if 8 + header_len > len(blob):
        raise SafetensorsError("header length exceeds file size")

    try:
        header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError("header must be a JSON object")

    payload = blob[8 + header_len :]
    metadata_raw = header.pop("__metadata__", {})
    if not isinstance(metadata_raw, dict):
        raise SafetensorsError("__metadata__ must be a JSON object")
    metadata = {str(k): str(v) for k, v in metadata_raw.items()}

    tensors: dict[str, Tensor] = {}
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise SafetensorsError(f"entry {name!r} is not an object")
        dtype = spec.get("dtype")
        if dtype not in _DTYPES:
            raise SafetensorsError(
                f"tensor {name!r} has unsupported dtype {dtype!r}; "
                f"supported: {sorted(_DTYPES)}"
            )
        shape = spec.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            raise SafetensorsError(f"tensor {name!r} has an invalid shape {shape!r}")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise SafetensorsError(f"tensor {name!r} has invalid data_offsets")

        start, end = offsets
        if not 0 <= start <= end <= len(payload):
            raise SafetensorsError(
                f"tensor {name!r} offsets [{start}, {end}] fall outside the "
                f"{len(payload)}-byte payload"
            )

        item_size, fmt = _DTYPES[dtype]
        raw = payload[start:end]
        if len(raw) % item_size:
            raise SafetensorsError(f"tensor {name!r} byte length is not a multiple of {item_size}")
        count = len(raw) // item_size

        expected = 1
        for dim in shape:
            expected *= dim
        if count != expected:
            raise SafetensorsError(
                f"tensor {name!r} shape {shape} implies {expected} elements "
                f"but the byte range holds {count}"
            )

        if fmt is None:
            values = _decode_bf16(raw, count)
        else:
            values = [v[0] for v in struct.iter_unpack(fmt, raw)]

        tensors[name] = Tensor(dtype=dtype, shape=tuple(shape), data=values)

    return tensors, metadata


def dump(tensors: dict[str, Tensor], metadata: dict[str, str] | None = None) -> bytes:
    """Serialize tensors into a safetensors byte string.

    Tensor names are written in sorted order so the output is byte-deterministic
    for a given input -- which is what lets a patch carry a stable checksum.
    """
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    body = bytearray()
    for name in sorted(tensors):
        tensor = tensors[name]
        item_size, fmt = _DTYPES[tensor.dtype]
        if fmt is None:
            raw = _encode_bf16(tensor.data)
        else:
            raw = b"".join(struct.pack(fmt, float(v)) for v in tensor.data)
        start = len(body)
        body += raw
        header[name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(body)],
        }

    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(body)


def vector(values: Iterable[float], dtype: str = "F32") -> Tensor:
    """Build a 1-D tensor from an iterable of floats."""
    if dtype not in _DTYPES:
        raise SafetensorsError(f"unsupported dtype {dtype!r}")
    data = [float(v) for v in values]
    return Tensor(dtype=dtype, shape=(len(data),), data=data)


def supported_dtypes() -> tuple[str, ...]:
    return tuple(sorted(_DTYPES))
