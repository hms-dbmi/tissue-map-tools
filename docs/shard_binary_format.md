# Neuroglancer Shard Binary Format

This document explains how `.shard` files are laid out on disk, based on the
[neuroglancer sharding specification](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md)
and the cloudvolume implementation in `synthesize_shard_file()`.

## Overview

A shard file consolidates many small data chunks (e.g. individual annotations or
spatial grid cells) into a single file using a two-level index scheme.

Each chunk is identified by a uint64 **key** (annotation ID, Morton code, etc.).
The key is hashed to determine which **shard** and which **minishard** within
that shard the chunk belongs to.

## Shard Parameters

Given a `ShardingSpecification`:

| Parameter                  | Meaning                                                              |
| -------------------------- | -------------------------------------------------------------------- |
| `preshift_bits`            | Low-order bits stripped before hashing: `hash(key >> preshift_bits)` |
| `hash`                     | Hash function: `"identity"` or `"murmurhash3_x86_128"`               |
| `minishard_bits`           | Number of minishards = `2^minishard_bits`                            |
| `shard_bits`               | Number of shards = `2^shard_bits`                                    |
| `minishard_index_encoding` | `"raw"` or `"gzip"` — compression for minishard indices              |
| `data_encoding`            | `"raw"` or `"gzip"` — compression for chunk data                     |

After hashing the shifted key to a 64-bit value:

- Bits `[0, minishard_bits)` → **minishard number**
- Bits `[minishard_bits, minishard_bits + shard_bits)` → **shard number**

Shard files are named `<shard_number>.shard` (zero-padded hex in neuroglancer;
cloudvolume uses decimal, e.g. `0.shard`, `1.shard`).

## On-Disk Layout of a `.shard` File

A shard file has three contiguous sections with **no gaps** between them:

```
┌──────────────────────────────────────────────────────────┐
│  Section 1: SHARD INDEX                                  │
│  Size: 2^minishard_bits × 16 bytes                       │
│                                                          │
│  Array of (start_offset, end_offset) pairs, one per      │
│  minishard. Each value is uint64le.                       │
│  Offsets are relative to the END of this shard index.     │
│  A minishard with start == end is empty.                  │
├──────────────────────────────────────────────────────────┤
│  Section 2: MINISHARD DATA                               │
│  Size: variable (sum of all chunk data)                   │
│                                                          │
│  All chunk data concatenated. Within each minishard,      │
│  chunks are stored contiguously in sorted key order.      │
│  Minishards are concatenated in arbitrary order.          │
│  If data_encoding="gzip", each individual chunk is        │
│  compressed independently.                                │
├──────────────────────────────────────────────────────────┤
│  Section 3: MINISHARD INDICES                            │
│  Size: variable (sum of all minishard indices)            │
│                                                          │
│  All minishard indices concatenated. Each minishard       │
│  index may be independently gzip-compressed if            │
│  minishard_index_encoding="gzip".                         │
└──────────────────────────────────────────────────────────┘
```

### Why this order?

The shard index must come first because its size is fixed and known
(`index_length = 2^minishard_bits × 16` bytes), so a reader can always find it
at byte 0.

The minishard data comes second because the minishard indices contain offsets
pointing into this data section. If the data came after the indices, compressing
the indices would change their byte sizes, shifting the data offsets in a
circular dependency.

The minishard indices come last so their compressed sizes don't affect the data
layout they describe.

## Shard Index Format (Section 1)

```
For minishard i (0 ≤ i < 2^minishard_bits):
    byte offset = i × 16
    start_offset: uint64le  — start of minishard i's index (Section 3)
    end_offset:   uint64le  — end of minishard i's index (Section 3)
```

Both offsets are **relative to the end of the shard index** (i.e., relative to
byte `index_length`). To get the absolute file offset:

```
absolute_start = index_length + start_offset
absolute_end   = index_length + end_offset
```

If `start_offset == end_offset`, the minishard is empty (no chunks assigned).

## Minishard Index Format (Section 3)

Each minishard index is a `3 × N` array of uint64 values (where N is the number
of chunks in that minishard), serialized in C (row-major) order as `24 × N`
bytes:

```
Row 0: delta-encoded chunk keys
       [key₀, key₁ - key₀, key₂ - key₁, ...]

Row 1: delta-encoded data offsets (relative to end of shard index)
       [offset₀, gap₁, gap₂, ...]
       where gap_i = offset_i - (offset_{i-1} + size_{i-1})

Row 2: chunk data sizes (NOT delta-encoded)
       [size₀, size₁, size₂, ...]
```

To decode:

1. Cumulative-sum row 0 → actual keys
2. Cumulative-sum row 1, then add cumulative sizes → absolute offsets relative
   to end of shard index
3. Row 2 is already absolute sizes

If `minishard_index_encoding="gzip"`, the entire `24N` bytes of this minishard
index are gzip-compressed before being stored in Section 3.

## Minishard Data Format (Section 2)

For each chunk listed in the minishard index, the data is stored at:

```
absolute_offset = index_length + decoded_offset
```

The data for chunk `i` occupies bytes `[absolute_offset, absolute_offset + size_i)`.

If `data_encoding="gzip"`, each chunk's data is independently gzip-compressed.
The `size` in the minishard index refers to the **compressed** size on disk; the
reader must decompress to obtain the original binary content.

## Concrete Example

Consider a shard with `minishard_bits=1` (2 minishards), `data_encoding="raw"`,
`minishard_index_encoding="raw"`, containing 3 chunks:

- Chunk key=5, data=`b"AAAA"` (4 bytes) → assigned to minishard 1
- Chunk key=10, data=`b"BB"` (2 bytes) → assigned to minishard 0
- Chunk key=12, data=`b"CCC"` (3 bytes) → assigned to minishard 0

**Section 1 — Shard Index** (2 minishards × 16 bytes = 32 bytes):

```
Minishard 0: start=9, end=9+48=57   (index for 2 chunks = 24×2 = 48 bytes)
Minishard 1: start=57, end=57+24=81 (index for 1 chunk  = 24×1 = 24 bytes)
```

(Offsets relative to byte 32, the end of shard index.)

**Section 2 — Data** (starts at byte 32):

```
Byte 32: b"BB"   (chunk key=10, minishard 0)
Byte 34: b"CCC"  (chunk key=12, minishard 0)
Byte 37: b"AAAA" (chunk key=5,  minishard 1)
```

Total data = 9 bytes.

**Section 3 — Minishard Indices** (starts at byte 41):

```
Minishard 0 index (bytes 41-88):
  Row 0 (keys):    [10, 2]       (delta: 10, 12-10)
  Row 1 (offsets): [0,  0]       (delta: 0, gap=0 since chunks are contiguous)
  Row 2 (sizes):   [2,  3]

Minishard 1 index (bytes 89-112):
  Row 0 (keys):    [5]
  Row 1 (offsets): [5]           (offset from end of shard index)
  Row 2 (sizes):   [4]
```

## Compressed Morton Codes (for spatial index sharding)

When sharding the spatial index, the chunk key is a **compressed Morton code**
rather than a raw annotation ID. The Morton code interleaves bits of the 3D grid
coordinates:

For grid position `(x, y, z)` with grid shape `(gx, gy, gz)`:

- Compute bits needed per dim: `bits_x = ceil(log2(gx))`, etc.
- Interleave bits in round-robin order across dimensions that still have
  remaining bits, from least significant to most significant.

Example: grid shape `(4, 4, 2)`, position `(2, 3, 1)`:

- `bits_x=2, bits_y=2, bits_z=1`
- Interleaving: `z₀ y₀ x₀ y₁ x₁` → bits `1 1 0 1 1` → Morton code = 27

The inverse operation (Morton code → grid position) reverses this process to
recover the original `"x_y_z"` chunk name when reading sharded spatial data.
