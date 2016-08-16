# Copyright (c) BlackFurniture.
# See LICENSE for details.

import os

from cffi import FFI

ffi = FFI()

header = """
struct DSPHeader
{
    int16_t pred_scale;
    int16_t yn1, yn2;
    int16_t coefs[8][2];
};

void DecodeADPCM(uint8_t *src, int16_t *dst,
                 size_t num_nibbles, size_t nibble_pos,
                 struct DSPHeader * d, int channels);
"""

ffi.cdef(header)

ffi.set_source("ffcc._opt", header + """
#include "_opt_c.cpp"
""", source_extension='.cpp')

if __name__ == "__main__":
    ffi.compile()