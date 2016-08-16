# Copyright (c) BlackFurniture.
# See LICENSE for details.

# texture.py partly based on ffccp:
# https://github.com/drewler/ffccp

import struct
import sys
import numpy as np
from ffcc.tag import Tag, create_name_tag, NameTag

def swap16(x):
    return ((x >> 8) | (x << 8)) & 0xFFFF

def Convert3To8(v):
    return (v << 5) | (v << 2) | (v >> 1)

def Convert4To8(v):
    return (v << 4) | v

def Convert5To8(v):
    return (v << 3) | (v >> 2)

def Convert6To8(v):
    return (v << 2) | (v >> 4)

def decode_ia8(val):
    a = val & 0xFF
    i = val >> 8
    return (i, i, i, a)

def decode_rgba32(val):
    return (val & 0xFF, val >> 8)

def decode_rgb565(val):
    r = Convert5To8((val>>11) & 0x1f)
    g = Convert6To8((val>>5 ) & 0x3f)
    b = Convert5To8((val) & 0x1f)
    return (r, g, b, 0xFF)

def decode_rgb5a3(val):
    if val&0x8000:
        r=Convert5To8((val>>10) & 0x1f)
        g=Convert5To8((val>>5 ) & 0x1f)
        b=Convert5To8((val    ) & 0x1f)
        a=0xFF
    else:
        a=Convert3To8((val>>12) & 0x7);
        r=Convert4To8((val>>8) & 0xf);
        g=Convert4To8((val>>4) & 0xf);
        b=Convert4To8((val) & 0xf);
    return (r, g, b, a)

class TextureTag(Tag):
    tag = b'TXTR'
    handlers = (NameTag, b'FMT ', b'SIZE', b'IMAG', b'PALT')

    def read_fmt(self, tag, fp):
        self.format = fp.read(tag.size)[:3]

    def read_size(self, tag, fp):
        self.image_size = (fp.read_uint32(), fp.read_uint32())

    def read_palt(self, tag, fp):
        self.palette = fp.read(1024)

    def read_imag(self, tag, fp):
        if tag.size == 0:
            if self.image_size != (0, 0):
                raise NotImplementedError()
                return
            self.image = np.zeros((1, 1, 4), dtype=np.uint8)
            return
        data = fp.read(tag.size)
        if self.format == b"\x06\x01\x01": # CMPR
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            self.cmpr(data)
        elif self.format in (b"\x02\x01\x01", b"\x02\x01\x00"):
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            i = 0

            for y_off in range(0, self.image_size[1], 4):
                for x_off in range(0, self.image_size[0], 8):
                    for y in range(4):
                        for x in range(8):
                            new_y = y + y_off
                            new_x = x + x_off
                            index = data[i]
                            i += 1
                            p = self.palette[index * 2:(index + 1) * 2]
                            p, = struct.unpack('<H', p)
                            r1, g1, b1, a1 = decode_ia8(p)

                            p = self.palette[256*2 + index * 2:256*2 + (index + 1) * 2]
                            p, = struct.unpack('<H', p)
                            r2, g2, b2, a2 = decode_ia8(p)

                            self.image[new_y][new_x] = np.array([r1, a1, r2, a2])
        elif self.format in (b"\x07\x01\x01", b"\x08\x01\x00"):
            self.image = np.zeros((self.image_size[1],self.image_size[0],4),
                                   dtype=np.uint8)

            i = 0
            for y_off in range(0, self.image_size[1], 4):
                for x_off in range(0, self.image_size[0], 4):
                    for y in range(4):
                        for x in range(4):
                            new_y = y + y_off
                            new_x = x + x_off
                            v, = struct.unpack('<H', data[i*2:(i+1)*2])
                            i += 1
                            r, g, b, a = decode_ia8(v)
                            self.image[new_y][new_x] = np.array([r, g, b, a])
        elif self.format == b"\x01\x01\x01":
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            i = 0
            for y_off in range(0, self.image_size[1], 4):
                for x_off in range(0, self.image_size[0], 4):
                    for y in range(4):
                        for x in range(4):
                            new_y = y + y_off
                            new_x = x + x_off
                            v, = struct.unpack('>H', data[i*2:(i+1)*2])
                            i += 1
                            r, g, b, a = decode_rgb565(v)
                            self.image[new_y][new_x] = np.array([r, g, b, a])
        elif self.format in (b"\x05\x01\x01", b"\x06\x01\x00"):
            # 4 bits/entry, no palette
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            i = 0
            for y_off in range(0, self.image_size[1], 8):
                for x_off in range(0, self.image_size[0], 8):
                    for y in range(8):
                        for x in range(8):
                            new_y = y + y_off
                            new_x = x + x_off
                            v = data[i // 2]
                            if i % 2 == 0:
                                v >>= 4
                            else:
                                v &= 0xF
                            i += 1
                            v *= 0x11
                            self.image[new_y][new_x] = np.array([v, v, v, 255])
        elif self.format in (b"\x09\x01\x00", b"\x09\x01\x01"):
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            i = 0
            for y_off in range(0, self.image_size[1], 4):
                for x_off in range(0, self.image_size[0], 8):
                    for y in range(4):
                        for x in range(8):
                            new_y = y + y_off
                            new_x = x + x_off
                            v = data[i]
                            i += 1
                            self.image[new_y][new_x] = np.array([v, v, v, 255])
        elif self.format == b"\x03\x01\x01":
            self.image = np.zeros((self.image_size[1], self.image_size[0], 4),
                                   dtype=np.uint8)
            i = 0

            for y_off in range(0, self.image_size[1], 8):
                for x_off in range(0, self.image_size[0], 8):
                    for y in range(8):
                        for x in range(8):
                            new_y = y + y_off
                            new_x = x + x_off
                            index = data[i // 2]
                            if i % 2 == 0:
                                index >>= 4
                            else:
                                index &= 0xF
                            i += 1
                            p = self.palette[index * 2:(index + 1) * 2]
                            p, = struct.unpack('<H', p)
                            r1, g1, b1, a1 = decode_ia8(p)

                            p = self.palette[32 + index * 2:32+(index+1)*2]
                            p, = struct.unpack('<H', p)
                            r2, g2, b2, a2 = decode_ia8(p)

                            self.image[new_y][new_x] = np.array([r1, a1, r2, a2])
        elif self.format in (b"\x00\x01\x01", b"\x00\x01", b"\x00\x01\x00",
                             b"\x00\x01\x02"):
            self.image = np.zeros((self.image_size[1], self.image_size[0],4),
                                   dtype=np.uint8)
            i = 0

            off = 32
            for y_off in range(0, self.image_size[1], 4):
                for x_off in range(0, self.image_size[0], 4):
                    ii = 0
                    for y in range(4):
                        for x in range(4):
                            new_x = x + x_off
                            new_y = y + y_off
                            ind = i + ii * 2
                            a, = struct.unpack('<H', data[ind:ind+2])
                            b, = struct.unpack('<H', data[off+ind:off+ind+2])
                            ii += 1
                            index = data[i]
                            a, r = decode_rgba32(a)
                            g, b = decode_rgba32(b)
                            self.image[new_y][new_x] = np.array([r, g, b, a])
                    i += 4 * 4 * 4
        else:
            print("Unrecognized texture format: %s" % repr(self.format))
            print(self.image_size, self.palette is not None)

    def cmpr_subtile(self, subtile_data, x_offset, y_offset, cur_x, cur_y):
        COLOR0, COLOR1 = struct.unpack(">HH", subtile_data[0:4])
        RGB = [None, None, None, None]
        A   = [None, None, None, None]
        RGB[0] = bin(COLOR0)[2:].zfill(16)
        RGB[0] = np.array([255//31 * int(RGB[0][0:5],2), 255//63 * int(RGB[0][5:11],2), 255//31 * int(RGB[0][11:16],2)])
        A[0] = 255
        RGB[1] = bin(COLOR1)[2:].zfill(16)
        RGB[1] = np.array([255//31 * int(RGB[1][0:5],2), 255//63 * int(RGB[1][5:11],2), 255//31 * int(RGB[1][11:16],2)])
        A[1] = 255
        if COLOR0 > COLOR1:
            RGB[2] = (2 * RGB[0] + RGB[1])//3
            A[2] = 255
            RGB[3] = (2 * RGB[1] + RGB[0])//3
            A[3] = 255
        else:
            RGB[2] = (RGB[0] + RGB[1])//2
            A[2] = 255
            RGB[3] = (2 * RGB[1] + RGB[0])//3
            A[3] = 0
        texel_idx = struct.unpack(">I", subtile_data[4:8])
        texel_idx = bin(texel_idx[0])[2:].zfill(32)
        for y in range(0,4):
            for x in range(0,4):
                idx = int(texel_idx[(x*2)+(y*8):((x+1)*2)+(y*8)],2)
                try:
                    app = np.append(RGB[idx], A[idx])
                    self.image[cur_y+y_offset+y][cur_x+x_offset+x] = app
                except IndexError:
                    pass

    def cmpr_tile(self, tile_data, cur_x, cur_y):
        # tile_data : 32B, 4 sub-tiles, 64 texels
        # handle sub-tiles
        x_offset = 0
        y_offset = 0
        for subtile_i in range(0,4):
            subtile_data = tile_data[subtile_i*8:(subtile_i+1)*8]
            self.cmpr_subtile(subtile_data, x_offset, y_offset, cur_x, cur_y)
            x_offset = x_offset + 4
            if x_offset >= 8:
                x_offset = 0
                y_offset = 4

    def cmpr(self, data):
        cur_x = 0
        cur_y = 0
        bytes_read = 0
        while bytes_read < len(data):
            tile_data = data[bytes_read:bytes_read+32]
            self.cmpr_tile(tile_data, cur_x, cur_y)
            cur_x = cur_x + 8
            if cur_x >= self.image_size[0]:
                cur_x = 0
                cur_y = cur_y + 8
            bytes_read += 32

    def make_texture(self):
        if any(v is None for v in [self.name, self.format, self.image_size,
                                   self.image]):
            print("TXTR data is missing sections")
            return None
        img = {}
        img["name"] = self.name
        img["data"] = self.image
        return img
