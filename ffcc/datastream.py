# Copyright (c) BlackFurniture.
# See LICENSE for details.

from io import BytesIO
import struct

class DataStream:
    def __init__(self, fp):
        self.fp = fp
        self.write = self.fp.write
        self.seek = self.fp.seek
        self.read = self.fp.read
        self.close = self.fp.close
        self.tell = self.fp.tell
        self.readinto = self.fp.readinto

def add_method(name, f):
    f = struct.Struct(f)
    def read_meth(self):
        v, = f.unpack(self.read(f.size))
        return v
    def write_meth(self, v):
        self.write(f.pack(v))
    setattr(DataStream, 'read_%s' % name, read_meth)
    setattr(DataStream, 'write_%s' % name, write_meth)

types = (
    ('int8', 'b', True),
    ('int16', 'h', True),
    ('int32', 'i', True),
    ('int64', 'q', True),
    ('f32', 'f', False),
    ('f64', 'd', False)
 )

for name, f, has_unsigned in types:
    for name_post, f_suffix in (('', '>'), ('_le', '<')):
        add_method(name + name_post, f_suffix + f)
        if has_unsigned:
            add_method('u' + name + name_post, f_suffix + f.upper())