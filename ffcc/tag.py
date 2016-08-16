# Copyright (c) BlackFurniture.
# See LICENSE for details.

# tag.py partly based on ffccp:
# https://github.com/drewler/ffccp

import struct
import sys
import re
import string
from ffcc.datastream import DataStream
import pycollada

UPPERCASE = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ '

class InvalidTag(Exception):
    pass

class InvalidNullTag(Exception):
    pass

class NotDefault:
    pass

def get_handler(meth):
    class NewHandler(Tag):
        def __init__(self, fp, parent):
            self.read = lambda x: meth(parent, self, x)
            super().__init__(fp, parent)
    return NewHandler

class HandlerMetaclass(type):
    def __new__(cls, name, bases, namespace, **kw):
        ret = type.__new__(cls, name, bases, namespace, **kw)
        handlers = {}
        if not ret.handlers or isinstance(ret.handlers, dict):
            return ret
        for handler in ret.handlers:
            if isinstance(handler, bytes):
                name = handler.decode().strip().lower()
                meth = getattr(ret, 'read_%s' % name)
                handlers[handler] = get_handler(meth)
            else:
                handlers[handler.tag] = handler
        ret.handlers = handlers
        return ret

NULL_TAG = b'\x00\x00\x00\x00'

class Tag(metaclass=HandlerMetaclass):
    offset = None
    size = None
    subtags = None
    tag_data = None
    handlers = None
    read = None
    fp_list = None
    valid_tags = [
        b'ATRB', b'BANK', b'BINF', b'BUMP', b'CHD ', b'CHM ', b'COLR', b'DGRP',
        b'DLHD', b'DLST', b'DYN ', b'FMT ', b'FUR ', b'IMAG', b'INFO', b'KEY ',
        b'MATL', b'MESH', b'MIDX', b'MNAM', b'MSET', b'MSST', b'NAME', b'NODE',
        b'NORM', b'NSET', b'ONE ', b'PARM', b'QUAN', b'SCEN', b'SEQ ', b'SIZE',
        b'SKIN', b'TANM', b'TAST', b'TEX ', b'TFRM', b'TIDX', b'TSET', b'TXTR',
        b'UV  ', b'VERT', b'PALT', b'TWO ', b'NAM2', b'SCAL', b'PIDX', b'FACE',
        b'RMIN', b'SPHE', b'NIDX', b'NAM3', b'VSET', b'DSET', b'OTM ', b'OCTM',
        b'MSID', b'TRNS', b'BOBJ', b'LTST', b'SLIT', b'ID  ', b'LDAT', b'ANIM',
        b'SDST', b'PRIO', b'PLIT', b'GEOM', b'MJUN', b'FRAM', b'TSCL', b'TSDT',
        b'MFRM', b'MKEY', b'WATR', b'ASET', b'TYPE', b'OBJN', b'OBJ ', b'NODN',
        b'TREE', b'BOND', b'CHLD', b'TRAN', b'ROT ', b'NBT ', b'AMBI', b'HIT ',
        b'HITV', b'HITF', NULL_TAG
    ]
    subtag_tags = [
        b'TXTR', b'CHM ', b'MSST', b'MESH', b'DLHD', b'MSET', b'MATL', b'SKIN',
        b'DGRP', b'NSET', b'DYN ', b'NODE', b'TEX ', b'SCEN', b'TSET', b'MESH',
        b'VSET', b'DSET', b'OTM ', b'SLIT', b'PLIT', b'ASET', b'OCTM', b'TSCL',
        b'TREE', b'ANIM'
    ]

    def __init__(self, fp=None, parent=None):
        self.parent = parent
        if not fp:
            return
        self.read_stream(fp)

    @classmethod
    def read_files(cls, files):
        tag = cls()
        for name in files:
            fp = DataStream(open(name, 'rb'))
            tag.read_stream(fp)
            fp.close()
        return tag

    def read_stream(self, fp):
        self.offset = fp.tell()
        self.tag = fp.read(4)
        self.size = fp.read_uint32()
        unknown = fp.read(8)
        if self.tag not in self.valid_tags:
            raise InvalidTag('%s (%s)' % (repr(self.tag), self.offset))
        if self.subtags is None:
            self.init()
            self.subtags = []
        start_offset = fp.tell()
        if self.read:
            self.read(fp)
        elif self.handlers:
            self.read_children(fp)
        else:
            self.read_auto(fp)

        read_size = fp.tell() - start_offset
        if self.size != read_size:
            raise InvalidTag('expected to read %s, read %s. %s (%s)'
                             % (self.size, read_size,
                                repr(self.tag), self.offset))
        self.align(fp, 16)

    def init(self):
        pass

    def read_children(self, fp):
        offset = fp.tell()
        while fp.tell() - offset < self.size:
            test = fp.tell()
            tag = fp.read(4)
            fp.seek(test)
            if tag == NULL_TAG:
                Tag(fp, self)
            else:
                try:
                    klass = self.handlers[tag]
                except KeyError as e:
                    print('Could not find %s in %s' % (tag, self.tag))
                    raise e
                self.subtags.append(klass(fp, self))
        return self.subtags

    def read_auto(self, fp):
        offset = fp.tell()
        has_children = self.tag in self.subtag_tags
        if has_children:
            first = True
            end = offset + self.size
            try:
                while True:
                    if fp.tell() >= end:
                        break
                    before_offset = fp.tell()
                    new_tag = Tag(fp, self)
                    if new_tag.tag != NULL_TAG:
                        self.subtags.append(new_tag)
                    first = False
            except InvalidTag as e:
                fp.seek(before_offset)
                if not first:
                    raise e
                print('not reading children for', self.tag, e.args[0])
                has_children = False

        if not has_children:
            self.tag_data = fp.read(self.size)
            test_tag = self.tag_data[:4]
            if test_tag.strip():
                for c in test_tag:
                    if c not in UPPERCASE:
                        break
                else:
                    print('probably tag with subtags:', self.tag, test_tag)

    def align(self, fp, to):
        fp.seek(int(fp.tell() + to - 1
                             - (fp.tell() - 1) % to))

    def get_all(self, klass):
        for tag in self.subtags:
            if tag.tag != klass.tag:
                raise NotImplementedError('unexpected tag')
        return self.subtags

    def get_list(self, klass):
        return [tag for tag in self.subtags if tag.tag == klass.tag]

    def get(self, klass, default=NotDefault):
        ret = None
        for tag in self.subtags:
            if tag.tag != klass.tag:
                continue
            if ret:
                raise KeyError('expected single tag')
            ret = tag
        if ret is None:
            if default is NotDefault:
                raise KeyError('no tag found')
            else:
                ret = default
        return ret

    def get_repr_lines(self, lines=None, prefix=''):
        if self.tag == b'\x00\x00\x00\x00':
            return
        if lines is None:
            lines = []
        text = prefix + '- ' + self.tag.decode()
        if not self.subtags:
            text += ' (size %s, pos %s)' % (self.size, self.offset)
        lines.append(text)
        for tag in self.subtags:
            tag.get_repr_lines(lines, prefix + '    ')
        return lines

    def __repr__(self):
        return '\n'.join(self.get_repr_lines())

def create_name_tag(tag_name, dst='name'):
    class NameTag(Tag):
        tag = tag_name
        def read(self, fp):
            name = fp.read(self.size).decode()
            if name[-1] != '\x00':
                print(name)
                raise NotImplementedError()
            name = name[:-1]
            setattr(self.parent, dst, name)
    return NameTag

NameTag = create_name_tag(b'NAME')

class TSCLTag(Tag):
    tag = b'TSCL'
    handlers = (b'TSDT',)

    def read_tsdt(self, tag, fp):
        self.tsdt = fp.read(12)

class MaterialTag(Tag):
    tag = b'MATL'
    handlers = (b'TIDX', NameTag, b'ATRB', b'BUMP', b'FUR ', TSCLTag,
                b'WATR')

    def read_bump(self, tag, fp):
        print(repr(fp.read(28)))

    def read_fur(self, tag, fp):
        print(repr(fp.read(16)))

    def read_watr(self, tag, fp):
        self.watr = fp.read(24)

    def read_tidx(self, tag, fp):
        indices = [fp.read_uint32() for _ in range(tag.size//4)]
        self.index = None
        self.bump_index = None
        try:
            self.index = indices.pop()
            self.bump_index = indices.pop()
        except IndexError:
            pass
        if indices:
            print('texture indices remaining:', indices)

    def read_atrb(self, tag, fp):
        self.attributes = fp.read(16)

class SkinTag(Tag):
    tag = b'SKIN'
    handlers = (b'NODE', b'ONE ', b'TWO ', b'RMIN')

    def read_node(self, tag, fp):
        self.nodes.append(fp.read_uint32())

    def read_one(self, tag, fp):
        self.one = fp.read(tag.size)

    def read_two(self, tag, fp):
        self.two = fp.read(tag.size)

    def read_rmin(self, tag, fp):
        self.rmin = fp.read(tag.size)

    def init(self):
        self.nodes = []

class FaceTag(Tag):
    tag = b'DLHD'
    handlers = (b'DLST',)

    def read_dlst(self, tag, fp):
        dlst = []
        end = fp.tell() + tag.size

        a = fp.read_uint8()
        material = fp.read_uint8()
        start = fp.read(14)
        if a != 0 or start != b'\x00' * 14:
            print(a, start, fp.tell())
            raise NotImplementedError()
        self.align(fp, 32)
        while fp.tell() < end:
            typ = fp.read_uint8()
            if typ == 0:
                break
            count = fp.read_uint16()
            # - 0x92 and 0x9a come up in conjunction
            #   0x92 "fills out" 0x9a
            if typ in (0x98, 0x99, 0x90, 0x91):
                size = 4
            elif typ in (0x92, 0x9A):
                size = 5
            else:
                print(typ, fp.tell())
                raise NotImplementedError()
            data = []
            for _ in range(count):
                v = []
                for _ in range(size):
                    v.append(fp.read_uint16())
                data.append(v)

            if size == 4:
                pass
            elif size == 5:
                pass
                # for v in data:
                #     if v[2] != 0:
                #         raise NotImplementedError()
            dlst.append((typ, data))
        self.align(fp, 32)
        self.items.append((material, dlst))

    def make_obj_dlst(self, dlst, offsets):
        noff = offsets['normals']+1
        voff = offsets['vertices']+1
        toff = offsets['texcoords']+1
        fobj = ''
        for (typ, data) in dlst:
            if typ in (0x98, 0x99, 0x9a):
                for i in range(1, len(data)-1):
                    v = data[i - (i % 2)]
                    fc0 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    v = data[i - ((i + 1) % 2)]
                    fc1 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    v = data[i+1]
                    fc2 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    fobj = ''.join([fobj, 'f %s %s %s\n' % (fc2, fc1, fc0)])
            elif typ in (0x90, 0x91, 0x92):
                for i in range(0, len(data)-2, 3):
                    v = data[i]
                    fc0 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    v = data[i+1]
                    fc1 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    v = data[i+2]
                    fc2 = '%i/%i/%i' % (v[0]+voff, v[3]+toff, v[1]+noff)
                    fobj = ''.join([fobj, 'f %s %s %s\n' % (fc2, fc1, fc0)])
            else:
                raise Exception('Unknown lst type: %s\n' % typ)
        return fobj

    def make_obj(self, materials, offsets={}):
        dobj = ''
        name = getattr(self.parent, 'name', None)
        if name is None:
            parent_index = self.parent.parent.subtags.index(self.parent)
            name = 'noname_dlst_%s' % parent_index
        for index, dlst in enumerate(self.items):
            material, dlst = dlst
            dobj += 'usemtl %s\n' % materials[material]
            dobj = ''.join([dobj, 'g %s.%i\n' % (name, index),
                            self.make_obj_dlst(dlst, offsets)])
        return dobj

    def init(self):
        self.items = []

def create_vertex_tag(tag_name, dst, count, div=100.0):
    class VertexTag(Tag):
        tag = tag_name

        def read(self, fp):
            if self.tag == b'VERT':
                parent = self.parent
                next_parent = parent.parent
                if parent.tag == b'VSET' or next_parent.tag == b'SCEN':
                    self.read_float(fp, 100)
                    return
            elif self.tag == b'UV  ':
                # if self.parent.parent == b'SCEN':
                # return
                self.read_float(fp, 1)
                return
            data = []
            for i in range(self.size // 2 // count):
                v = []
                for _ in range(count):
                    v.append(fp.read_int16() / div)
                data.append(v)
            setattr(self.parent, dst, data)

        def read_float(self, fp, div):
            data = []
            for i in range(self.size // 4 // count):
                v = []
                for _ in range(count):
                    f = fp.read_f32()
                    v.append(f / div)
                data.append(v)
            setattr(self.parent, dst, data)

    return VertexTag

class PARMTag(Tag):
    tag = b'PARM'

    def read(self, fp):
        self.data = fp.read(48)

class DYNTag(Tag):
    tag = b'DYN '
    handlers = (PARMTag,)

class DGRPTag(Tag):
    tag = b'DGRP'
    handlers = (DYNTag,)

class MeshTag(Tag):
    tag = b'MESH'
    # NBT, HIT only in stg000
    handlers = (b'INFO', b'COLR', b'NIDX', b'NBT ',
                create_name_tag(b'MNAM'),
                create_vertex_tag(b'VERT', 'vertices', 3),
                create_vertex_tag(b'NORM', 'normals', 3),
                create_vertex_tag(b'UV  ', 'texcoords', 2, 4096),
                SkinTag, FaceTag)

    def read_info(self, tag, fp):
        self.material = fp.read_uint32()
        print(self.material)
        self.info = fp.read(48-4)

    def read_nbt(self, tag, fp):
        self.nbt = True

    def read_colr(self, tag, fp):
        self.color = fp.read(tag.size)

    def read_skin(self, tag, fp):
        self.skin = fp.read(tag.size)

    def read_nidx(self, tag, fp):
        self.nidx = fp.read(4)

    def make_obj(self, typ, data):
        obj = ''
        for vertex in data:
            obj += typ + ' ' + ' '.join(str(item) for item in vertex) + '\n'
        return obj

    def make_mesh_2(self, *arg, **kw):
        if not self.vertices or not self.normals or not self.texcoords:
            raise ValueError()
        obj = {}
        if self.tag == b'VSET':
            name = getattr(self, 'name', None)
            if name is None:
                name = 'map_vset_%s' % self.parent.subtags.index(self)
        else:
            name = self.name
        obj['name'] = name
        obj['vertices'] = self.make_obj('v', self.vertices)
        obj['normals'] = self.make_obj('vn', self.normals)
        obj['uv'] = self.make_obj('vt', self.texcoords)
        fobj = ''
        for i in range(len(self.vertices)):
            fobj += 'f %s %s %s\n' % (i, i, i)
        obj['faces'] = fobj
        obj['faces'] = face_tag.make_obj(*arg, **kw)
        return obj

    def make_mesh(self, *arg, **kw):
        if not self.vertices or not self.normals or not self.texcoords:
            raise ValueError()
        obj = {}
        name = getattr(self, 'name', None)
        if name is None:
            name = 'noname_mesh_%s' % self.parent.subtags.index(self)
        mesh = pycollada.Collada()
        vertices = self.make_float_source(name, 'pos', self.vertices)
        normals = self.make_float_source(name, 'norm', self.normals)
        uv = self.make_float_source(name, 'uv', self.texcoords)4
        geom = geometry.Geometry(mesh, "mesh1-geometry", "mesh1-geometry",
                                 [m1position_src, m1normal_src, m1uv_src])
        if self.tag == b'VSET':
            index = self.parent.subtags.index(self)
            try:
                face_tag = self.parent.subtags[index+1].get(FaceTag)
            except IndexError:
                return None
        else:
            face_tag = self.get(FaceTag)
        obj['faces'] = face_tag.make_obj(*arg, **kw)
        return obj

    @staticmethod
    def make_meshes(meshes, materials, merge=False):
        objs = []
        offsets = {'vertices': 0, 'normals': 0, 'texcoords': 0}
        for mesh in meshes:
            new_obj = mesh.make_mesh(materials, offsets)
            if new_obj is None:
                continue
            objs.append(new_obj)
            if not merge:
                continue
            for k in offsets:
                offsets[k] += len(getattr(mesh, k))
        return objs

class MeshSetTag(Tag):
    tag = b'MSST'
    handlers = (MeshTag,)

class MaterialSetTag(Tag):
    tag = b'MSET'
    handlers = (MaterialTag,)

class PARMTag2(Tag):
    tag = b'PARM'

    def read(self, fp):
        self.data = fp.read(16)

class DYNTag2(Tag):
    tag = b'DYN '
    handlers = (PARMTag2,)

class NodeTag(Tag):
    tag = b'NODE'
    handlers = (b'INFO', b'TFRM', b'BINF', b'MIDX',
                create_name_tag(b'NAME'),
                create_name_tag(b'NAM2', 'name2'),
                create_name_tag(b'NAM3', 'name3'),
                DYNTag2)

    def read_midx(self, tag, fp):
        self.data = fp.read(4)

    def read_info(self, tag, fp):
        self.data = fp.read(16)

    def read_tfrm(self, tag, fp):
        self.transform = fp.read(48)

    def read_binf(self, tag, fp):
        self.binf = fp.read(16)

class NodeSetTag(Tag):
    tag = b'NSET'
    handlers = (NodeTag,)

def create_ignore_tag(name):
    class IgnoreTag(Tag):
        tag = name
        def read(self, fp):
            self.data = fp.read(self.size)
    return IgnoreTag

class ModelTag(Tag):
    tag = b'CHM '
    handlers = (b'INFO', b'QUAN', b'BANK', b'TAST',
                MaterialSetTag, MeshSetTag,
                DGRPTag, NodeSetTag,
                create_ignore_tag(b'SCAL'),
                create_ignore_tag(b'NODE'))

    def read(self, fp):
        self.read_children(fp)
        try:
            tag = self.get(ModelScenTag)
        except KeyError:
            tag = self
        self.materials = tag.get(MaterialSetTag).subtags
        try:
            self.meshes = tag.get(MeshSetTag).subtags
        except KeyError:
            self.meshes = []
        try:
            self.nodes = tag.get(NodeSetTag).subtags
        except KeyError:
            self.nodes = []

    def read_scal(self, tag, fp):
        # only in SCEN!
        self.scale = fp.read(tag.size)

    def read_quan(self, tag, fp):
        self.quan = fp.read(32)

    def read_info(self, tag, fp):
        self.info = fp.read(32)

    def read_bank(self, tag, fp):
        self.bank = fp.read(tag.size)

    def read_tast(self, tag, fp):
        self.tast = fp.read(tag.size)
        # READ CHILDREN!
        print(repr(self.tast))

    def get_meshes(self):
        return self.meshes

class ModelScenTag(ModelTag):
    tag = b'SCEN'

ModelTag.handlers[ModelScenTag.tag] = ModelScenTag

class VertexSetTag(MeshTag):
    tag = b'VSET'

class DisplayListSetTag(Tag):
    tag = b'DSET'
    handlers = (FaceTag,)

class MapModelTag(Tag):
    tag = b'MESH'
    handlers = (VertexSetTag, DisplayListSetTag)

    def get_meshes(self):
        return self.get_list(VertexSetTag)

class SlitTag(Tag):
    tag = b'SLIT'
    handlers = (b'LDAT',)

    def read_ldat(self, tag, fp):
        self.ldat = fp.read(58)
        print(repr(self.ldat))

class PlitTag(Tag):
    tag = b'PLIT'
    handlers = (b'MJUN', b'MFRM', b'MKEY', b'LDAT')

    def read_mjun(self, tag, fp):
        self.mjun = fp.read(4)

    def read_mfrm(self, tag, fp):
        self.mfrm = fp.read(12)

    def read_mkey(self, tag, fp):
        self.mkey = fp.read(40)

    def read_ldat(self, tag, fp):
        self.ldat = fp.read(38)

class MapMaterialNodeTag(Tag):
    tag = b'NODE'
    handlers = (b'PIDX', b'MSID', b'TFRM', b'TRNS', b'BOBJ', b'LTST',
                SlitTag, PlitTag, b'ID  ', b'ANIM', b'SDST', b'PRIO',
                b'GEOM', b'AMBI')

    def read_pidx(self, tag, fp):
        self.pidx = fp.read(tag.size)
        print(self.pidx.hex())

    def read_msid(self, tag, fp):
        self.msid = fp.read(16)

    def read_ambi(self, tag, fp):
        self.ambi = fp.read(4)

    def read_tfrm(self, tag, fp):
        self.trfm = [[fp.read_float() for _ in range(3)] for _ in range(3)]

    def read_trns(self, tag, fp):
        self.trns = fp.read(16)

    def read_bobj(self, tag, fp):
        self.bobj = fp.read(16)

    def read_ltst(self, tag, fp):
        self.ltst = fp.read(8)

    def read_id(self, tag, fp):
        self.node_id = fp.read(4)

    def read_anim(self, tag, fp):
        self.anim = fp.read(tag.size)

    def read_sdst(self, tag, fp):
        self.sdst = fp.read(8)

    def read_prio(self, tag, fp):
        self.prio = fp.read(4)

    def read_geom(self, tag, fp):
        self.geom = fp.read(32)

class AnimationNodeTag(Tag):
    tag = b'NODE'
    handlers = (b'NIDX', b'TRAN', b'ROT ', b'SCAL')

    def read_nidx(self, tag, fp):
        self.nidx = fp.read_uint32()

    def read_tran(self, tag, fp):
        self.tran = fp.read(tag.size)

    def read_rot(self, tag, fp):
        self.rot = fp.read(tag.size)

    def read_scal(self, tag, fp):
        self.scal = fp.read(16)

class MapMaterialAnimationTag(Tag):
    tag = b'ANIM'
    handlers = (b'FRAM', AnimationNodeTag)

    def read_fram(self, tag, fp):
        self.frma = fp.read(8)

class AnimationTag(Tag):
    tag = b'TANM'

    def read(self, fp):
        self.data = fp.read(self.size)

class AnimationSetTag(Tag):
    tag = b'ASET'
    handlers = (AnimationTag,)

class TreeNodeTag(Tag):
    tag = b'NODE'
    handlers = (b'OBJ ', b'BOND', b'CHLD')

    def read_obj(self, tag, fp):
        self.obj = fp.read(6)

    def read_bond(self, tag, fp):
        self.bond = fp.read(24)

    def read_chld(self, tag, fp):
        self.chld = fp.read(16)

class TreeTag(Tag):
    tag = b'TREE'
    handlers = (TreeNodeTag,)

class OCTMTag(Tag):
    tag = b'OCTM'
    handlers = (b'TYPE', b'OBJN', b'INFO', b'NODN', TreeTag)

    def read_type(self, tag, fp):
        self.typ = fp.read(2)

    def read_objn(self, tag, fp):
        self.objn = fp.read(2)

    def read_info(self, tag, fp):
        self.info = fp.read(4)

    def read_nodn(self, tag, fp):
        self.nodn = fp.read(2)

from ffcc.texture import TextureTag

class TextureSetTag(Tag):
    tag = b'TSET'
    handlers = (TextureTag,)

    def get_set(self):
        return self

    def make_textures(self):
        imgs = []
        for texture in self.subtags:
            tex = texture.make_texture()
            if tex != None:
                imgs.append(tex)
        return imgs

    def make_pil_textures(self):
        from PIL import Image
        imgs = []
        for img in self.make_textures():
            pil_image = Image.fromarray(img["data"], 'RGBA')
            pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
            imgs.append((img['name'], pil_image))
        return imgs

class TextureSceneTag(Tag):
    tag = b'SCEN'
    handlers = (TextureSetTag,)

class TextureFileTag(Tag):
    tag = b'TEX '
    handlers = (TextureSceneTag,)

    def get_set(self):
        return self.get(TextureSceneTag).get(TextureSetTag)

class HITTag(Tag):
    tag = b'HIT '
    handlers = (b'HITV', b'HITF')

    def read_hitv(self, tag, fp):
        self.hitv = fp.read(tag.size)

    def read_hitf(self, tag, fp):
        self.hitf = fp.read(tag.size)

class MaterialSceneTag(Tag):
    tag = b'SCEN'
    # TextureSetTag, MeshTag, HITTag only in stg000
    handlers = (MapMaterialNodeTag, MapMaterialAnimationTag, MaterialSetTag,
                AnimationSetTag, TextureSetTag, MeshTag, HITTag)

class MapMaterialTag(Tag):
    tag = b'OTM '
    handlers = (MaterialSceneTag, OCTMTag)

    def get_scene(self):
        return self.get(MaterialSceneTag)

    def read(self, fp):
        self.read_children(fp)
        self.materials = self.get_scene().get(MaterialSetTag).subtags
