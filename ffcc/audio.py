# Copyright (c) BlackFurniture.
# See LICENSE for details.

from ffcc._opt import lib, ffi
from ffcc.datastream import DataStream
import wave
import os
import math
from io import BytesIO

def get_sample_rate(key):
    return int(round(32000 * 2**(key / 0x1000000 / 12)))

def get_nibbles_end(nibbles):
    # from Dolphin
    t = nibbles & 15
    if t == 0:
        step_size_bytes = 1
    elif t == 1: 
        step_size_bytes = 0
    else:
        step_size_bytes = 2
    return nibbles + step_size_bytes - 1

def nibbles_to_samples(nibbles):
    samples = (nibbles // 16) * 14
    rem = nibbles % 16
    if rem:
        if rem == 1:
            # can't have half frame header
            raise ValueError()
        samples += rem - 2
    return samples

def create_header(coefs, pred_scale, yn1, yn2):
    header = ffi.new('struct DSPHeader*')
    header.pred_scale = pred_scale
    header.yn1 = yn1
    header.yn2 = yn2
    for i in range(8):
        for ii in range(2):
            header.coefs[i][ii] = coefs[i * 2 + ii]
    return header

def convert_pcm16(headers, num_nibbles, data, loop=None):
    channels = len(headers)
    loop_samples = 0
    num_samples = total_samples = nibbles_to_samples(num_nibbles)
    if loop is not None:
        loop_pos, loop_end, pred_scale, yn1, yn2 = loop
        total_samples += num_samples - nibbles_to_samples(loop_pos)

    out = ffi.new('int16_t[]', total_samples*channels)
    for i in range(len(out)):
        out[i] = 0x7FFF
    channel_size = len(data) // channels

    for channel in range(channels):
        lib.DecodeADPCM(data + channel*channel_size,
                        out + channel, num_nibbles,
                        0, headers[channel], channels)

    if loop is not None:
        out_off = num_samples * channels
        for channel in range(channels):
            header = headers[channel]
            header.yn1 = yn1
            header.yn2 = yn2
            header.pred_scale = pred_scale
            lib.DecodeADPCM(data + channel*channel_size,
                            out + out_off + channel, num_nibbles,
                            loop_pos, headers[channel], channels)

    return out

def convert_wave(headers, num_nibbles, sample_rate, data, loop=None):
    fp = BytesIO()
    channels = len(headers)
    out = convert_pcm16(headers, num_nibbles, data, loop)

    wav = wave.open(fp, 'wb')
    wav.setnchannels(channels)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)
    wav.writeframes(ffi.buffer(out))
    wav.close()
    return fp.getvalue()

def read_string(fp):
    size = fp.read_uint8()
    s = fp.read(size).decode('ascii')
    return s

class SampleInfo:
    def __init__(self, fp, instrument):
        self.instrument = instrument
        self.offset = fp.tell()
        something1 = fp.read_uint32()
        something2 = fp.read_uint32()
        if fp.read_uint32() != 0:
            raise ValueError()
        if fp.read_uint32() != 0:
            raise ValueError()
        self.name = read_string(fp)
        left = self.offset + 48 - fp.tell()
        for c in fp.read(left):
            if c != 0:
                raise ValueError()

class InstrumentInfo:
    def __init__(self, fp):
        start = fp.tell()
        offset = fp.read_uint32()
        sample_count = fp.read_uint32()
        if fp.read_uint32() != 0:
            raise ValueError()
        if fp.read_uint32() != 0:
            raise ValueError()
        self.name = read_string(fp)
        left = start + 48 - fp.tell()
        for c in fp.read(left):
            if c != 0:
                raise ValueError()
        fp.seek(offset)
        self.samples = [SampleInfo(fp, self) for _ in range(sample_count)]

class InstrumentInfoFile:
    def __init__(self, fp):
        if fp.read(4) != b'WIv2':
            raise ValueError()
        if fp.read_uint32() != 0:
            raise ValueError()
        instrument_count = fp.read_uint32()
        samples = fp.read_uint32()
        start_offset = fp.tell()
        self.instruments = []
        self.samples = []
        for i in range(instrument_count):
            fp.seek(start_offset + i*48)
            instrument = InstrumentInfo(fp)
            self.instruments.append(instrument)
            for sample in instrument.samples:
                self.samples.append(sample)
        self.samples = list(sorted(self.samples, key=lambda x: x.offset))

class WaveSample:
    is_null = False
    stereo = False

    def __init__(self, fp, info, data_offset):
        self.info = info
        start = fp.tell()
        for c in fp.read(96):
            if c != 0:
                break
        else:
            self.is_null = True
            return
        fp.seek(start)
        self.header = fp.read(4)
        fp.seek(start)
        if fp.read_uint8() != 0: # padding
            raise ValueError()
        frame_marker = fp.read_uint8()
        if frame_marker not in (0, 1):
            raise ValueError()
        frame_marker = frame_marker == 1
        flags = fp.read_uint8()
        if flags not in (0, 1, 2):
            raise ValueError()
        self.first = (flags & 1) != 0
        self.last = (flags & 2) != 0
        if frame_marker and not flags:
            raise ValueError()
        stereo = fp.read_uint8()
        if stereo not in (0, 1):
            raise ValueError()
        self.stereo = stereo == 1
        self.offset = fp.read_uint32()
        self.loop_start = fp.read_int32() # in nibbles
        self.loop_end = fp.read_uint32() # in nibbles
        byte_size = fp.read_uint32() # 8-byte aligned
        if byte_size % 8:
            raise ValueError()

        self.num_nibbles = min(byte_size * 2, get_nibbles_end(self.loop_end))
        self.key = fp.read_int32() # in 8.24 key notation
        self.key_high = fp.read_uint8()
        self.vel_high = fp.read_uint8()
        if self.vel_high != 0x7f:
            # probably velocity range?
            # when something8 == 0, everything else is 0
            raise ValueError()
        self.volume = fp.read_uint8() / 127
        pan = fp.read_uint8()
        if pan == 255:
            pan = 1.0
        elif pan == 128:
            pan = 0
        elif pan in (192, 64, 0):
            pan = 0.5
        elif pan > 127:
            pan = (pan - 128) / 127
        else:
            raise ValueError()

        self.pan = pan
        something3 = fp.read_uint32()
        something4 = fp.read_uint16()
        if something3 != 0 or something4 != 0: # padding?
            raise ValueError()

        self.coefs = [fp.read_int16() for _ in range(16)]
        self.gain = fp.read_int16()
        self.pred_scale = fp.read_int16()
        self.yn1 = fp.read_int16()
        self.yn2 = fp.read_int16()

        self.loop_pred_scale = fp.read_int16()
        self.loop_yn1 = fp.read_int16()
        self.loop_yn2 = fp.read_int16()

        self.adsr_time = [fp.read_uint16() for _ in range(4)]
        self.adsr_volume = [fp.read_uint8() for _ in range(4)]

        if self.adsr_volume[3] != 0x7f:
            # release volume always max
            raise ValueError()
        if self.adsr_time[1] != 0 or self.adsr_volume[1] != 0x7f:
            # decay not used
            raise ValueError()
            # print(adsr_time, adsr_volume, self.get_name())
        if fp.read_uint32() != 0: # probably padding
            raise ValueError()

        fp.seek(data_offset + self.offset)
        self.data = ffi.new('uint8_t[]', byte_size)
        fp.readinto(ffi.buffer(self.data, byte_size))

    def get_base_key(self):
        return 0x3C - (self.key >> 24)

    def get_finetune(self):
        return (self.key & 0xFFFFFF) / 0x1000000

    def get_sample_count(self, loop=True):
        samples = nibbles_to_samples(self.num_nibbles)
        if loop and self.loop_start != -1:
            samples += samples - nibbles_to_samples(self.loop_start)
        return samples

    def get_name(self):
        return getattr(self.info, 'name', 'noname') + '_%s' % self.offset

    def get_wav(self, with_loop=False, is_instrument=True):
        header = create_header(self.coefs, self.pred_scale, self.yn1, self.yn2)
        loop = None
        if self.loop_start != -1 and with_loop:
            loop = (self.loop_start, self.loop_end, self.loop_pred_scale,
                    self.loop_yn1, self.loop_yn2)
        if is_instrument:
            sample_rate = 32000
        else:
            sample_rate = get_sample_rate(self.tone, self.finetune)
        return convert_wave([header], self.num_nibbles, 32000, self.data, loop)

class WaveInstrument:
    info = None
    def __init__(self):
        self.samples = []

    def get_name(self):
        return getattr(self.info, 'name', 'noname') + '_%s' % self.offset

class WaveData:
    info = None
    def __init__(self, fp, filename):
        self.samples = []
        self.instruments = []
        wi_path = os.path.splitext(filename)[0] + '.wi'
        if os.path.isfile(wi_path):
            with open(wi_path, 'rb') as fp2:
                self.info = InstrumentInfoFile(DataStream(fp2))
        size = fp.seek(0, 2)
        fp.seek(0)
        if fp.read(2) != b'WD':
            raise ValueError()
        self.bank_id = fp.read_uint16()
        data_size = fp.read_uint32()
        data_start = size - data_size
        instrument_count = fp.read_uint32() # number of instruments
        sample_count = fp.read_uint32() # number of samples
        if fp.read_uint32() != 0:
            raise ValueError()
        fp.seek(32)
        offsets = [fp.read_uint32() for _ in range(instrument_count)]
        start_offset = ((32 + instrument_count*4) + 31) & ~31
        if start_offset != offsets[0]:
            if offsets[0] == 0:
                print('Skipping...')
                return
            print(start_offset, offsets[0])
            raise ValueError()

        def print_desc(sample):
            attrs = [sample.header.hex()]
            if sample.info:
                attrs.append(sample.info.name)
                attrs.append(sample.info.instrument.name)
            print(*attrs)

        for i, offset in enumerate(offsets):
            def create_sample(fp, i):
                fp.seek(offset + 96 * i)
                sample_info = None
                if self.info:
                    sample_info = self.info.samples[len(self.samples)]
                sample = WaveSample(fp, sample_info, data_start)
                self.samples.append(sample)
                return sample

            if i == len(offsets)-1:
                end = data_start
            else:
                end = offsets[i+1]
            instrument_sample_count = (end - offset) // 96
            instrument = WaveInstrument()
            instrument.offset = offset
            self.instruments.append(instrument)
            ii = 0
            while ii < instrument_sample_count:
                sample = create_sample(fp, ii)
                ii += 1
                channels = [sample]
                if sample.stereo:
                    other_sample = create_sample(fp, ii)
                    ii += 1
                    channels.append(other_sample)
                instrument.samples.append(channels)
            if self.info:
                instrument.info = self.info.instruments[i]
                if len(instrument.info.samples) < ii:
                    print(len(instrument.info.samples), ii,
                              sample_count,
                              instrument_sample_count)
                    raise ValueError()
                elif len(instrument.info.samples) != ii:
                    print('Info data does not match completely')

        if len(self.samples) != sample_count:
            raise ValueError()

from ffcc.midictrl import CTRL_NAMES

class SequenceTrack:
    program = None
    reverb = None
    reverb_depth = None
    has_notes = False

    def __init__(self, parent, index, fp, size):
        self.index = index
        self.parent = parent
        self.track_delta = 0

        self.commands = []
        end = fp.tell() + size
        while True:
            delta = 0
            while True:
                value = fp.read_uint8()
                delta = (delta << 7) + (value & 0x7f)
                if value & 0x80 == 0:
                    break
            self.track_delta += delta
            cmd = fp.read_uint8()
            cmd_name = CTRL_NAMES[cmd]
            ret = getattr(self, 'handle_{0:02x}'.format(cmd))(fp)
            if ret:
                self.commands.append([self.track_delta] + ret + [cmd_name])
            if fp.tell() >= end:
                break

        if self.program is None and self.has_notes:
            print('no program for track', index)

    def handle_0d(self, fp):
        # KeySignature
        arg = fp.read_uint8()
        return ['key_signature', arg]

    def handle_0c(self, fp):
        # TimeSignature
        numer = fp.read_uint8()
        denom = fp.read_uint8()
        if self.parent.time_signature is None:
            self.parent.time_signature = (numer, denom)
        return ['time_signature', numer, denom]

    def handle_08(self, fp):
        # TempoDirect (BPM)
        bpm = fp.read_uint8()
        self.parent.bpm = bpm
        return ['bpm', bpm]

    def handle_00(self, fp):
        # Stop
        return ['end']

    def handle_22(self, fp):
        # VolumeDirect
        volume = fp.read_uint8()
        return ['volume', volume]

    def handle_26(self, fp):
        # PanDirect
        pan = fp.read_uint8()
        return ['pan', pan]

    def handle_0a(self, fp):
        # ReverbDepthDirect
        return ['reverb_depth', fp.read_uint8()]

    def handle_60(self, fp):
        # ReverbOn
        return ['reverb_on']

    def handle_20(self, fp):
        # Wave
        program = fp.read_uint8()
        if self.program is not None and self.program != program:
            raise ValueError()
        self.program = program

    def handle_02(self, fp):
        # WholeLoopStart
        if self.parent.loop_start is not None:
            raise ValueError()
        self.parent.loop_start = self.track_delta

    def handle_11(self, fp):
        # KeyOnNoteVelocity
        key = fp.read_uint8()
        vel = fp.read_uint8()
        self.last_key = key
        self.last_vel = vel
        self.has_notes = True
        return ['note_on', key, vel]

    def handle_18(self, fp):
        # KeyOffSame
        self.has_notes = True
        return ['note_off', self.last_key]

    def handle_12(self, fp):
        # KeyOnNote
        key = fp.read_uint8()
        self.last_key = key
        self.has_notes = True
        return ['note_on', key, self.last_vel]

    def handle_03(self, fp):
        # WholeLoopEnd
        if self.parent.loop_end:
            raise ValueError()
        self.parent.loop_end = self.track_delta

    def handle_10(self, fp):
        # KeyOnSame
        self.has_notes = True
        return ['note_on', self.last_key, self.last_vel]

    def handle_13(self, fp):
        # KeyOnVelocity
        self.has_notes = True
        vel = fp.read_uint8()
        self.last_vel = vel
        return ['note_on', self.last_key, vel]

    def handle_1a(self, fp):
        # KeyOffNote
        key = fp.read_uint8()
        self.has_notes = True
        self.last_key = key
        return ['note_off', key]

    def handle_7f(self, fp):
        # Pass
        return ['pass']

    def handle_5d(self, fp):
        # PitchBendRange
        self.pitch_bend_range = fp.read_uint8()

    def handle_5c(self, fp):
        # PitchBend
        a = fp.read_uint8()
        b = fp.read_uint8()
        val = a + (b << 7) - 0x2000
        semitones = (val * self.pitch_bend_range) / 0x2000
        return ['pitch_bend', semitones]

    def handle_01(self, fp):
        # Sleep
        return ['sleep']

# 0: reverb type
#     1: reverb std
#     2: reverb hi
#     3: delay
#     4: chorus
#     5: reverb hi dpl2
# 1: predelay (in seconds)
# 2: time (in seconds)
# 3: coloration (0-1)
# 4: damping (0-1, filter coef for lpf)
# 5: mix (0-1, output gain)
# 6: crosstalk
EFFECT_NAMES = {
    1: 'ReverbStd',
    2: 'ReverbHi',
    3: 'Delay',
    4: 'Chorus',
    5: 'ReverbHiDpl2'
}

# retrieved from binary
EFFECT_DATA = [(2, 0.01,  1.4,  0.3, 0.7, 1, 0),
               (2, 0.02,  1.8,  0.3, 0.8, 1, 0),
               (2, 0.03,  2.4,  0.3, 0.8, 1, 0),
               (2, 0.04,  2.8,  0.3, 0.8, 1, 0),
               (2, 0.015, 0.85, 0.5, 0.8, 1, 0),
               (2, 0.03,  1.5,  0.5, 0.6, 1, 0),
               (2, 0.04,  2.5,  0.5, 0.4, 1, 0),
               (2, 0.04,  2.5,  0.5, 0.8, 1, 0)]

class SequenceFile:
    loop_start = None
    loop_end = None
    bpm = None
    key_signature = None
    time_signature = None
    reverb_depth = None

    def __init__(self, fp, dummy=None):
        if fp.read(4) != b'BGM ':
            raise ValueError()
        self.self_id = fp.read_uint16()
        self.wd_id = fp.read_uint16()
        self.track_count = fp.read_uint8()
        self.effect = fp.read_uint8()
        padding = fp.read_uint16()
        if padding != 0:
            raise ValueError()
        self.volume = fp.read_uint16() / 127.0
        self.ppqn = fp.read_uint16()

        size = fp.read_uint32() # rounded up
        for c in fp.read(12):
            if c != 0:
                raise ValueError()

        off = fp.tell()
        fp.seek(0, 2)
        size = fp.tell()
        fp.seek(off)
        self.tracks = []
        while fp.tell() < size:
            track_size = fp.read_uint32_le()
            off = fp.tell()
            track_index = len(self.tracks)
            self.tracks.append(SequenceTrack(self, track_index,
                                             fp, track_size))
            if off + track_size != fp.tell():
                raise ValueError()

        loop_count = 0
        for track in self.tracks:
            for commands in track.commands:
                if commands[1] == 'loop_start':
                    loop_count += 1
        if loop_count > 1:
            raise ValueError()

        if len(self.tracks) != self.track_count:
            raise ValueError()

    def get_effect(self):
        effect = EFFECT_DATA[self.effect]
        if effect[0] != 2:
            raise ValueError()
        return {'name': EFFECT_NAMES[effect[0]],
                'predelay': effect[1],
                'time': effect[2],
                'coloration': effect[3],
                'damping': effect[4],
                'mix': effect[5],
                'crosstalk': effect[6]}

class AudioStream:
    def __init__(self, fp):
        magic = fp.read(3)
        if magic != b'STR':
            raise ValueError()
        if fp.read(5) != b'\x00\x00\x00\x00\x00':
            raise ValueError()
        self.size = fp.read_uint32()
        frames = fp.read_uint32()
        loop_pos = fp.read_int32()
        if loop_pos != -1:
            raise ValueError()
        self.key = fp.read_int32() # in 8.24 key notation
        if fp.read_uint16() != 0:
            raise ValueError()
        channels = fp.read_uint16()
        if fp.read_uint16() != 0: # check for != 0 in asm
            raise ValueError()
        if fp.read_uint16() != 127:
            raise ValueError()
        self.channels = []
        self.num_nibbles = frames * 8 * 2
        channel_size = frames * 8
        for _ in range(channels):
            coefs = [fp.read_int16() for _ in range(16)]
            gain = fp.read_int16()
            pred_scale = fp.read_int16()
            yn1 = fp.read_int16()
            yn2 = fp.read_int16()
            loop_pred_scale = fp.read_int16()
            loop_yn1 = fp.read_int16()
            loop_yn2 = fp.read_int16()
            self.channels.append((coefs, gain, pred_scale, yn1, yn2,
                                  loop_pred_scale, loop_yn1, loop_yn2))

        data = ffi.new('uint8_t[]', channel_size*channels)
        fp.seek(4096)
        for i in range(0, channel_size, 4096):
            for channel in range(channels):
                read_size = min(4096, channel_size - i)
                buf = data + i + channel*channel_size
                fp.readinto(ffi.buffer(buf, read_size))
        self.data = data

    def get_wav(self):
        headers = []
        for channel in self.channels:
            coefs, gain, pred_scale, yn1, yn2, *rest = channel
            headers.append(create_header(coefs, pred_scale, yn1, yn2))
        return convert_wave(headers, self.num_nibbles,
                            get_sample_rate(self.key), self.data)

NOTE_NAMES = ['C-', 'C#', 'D-', 'D#', 'E-', 'F-',
              'F#', 'G-', 'G#', 'A-', 'A#', 'B-']

def get_key_name(note):
    octave = note // 12 - 2
    note = note % 12
    ret = '%s%s' % (NOTE_NAMES[note], octave)
    return ret

def to_hex(v, count=2):
    return '{0:0{1}X}'.format(v, count)

from xml.etree import ElementTree

def get_track_lines(track):
    lines = []
    note_column = {}
    volume = 127

    program = track.program
    if program is None:
        if track.has_notes:
            print('program not set', program)
        program = 50
    volume = 127
    pan = 0.5
    last_t = None
    line = None
    current_keys = []
    current_columns = set()
    pitch_bend = None

    lens = len(track.commands)
    commands = list(track.commands)

    from functools import cmp_to_key
    def compare(a, b):
        tt1 = a[0]
        tt2 = b[0]
        if tt1 < tt2:
            return -1
        elif tt1 > tt2:
            return 1
        t1 = a[1]
        t2 = b[1]
        if t1 == t2:
            return 0
        if t1 == 'note_off':
            return -1
        elif t2 == 'note_off':
            return 1
        return 0

    commands.sort(key=cmp_to_key(compare))
    commands += [(None, None, None)]
    for cmd_index, (t, typ, *args, real) in enumerate(commands):
        if t != last_t or t is None:
            if line is not None:
                lines.append((last_t, line))
            if t is None:
                break
            if pitch_bend is not None or len(lines) == 1:
                if pitch_bend is None:
                    pitch_bend = 0
                pitch_bend = int(0x2000 + (pitch_bend / 2.0) * 0x2000)
                if pitch_bend > 0x3fff or pitch_bend < 0:
                    raise ValueError()
                effect = ElementTree.Element('EffectColumn')
                a = pitch_bend // 128
                b = pitch_bend % 128
                ElementTree.SubElement(effect, 'Number').text = to_hex(a)
                ElementTree.SubElement(effect, 'Value').text = to_hex(b)
                effect_columns.insert(0, effect)
                note = ElementTree.Element('NoteColumn')
                prog = to_hex(program)
                ElementTree.SubElement(note, 'Instrument').text = prog
                ElementTree.SubElement(note, 'Panning').text = 'M1'
                note_columns.append(note)
                pitch_bend = None
            last_t = t
            line = ElementTree.Element('Line')
            note_columns = ElementTree.SubElement(line, 'NoteColumns')
            effect_columns = ElementTree.SubElement(line, 'EffectColumns')
            used_columns = set()

        def get_note_column(in_key, create=True):
            if create:
                i = 0
                while i in current_columns or i in used_columns:
                    i += 1
                current_columns.add(i)
                current_keys.append((in_key, i))
            else:
                for search_key, i in current_keys:
                    if search_key == in_key:
                        break
                else:
                    raise IndexError()
            used_columns.add(i)
            while len(note_columns) < i+1:
                ElementTree.SubElement(note_columns, 'NoteColumn')
            return (i, in_key, note_columns[i])

        def create_effect_column(number, value):
            effect = ElementTree.SubElement(effect_columns, 'EffectColumn')
            ElementTree.SubElement(effect, 'Number').text = number
            ElementTree.SubElement(effect, 'Value').text = to_hex(value)

        if typ == 'note_on':
            key, vel = args
            # print('note on:', key, current_keys, current_columns)
            column = get_note_column(key)[2]
            ElementTree.SubElement(column, 'Note').text = get_key_name(key)
            ElementTree.SubElement(column, 'Instrument').text = to_hex(program)
            ElementTree.SubElement(column, 'Volume').text = to_hex(vel)
        elif typ == 'note_off':
            key, = args
            last_used_columns = used_columns.copy()
            while True:
                try:
                    i, key, column = get_note_column(key, create=False)
                except IndexError:
                    break
                if i in last_used_columns:
                    for other_index, cmd in enumerate(track.commands):
                        print(cmd, program, other_index, cmd_index)
                    raise ValueError()
                ElementTree.SubElement(column, 'Note').text = 'OFF'
                current_keys.remove((key, i))
                current_columns.remove(i)
        elif typ == 'volume':
            create_effect_column('0L', (args[0] * 255) // 127)
        elif typ == 'bpm':
            create_effect_column('ZT', args[0])
        elif typ == 'pan':
            create_effect_column('0P', (args[0] * 255) // 127)
        elif typ == 'reverb_on':
            create_effect_column('10', 1)
        elif typ == 'reverb_depth':
            create_effect_column('18', (args[0] * 255) // 127)
        elif typ == 'pitch_bend':
            pitch_bend = args[0]
        elif typ in ('end', 'key_signature', 'time_signature',
                     'sleep', 'pass'):
            continue
        else:
            print('typ not handled:', t, typ, args)
            continue

    return (last_t, lines)

def convert_bgm(bgm, outdir):
    with open(bgm, 'rb') as fp:
        bgm_data = SequenceFile(DataStream(fp))

    wd_path = os.path.join(os.path.dirname(bgm), '..', 'Wave',
                           'wave0%s.wd' % bgm_data.wd_id)
    with open(wd_path, 'rb') as fp:
        wd_data = WaveData(DataStream(fp), wd_path)

    # create XRNS
    import zipfile
    xrns_path = os.path.join(os.path.dirname(__file__), 'data', 'test.xrns')
    with zipfile.ZipFile(xrns_path, 'r') as fp:
        with fp.open('Song.xml', 'r') as fp2:
            song = fp2.read()

    from copy import deepcopy

    tree = ElementTree.fromstring(song)
    # general song info
    song_data = tree.find('GlobalSongData')
    song_data.find('Octave').text = '3'
    song_data.find('BeatsPerMin').text = str(bgm_data.bpm)
    song_data.find('LinesPerBeat').text = str(bgm_data.ppqn)
    num, denom = bgm_data.time_signature
    song_data.find('SignatureNumerator').text = str(num)
    song_data.find('SignatureDenominator').text = str(denom)

    effect_data = bgm_data.get_effect()

    instruments = tree.find('Instruments')
    instrument_template = instruments[0]
    instruments.remove(instrument_template)
    generator_template = instrument_template.find('SampleGenerator')
    generator_samples = generator_template.find('Samples')
    sample_template = generator_samples[0]
    # sample_template.find('AutoSeek').text = 'true'
    generator_samples.remove(sample_template)
    mod_sets = generator_template.find('ModulationSets')
    mod_set_template = mod_sets[0]
    mod_sets.remove(mod_set_template)

    def get_key(note):
        return str(min(127, max(0, note - 24)))

    input_samples = []
    for index, instrument in enumerate(wd_data.instruments):
        instrument_elem = deepcopy(instrument_template)
        instrument_name = instrument.get_name()
        instrument_path = 'Instrument%02d (%s)' % (index, instrument_name)
        instrument_elem.find('Name').text = instrument_name
        gen = instrument_elem.find('SampleGenerator')
        samples_elem = gen.find('Samples')
        global_prop = instrument_elem.find('GlobalProperties')
        pitch_bend = global_prop.find('PitchbendMacro')
        pitch_maps = pitch_bend.find('Mappings')
        pitch_map_templ = pitch_maps[0]
        pitch_maps.remove(pitch_map_templ)

        mod_sets = gen.find('ModulationSets')

        sample_index = 0
        key_low = 0
        for sample in instrument.samples:
            key_high = sample[0].key_high
            for channel in sample:
                wav = channel.get_wav(True)
                sample_elem = deepcopy(sample_template)
                sample_name = channel.get_name()
                sample_elem.find('Name').text = sample_name
                sample_elem.find('Volume').text = str(channel.volume)
                sample_elem.find('Panning').text = str(channel.pan)
                mod_index = str(sample_index)
                sample_elem.find('ModulationSetIndex').text = mod_index
                finetune = int(channel.get_finetune() * 128)
                sample_elem.find('Finetune').text = str(finetune)
                sample_path = 'Sample%02d (%s)' % (sample_index, sample_name)
                input_samples.append((instrument_path, sample_path, wav))
                mapping = sample_elem.find('Mapping')

                if channel.loop_start != -1:
                    sample_elem.find('LoopMode').text = 'Forward'
                    sample_elem.find('LoopRelease').text = 'false'
                    num_samples = channel.get_sample_count(loop=False)
                    sample_elem.find('LoopStart').text = str(num_samples)
                    num_samples = channel.get_sample_count(loop=True)
                    sample_elem.find('LoopEnd').text = str(num_samples)

                mapping.find('BaseNote').text = get_key(channel.get_base_key())
                mapping.find('NoteStart').text = get_key(key_low)
                mapping.find('NoteEnd').text = get_key(key_high)
                samples_elem.append(sample_elem)

                # add mapping to pitch bend
                pitch_map = deepcopy(pitch_map_templ)
                pitch_map.find('DestChainIndex').text = str(sample_index)
                pitch_maps.append(pitch_map)

                # create mod set
                mod_set = deepcopy(mod_set_template)
                devs = mod_set.find('Devices')
                adsr = devs.find('SampleAhdsrModulationDevice')

                # we can do this since FFCC only uses attack/release with
                # full volume

                mod_sets.append(mod_set)

                def get_time(ms):
                    return (ms/60000.0)**(1/3.0)

                if channel.adsr_time[0] != 0:
                    time = get_time(channel.adsr_time[0] * 5)
                    adsr.find('Attack').find('Value').text = str(time)

                if channel.adsr_time[3] != 0:
                    time = get_time(channel.adsr_time[3] * 5)
                    adsr.find('Release').find('Value').text = str(time)

                if (channel.adsr_volume != [0, 127, 127, 127] or
                        channel.adsr_time[:3] != [0, 0, 0]):
                    print('adsr:', channel.adsr_time, channel.adsr_volume,
                          channel.get_name())

                sample_index += 1
            key_low = key_high + 1
        instruments.append(instrument_elem)

    tracks = tree.find('Tracks')
    master = tracks.find('SequencerMasterTrack')
    devices = master.find('FilterDevices').find('Devices')
    mixer = devices.find('MasterTrackMixerDevice')
    mixer.find('Volume').find('Value').text = str(bgm_data.volume)

    track_template = tracks[0]
    tracks.remove(track_template)
    devices = track_template.find('FilterDevices').find('Devices')
    reverb = devices.find('Reverb3Device')

    def set_effect_val(dev, k, v):
        reverb.find(k).find('Value').text = str(v)

    set_effect_val(reverb, 'ReverbTime', effect_data['time'] * 10)
    set_effect_val(reverb, 'PreDelay', effect_data['predelay'] * 1000)

    all_track_lines = []
    max_len = 0
    for index, track in enumerate(bgm_data.tracks):
        track_elem = deepcopy(track_template)
        track_elem.find('Name').text = 'Track %02d' % (index+1)
        tracks.insert(-1, track_elem)
        last_time, track_lines = get_track_lines(track)
        max_note = 0
        max_effect = 0
        for t, line in track_lines:
            note_columns = line.find('NoteColumns')
            if note_columns is not None:
                max_note = max(max_note, len(note_columns))
            effect_columns = line.find('EffectColumns')
            if effect_columns is not None:
                max_effect = max(max_effect, len(effect_columns))

        # move pitch bends, add note off to loop start
        loop_line = None
        for t, line in track_lines:
            if t == bgm_data.loop_start:
                loop_line = line
            note_columns = line.find('NoteColumns')
            if note_columns is None:
                continue
            for note_index, column in enumerate(note_columns):
                pan = column.find('Panning')
                if pan is None:
                    continue
                if pan.text != 'M1':
                    continue
                break
            else:
                continue
            for _ in range(max_note - len(note_columns)):
                note_columns.insert(note_index,
                                    ElementTree.Element('NoteColumn'))

        if bgm_data.loop_start is not None:
            if loop_line is None:
                loop_line = ElementTree.Element('Line')
                track_lines.append((bgm_data.loop_start, loop_line))
                track_lines.sort(key=lambda x: x[0])
            note_columns = loop_line.find('NoteColumns')
            effect_columns = loop_line.find('EffectColumns')
            if note_columns is None:
                note_columns = ElementTree.SubElement(loop_line, 'NoteColumns')
            if effect_columns is None:
                effect_columns = ElementTree.SubElement(loop_line,
                                                        'EffectColumns')
            for i in range(max_note):
                try:
                    column = note_columns[i]
                except IndexError:
                    column = ElementTree.SubElement(note_columns,
                                                    'NoteColumn')
                if column.find('Note') is not None:
                    continue
                ElementTree.SubElement(column, 'Note').text = 'OFF'

        track_elem.find('NumberOfVisibleNoteColumns').text = str(max_note)    
        track_elem.find('NumberOfVisibleEffectColumns').text = str(max_effect)    
        max_len = max(max_len, last_time)
        all_track_lines.append(track_lines)

    pool = tree.find('PatternPool')
    patterns = pool.find('Patterns')
    pattern_template = patterns[0]
    for pattern in list(patterns):
        patterns.remove(pattern)

    pat_tracks = pattern_template.find('Tracks')
    pat_track_template = pat_tracks[0]
    pat_tracks.remove(pat_track_template)
    lines = pat_track_template.find('Lines')
    if lines is not None:
        pat_track_template.remove(lines)

    pattern_size = 384
    pattern_p = 0
    pat_id = 0
    has_loop = bgm_data.loop_start is not None
    loop_start = bgm_data.loop_start
    loop_end = bgm_data.loop_end

    loop_start_pattern = None
    loop_end_pattern = None
    while pattern_p < max_len:
        pattern = deepcopy(pattern_template)
        patterns.append(pattern)
        tracks = pattern.find('Tracks')
        start_off = pattern_p
        end_off = start_off + pattern_size
        if has_loop and (start_off < bgm_data.loop_start and
                         end_off > bgm_data.loop_start):
            end_off = bgm_data.loop_start
        if has_loop and (start_off < bgm_data.loop_end and
                         end_off > bgm_data.loop_end):
            end_off = bgm_data.loop_end

        is_loop_start = start_off == bgm_data.loop_start
        if is_loop_start:
            loop_start_pattern = pat_id

        if end_off == bgm_data.loop_end:
            loop_end_pattern = pat_id

        pattern.find('NumberOfLines').text = str(end_off - start_off)
        pattern_p = end_off
        for index in range(len(bgm_data.tracks)):
            pat_track = deepcopy(pat_track_template)
            tracks.insert(-1, pat_track)
            lines = []
            for t, line in all_track_lines[index]:
                if t < start_off or t >= end_off:
                    continue
                line.set('index', str(t - start_off))
                lines.append(line)
            if not lines:
                continue

            lines_elem = ElementTree.SubElement(pat_track, 'Lines')
            lines_elem.extend(lines)

        pat_id += 1

    seq = tree.find('PatternSequence')
    seq_entries = seq.find('SequenceEntries')
    entry_template = seq_entries[0]
    seq_entries.remove(entry_template)

    for i in range(pat_id):
        new_entry = deepcopy(entry_template)
        new_entry.find('Pattern').text = str(i)
        seq_entries.append(new_entry)

    if has_loop:
        loop_sel = seq.find('LoopSelection')
        loop_sel.find('CursorPos').text = str(loop_start_pattern)
        loop_sel.find('RangePos').text = str(loop_end_pattern)

    out_path = os.path.join(outdir, '%s.xrns' % bgm_data.self_id)
    with zipfile.ZipFile(out_path, 'w') as fp:
        fp.writestr('Song.xml', ElementTree.tostring(tree))
        for (instrument_path, sample_name, s) in input_samples:
            fp.writestr('SampleData/%s/%s.wav' % (instrument_path,
                                                  sample_name), s)

def convert_str(path, outdir):
    with open(path, 'rb') as fp:
        data = AudioStream(DataStream(fp))
    wav = data.get_wav()
    base = os.path.basename(path).split('.')[0]
    new_path = os.path.join(outdir, base + '.wav')
    with open(new_path, 'wb') as fp:
        fp.write(wav)

def convert_wd(path, outdir):
    with open(path, 'rb') as fp:
        data = WaveData(DataStream(fp), path)
    base = os.path.basename(path)
    for instrument in data.instruments:
        inst_path = os.path.join(outdir, base + '_' + instrument.get_name())
        for sample_index, sample in enumerate(instrument.samples):
            sample_path = inst_path + '_' + str(sample_index)
            for channel_index, channel in enumerate(sample):
                if channel.is_null:
                    continue
                channel_path = (sample_path + '_' + str(channel_index) + '_' +
                                channel.get_name() + '.wav')
                wav = channel.get_wav()
                with open(channel_path, 'wb') as fp:
                    fp.write(wav)

def convert(path, outdir):
    if path.endswith('.str'):
        handler = convert_str
    elif path.endswith('.wd'):
        handler = convert_wd
    elif path.endswith('.bgm'):
        handler = convert_bgm
    else:
        return
    print(path)
    handler(path, outdir)

def convert_all(path, outdir):
    for name in os.listdir(path):
        new_path = os.path.join(path, name)
        convert(new_path, outdir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path to file or directory')
    parser.add_argument('--outdir', dest='outdir',help='output directory',
                        default=os.getcwd())
    args = parser.parse_args()
    try:
        os.makedirs(args.outdir)
    except OSError:
        pass
    if os.path.isfile(args.input):
        convert(args.input, args.outdir)
    else:
        convert_all(args.input, args.outdir)