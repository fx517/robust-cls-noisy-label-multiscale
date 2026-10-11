"""Minimal tfevents parser (no tensorboard dependency).

Generic purpose: extract tag -> (step, scalar_value) from Event protobuf.
Wire format parsing for Event / Summary / Summary.Value.
"""
import struct
import sys


def _read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_len_delimited(buf, pos, end):
    out = []
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3
        wt = tag & 0x7
        if field == 1 and wt == 2:  # tag (string)
            length, pos = _read_varint(buf, pos)
            out.append(('tag', buf[pos:pos + length].decode('utf-8', 'replace')))
            pos += length
        elif field == 2 and wt == 5:  # simple_value (float)
            out.append(('value', struct.unpack('<f', buf[pos:pos + 4])[0]))
            pos += 4
        elif field == 2 and wt == 1:  # tensor (unused)
            pos += 8
        else:
            # skip unknown
            if wt == 0:
                _, pos = _read_varint(buf, pos)
            elif wt == 1:
                pos += 8
            elif wt == 2:
                length, pos = _read_varint(buf, pos)
                pos += length
            elif wt == 5:
                pos += 4
            else:
                break
    return dict(out)


def parse_summary(buf):
    """Parse a Summary message -> {tag: value}."""
    result = {}
    pos = 0
    end = len(buf)
    while pos < end:
        tag, pos = _read_varint(buf, pos)
        field = tag >> 3
        wt = tag & 0x7
        if field == 1 and wt == 2:  # value (repeated Summary.Value)
            length, pos = _read_varint(buf, pos)
            sub = buf[pos:pos + length]
            pos += length
            v = _parse_len_delimited(sub, 0, len(sub))
            if 'tag' in v and 'value' in v:
                result[v['tag']] = v['value']
        else:
            if wt == 0:
                _, pos = _read_varint(buf, pos)
            elif wt == 1:
                pos += 8
            elif wt == 2:
                length, pos = _read_varint(buf, pos)
                pos += length
            elif wt == 5:
                pos += 4
            else:
                break
    return result


def parse_file(path):
    data = open(path, 'rb').read()
    pos = 0
    records = []
    while pos < len(data):
        # 8-byte length + 4-byte CRC
        if pos + 12 > len(data):
            break
        length = struct.unpack('<Q', data[pos:pos + 8])[0]
        pos += 8
        pos += 4  # skip the length CRC
        payload = data[pos:pos + length]
        pos += length
        pos += 4  # skip masked CRC of data
        # Event protobuf
        step = None
        summary = None
        p = 0
        e = len(payload)
        while p < e:
            tag, p = _read_varint(payload, p)
            field = tag >> 3
            wt = tag & 0x7
            if field == 1 and wt == 1:
                p += 8  # skip wall time
            elif field == 2 and wt == 0:
                step, p = _read_varint(payload, p)
            elif field == 3 and wt == 2:
                # file_version string
                slen, p = _read_varint(payload, p)
                p += slen
            elif field == 5 and wt == 2:
                slen, p = _read_varint(payload, p)
                summary = parse_summary(payload[p:p + slen])
                p += slen
            else:
                if wt == 0:
                    _, p = _read_varint(payload, p)
                elif wt == 1:
                    p += 8
                elif wt == 2:
                    slen, p = _read_varint(payload, p)
                    p += slen
                elif wt == 5:
                    p += 4
                else:
                    break
        if step is not None and summary:
            for k, v in summary.items():
                records.append((step, k, v))
    return records


if __name__ == '__main__':
    path = sys.argv[1]
    keys = sys.argv[2:] or None
    recs = parse_file(path)
    print('TOTAL RECORDS:', len(recs), file=sys.stderr)
    for step, k, v in recs:
        if keys is None or k in keys:
            print('step=%d %s=%.5f' % (step, k, v))