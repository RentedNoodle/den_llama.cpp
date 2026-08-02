#!/usr/bin/env python3
"""den_probe.py -- dump a .den header + entries (mirrors den_loader.cpp)."""
import struct, sys

DEN_MAGIC = 0x4E454400
HDR = 4096
ENT = 128

def rd_u32(b, o): return struct.unpack_from("<I", b, o)[0]
def rd_i64(b, o): return struct.unpack_from("<q", b, o)[0]
def rd_u64(b, o): return struct.unpack_from("<Q", b, o)[0]
def rd_f32(b, o): return struct.unpack_from("<f", b, o)[0]

def parse(path):
    with open(path, "rb") as f:
        hb = f.read(HDR)
        magic = rd_u32(hb, 0); version = rd_u32(hb, 4)
        print("magic=%#x version=%#x arch=%d flags=%d" % (magic, version, rd_u32(hb,8), rd_u32(hb,12)))
        if magic != DEN_MAGIC:
            print("NOT A .den FILE"); return
        tc = rd_u32(hb,104); idx_off = rd_u32(hb,108)
        data_off = rd_u64(hb,112); total = rd_u64(hb,120)
        print("layers=%d heads=%d kv=%d hidden=%d ffn=%d vocab=%d maxseq=%d nrot=%d" % (
            rd_u32(hb,16),rd_u32(hb,20),rd_u32(hb,24),rd_u32(hb,28),rd_u32(hb,32),rd_u32(hb,36),rd_u32(hb,40),rd_u32(hb,44)))
        print("experts=%d used=%d rope_theta=%f eps=%f" % (rd_u32(hb,48),rd_u32(hb,52),rd_f32(hb,56),rd_f32(hb,60)))
        print("ssm_state=%d conv=%d inner=%d grp=%d tstep=%d fullattn=%d mtp=%d vdim=%d" % (
            rd_u32(hb,64),rd_u32(hb,68),rd_u32(hb,72),rd_u32(hb,76),rd_u32(hb,80),rd_u32(hb,84),rd_u32(hb,88),rd_u32(hb,92)))
        print("tensor_count=%d index_offset=%d data_offset=%d total_data=%d" % (tc, idx_off, data_off, total))
        idx = f.read(tc*ENT)
        for i in range(tc):
            e = idx[i*ENT:(i+1)*ENT]
            slot = rd_u32(e,0); hw = rd_u32(e,4); nd = rd_u32(e,8); fl = rd_u32(e,12)
            dims = [rd_i64(e,16+8*j) for j in range(4)]
            numel = rd_u64(e,48); doff = rd_u64(e,56); dsz = rd_u64(e,64)
            print("  slot=%3d hw=%d nd=%d flags=%#x dims=%s numel=%d off=%d size=%d" % (
                slot, hw, nd, fl, dims[:nd], numel, doff, dsz))

if __name__ == "__main__":
    parse(sys.argv[1])
