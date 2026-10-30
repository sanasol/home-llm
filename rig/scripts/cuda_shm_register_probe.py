#!/usr/bin/env python3
"""Probe: how much of a MAP_SHARED file mapping can CUDA page-lock on this box?

Mimics Strata --shared-expert-arena (open O_RDWR|O_CREAT, ftruncate, mmap MAP_SHARED,
cudaHostRegister PORTABLE|MAPPED before touching pages): first the whole range, then
1 GiB slices, reporting how far the slices got.
usage: CUDA_VISIBLE_DEVICES=N python3 cuda_shm_register_probe.py FILE SIZE_GIB
"""
import ctypes
import glob
import mmap
import os
import sys
import time

path, gib = sys.argv[1], int(sys.argv[2])
size = gib << 30
lib = ctypes.CDLL(sorted(glob.glob(
    "~/HyperQwen-0.32/venv/lib/python3*/site-packages/nvidia/cu13/lib/libcudart.so*"))[0])
lib.cudaGetErrorString.restype = ctypes.c_char_p
FLAGS = 1 | 2  # cudaHostRegisterPortable | cudaHostRegisterMapped

fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
os.ftruncate(fd, size)
mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
buf = (ctypes.c_char * size).from_buffer(mm)
ptr = ctypes.addressof(buf)
lib.cudaFree(ctypes.c_void_p(0))

t = time.time()
rc = lib.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(size), ctypes.c_uint(FLAGS))
print(f"whole {gib} GiB: rc={rc} {lib.cudaGetErrorString(rc).decode()} in {time.time() - t:.1f}s", flush=True)
if rc == 0:
    lib.cudaHostUnregister(ctypes.c_void_p(ptr))
else:
    lib.cudaGetLastError()
    done = 0
    for i in range(gib):
        r = lib.cudaHostRegister(ctypes.c_void_p(ptr + (i << 30)), ctypes.c_size_t(1 << 30), ctypes.c_uint(FLAGS))
        if r != 0:
            lib.cudaGetLastError()
            print(f"slice {i} failed rc={r} {lib.cudaGetErrorString(r).decode()}", flush=True)
            break
        done += 1
    print(f"slices pinned: {done} GiB", flush=True)
    os.system("grep -E 'MemAvailable|Shmem:|Unevictable|Mlocked' /proc/meminfo")
del buf
mm.close()
os.close(fd)
os.unlink(path)
