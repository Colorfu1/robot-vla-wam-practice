"""Read-only EGL/CUDA/PCI mapping probe; no model or rendering context is created."""

import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys


def pci_key(value):
    domain, bus, slot = value.lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{slot}"


def main():
    # Fresh interpreter: query the complete container-visible CUDA enumeration,
    # independent of the caller's CUDA mask. Never change NVIDIA_VISIBLE_DEVICES.
    if "--unmasked-child" not in sys.argv:
        print("Original selection: " + json.dumps({
            key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "MUJOCO_EGL_DEVICE_ID"
            )
        }), flush=True)
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        return subprocess.call([sys.executable, __file__, "--unmasked-child"], env=env)

    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"
    ], text=True)
    physical = {}
    for line in raw.strip().splitlines():
        index, uuid, pci = [part.strip() for part in line.split(",")]
        physical[pci_key(pci)] = (index, uuid)

    cuda = C.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [C.c_uint]
    cuda.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
    cuda.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    cuda.cuDeviceGetPCIBusId.argtypes = [C.c_void_p, C.c_int, C.c_int]

    def checked(status):
        if status != 0:
            raise RuntimeError(f"CUDA driver query failed: {status}")

    checked(cuda.cuInit(0))
    count = C.c_int()
    checked(cuda.cuDeviceGetCount(C.byref(count)))
    cuda_pci = {}
    for ordinal in range(count.value):
        device = C.c_int()
        checked(cuda.cuDeviceGet(C.byref(device), ordinal))
        bus = C.create_string_buffer(64)
        checked(cuda.cuDeviceGetPCIBusId(bus, len(bus), device))
        cuda_pci[ordinal] = pci_key(bus.value.decode())

    egl = C.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [C.c_char_p]
    egl.eglGetProcAddress.restype = C.c_void_p

    def function(name, result, *args):
        pointer = egl.eglGetProcAddress(name.encode())
        if not pointer:
            raise RuntimeError(f"Missing EGL function: {name}")
        return C.CFUNCTYPE(result, *args)(pointer)

    query = function("eglQueryDevicesEXT", C.c_uint, C.c_int,
                     C.POINTER(C.c_void_p), C.POINTER(C.c_int))
    strings = function("eglQueryDeviceStringEXT", C.c_char_p, C.c_void_p, C.c_int)
    attrib = function("eglQueryDeviceAttribEXT", C.c_uint, C.c_void_p, C.c_int,
                      C.POINTER(C.c_ssize_t))
    total = C.c_int()
    if not query(0, None, C.byref(total)):
        raise RuntimeError("EGL device count query failed")
    devices = (C.c_void_p * total.value)()
    if not query(total.value, devices, C.byref(total)):
        raise RuntimeError("EGL enumeration failed")
    print("EGL_ID\tCUDA_ORDINAL_UNMASKED\tNVIDIA_SMI_INDEX\tPCI_BUS\tGPU_UUID", flush=True)
    unresolved = 0
    for index, device in enumerate(devices):
        extensions = (strings(device, 0x3055) or b"").decode().split()
        ordinal, pci = None, None
        if "EGL_NV_device_cuda" in extensions:
            value = C.c_ssize_t()
            if attrib(device, 0x323A, C.byref(value)):
                ordinal = value.value
                pci = cuda_pci.get(ordinal)
        if pci is None:
            for extension, token in (("EGL_EXT_device_drm_render_node", 0x3377),
                                     ("EGL_EXT_device_drm", 0x3233)):
                if extension in extensions:
                    node = strings(device, token)
                    if node:
                        path = Path("/sys/class/drm") / Path(node.decode()).name / "device"
                        if path.exists() and path.resolve().name.count(":") == 2:
                            pci = pci_key(path.resolve().name)
                            ordinal = next((k for k, v in cuda_pci.items() if v == pci), None)
                            break
        smi, uuid = physical.get(pci, ("UNRESOLVED", "UNRESOLVED"))
        print(f"{index}\t{ordinal}\t{smi}\t{pci}\t{uuid}", flush=True)
        if uuid == "UNRESOLVED":
            unresolved += 1
    print(f"DONE enumerated={total.value} unresolved={unresolved}")
    print("CUDA ordinals above are UNMASKED; do not assume they equal EGL or nvidia-smi indices.")
    print("This verifies identity only, not rendering stability or a safe robosuite binding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
