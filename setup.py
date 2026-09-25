import glob
import os

import torch.utils.cpp_extension as cpp_ext
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, SyclExtension

# TernSYCL kernels (ternsycl submodule, or TERNSYCL_ROOT), built ahead of time
# for TERNSYCL_AOT_DEVICES: the JIT path makes some of them slow and unstable.
cwd = os.path.dirname(os.path.realpath(__file__))
ternsycl_root = os.path.realpath(os.getenv("TERNSYCL_ROOT", os.path.join(cwd, "ternsycl")))
os.environ["TORCH_XPU_ARCH_LIST"] = os.getenv("TERNSYCL_AOT_DEVICES", "bmg-g31,lnl-m")
# The host part is compiled by icpx itself.
cpp_ext._wrap_sycl_host_flags = lambda cflags: ""
# One device image per kernel (each tile has its own GRF mode), linked in parallel.
cpp_ext._SYCL_DLINK_FLAGS = cpp_ext._SYCL_DLINK_FLAGS + [
    "-fsycl-device-code-split=per_kernel", "-fsycl-max-parallel-link-jobs=16"]

print("ternsycl_root =", ternsycl_root)
setup(
    name="ternsycl_vllm_plugin",
    ext_modules=[
        SyclExtension(
            "ternsycl_pt_ext",
            sources=sorted(glob.glob("csrc/*.sycl") + glob.glob("csrc/*.cpp")),
            include_dirs=[os.path.join(ternsycl_root, d) for d in
                          ("common", "int2_fp16_upcvt", "int2_via_int2_x_int8_dpas", "hadamard")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "sycl": ["-O3", "-std=c++20", "-fp-model=precise", "-Wno-psabi",
                         "-fsycl-targets=spir64_gen"],
            },
        ),
    ],
    py_modules=["ternsycl_vllm_plugin"],
    entry_points={"vllm.general_plugins": ["ternsycl_model = ternsycl_vllm_plugin:register"]},
    cmdclass={"build_ext": BuildExtension},
)
