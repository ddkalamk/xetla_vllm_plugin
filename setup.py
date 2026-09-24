import os
import torch
import glob
import torch.utils.cpp_extension as cpp_ext
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

def _wrap_sycl_host_flags(cflags):
    #print("cflags: ", cflags)
    return ""
    # host_cxx = get_cxx_compiler()
    # host_cflags = [
    #     f'-fsycl-host-compiler={host_cxx}',
    #     shlex.quote(f'-fsycl-host-compiler-options={cflags}'),
    # ]
    # return host_cflags

torch.utils.cpp_extension._wrap_sycl_host_flags = _wrap_sycl_host_flags

os.environ["TORCH_XPU_ARCH_LIST"] = ""

# Find all .cpp and .sycl files in csrc
source_files = []
for root, dirs, files in os.walk('csrc'):
    for file in files:
        if file.endswith('.cpp') or file.endswith('.sycl'):
            source_files.append(os.path.join(root, file))

cwd = os.path.dirname(os.path.realpath(__file__))
#print(f"cwd={cwd}")
xetla_root = os.path.join(cwd, "xetla")
if "XETLA_ROOT" in os.environ:
    xetla_root = os.getenv("XETLA_ROOT")
xetla_root = os.path.realpath(xetla_root)
print("xetla_root = ", xetla_root)
include_dirs = [os.path.join(xetla_root, "include")]
# print(include_dirs)
print("source_files: ", source_files)

# TernSYCL kernels (ternsycl submodule): plain SIMT SYCL, so they cannot share
# the xetla extension's -vc-codegen backend flags. Built ahead of time for
# TERNSYCL_AOT_DEVICES (JIT makes some of them slow and unstable).
ternsycl_root = os.path.realpath(os.getenv("TERNSYCL_ROOT", os.path.join(cwd, "ternsycl")))
ternsycl_aot = os.getenv("TERNSYCL_AOT_DEVICES", "bmg-g31,lnl-m")
ternsycl_sources = sorted(glob.glob("csrc_ternsycl/*.sycl") + glob.glob("csrc_ternsycl/*.cpp"))
ternsycl_includes = [os.path.join(ternsycl_root, d) for d in
                     ("common", "int2_fp16_upcvt", "int2_via_int2_x_int8_dpas", "hadamard")]


class TernBuildExtension(BuildExtension):
    """torch reads the SYCL device targets from globals when it compiles, so
    set them per extension: JIT for xetla, AOT with per-kernel images for
    TernSYCL."""

    def build_extension(self, ext):
        dlink = cpp_ext._SYCL_DLINK_FLAGS
        if ext.name == "ternsycl_pt_ext":
            os.environ["TORCH_XPU_ARCH_LIST"] = ternsycl_aot
            cpp_ext._SYCL_DLINK_FLAGS = dlink + ["-fsycl-device-code-split=per_kernel",
                                                 "-fsycl-max-parallel-link-jobs=16"]
        else:
            os.environ["TORCH_XPU_ARCH_LIST"] = ""
        try:
            super().build_extension(ext)
        finally:
            cpp_ext._SYCL_DLINK_FLAGS = dlink


setup(
    name='xetla_pt_ext',
    ext_modules=[
        SyclExtension(
            'xetla_pt_ext',
            sources=source_files,
            include_dirs = include_dirs,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20'],
                'sycl': ['-O3', '-std=c++20', '-Wno-unusable-partial-specialization',
                         # BITCOS SOTA unpack: fp16 SLM LUT, exact 3-word sign
                         # gather, fp16 store. See xetla bitcos test Makefile.
                         '-DBITCOS_FP16_LUT', '-DBITCOS_SIGN_GATHER4',
                         '-DBITCOS_SIGN_GATHER3', '-DBITCOS_FP_STORE',
                         '-Xsycl-target-backend="-vc-codegen -vc-disable-indvars-opt -Xfinalizer \' -printregusage -enableBCR -DPASTokenReduction \' -doubleGRF"'],
            },
        ),
        SyclExtension(
            'ternsycl_pt_ext',
            sources=ternsycl_sources,
            include_dirs=ternsycl_includes,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++20'],
                'sycl': ['-O3', '-std=c++20', '-fp-model=precise', '-Wno-psabi',
                         '-fsycl-targets=spir64_gen'],
            },
        ),
    ],
    py_modules=["xetla_vllm_plugin"],
    entry_points={
        'vllm.general_plugins':
        ["xetla_model = xetla_vllm_plugin:register"]
    },
    cmdclass={
        'build_ext': TernBuildExtension
    },
)
