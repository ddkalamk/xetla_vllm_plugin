import os
import torch
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension


def _wrap_sycl_host_flags(cflags):
    # print("cflags: ", cflags)
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
for root, dirs, files in os.walk("csrc"):
    for file in files:
        if file.endswith(".cpp") or file.endswith(".sycl"):
            source_files.append(os.path.join(root, file))

cwd = os.path.dirname(os.path.realpath(__file__))
# print(f"cwd={cwd}")
xetla_root = os.path.join(cwd, "xetla")
if "XETLA_ROOT" in os.environ:
    xetla_root = os.getenv("XETLA_ROOT")
xetla_root = os.path.realpath(xetla_root)
print("xetla_root = ", xetla_root)
include_dirs = [os.path.join(xetla_root, "include")]
# print(include_dirs)
print("source_files: ", source_files)
setup(
    name="xetla_pt_ext",
    ext_modules=[
        SyclExtension(
            "xetla_pt_ext",
            sources=source_files,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "sycl": [
                    "-O3",
                    "-std=c++20",
                    "-Wno-unusable-partial-specialization",
                    "-Xsycl-target-backend=\"-vc-codegen -vc-disable-indvars-opt -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' -doubleGRF\"",
                ],
            },
        ),
    ],
    py_modules=["xetla_vllm_plugin"],
    entry_points={"vllm.general_plugins": ["xetla_model = xetla_vllm_plugin:register"]},
    cmdclass={"build_ext": BuildExtension},
)
