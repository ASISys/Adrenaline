import os
import pathlib

from setuptools import setup
from torch.utils import cpp_extension

CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.getcwd()

src_files = list(pathlib.Path("adrenaline/csrc").glob("**/*.cu")) + list(
    pathlib.Path("adrenaline/csrc").glob("**/*.cpp")
)
sources = [str(x) for x in src_files]

if __name__ == "__main__":
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            cpp_extension.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass

    ops_ext = cpp_extension.CppExtension(
        name="adrenaline_ops",
        sources=sources,
        include_dirs=[
            f"/usr/local/cuda/include",
            f"{CURRENT_DIR}/include",
        ],
        libraries=["cuda", "c10_cuda"],
        extra_compile_args=["-std=c++17"],
    )

    setup(
        name="adrenaline",
        ext_modules=[ops_ext],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
    )
