import os
import sys
from setuptools import setup, find_packages

# ---------- C++ 扩展 (可选) ----------
ext_modules = []
cmdclass = {}

try:
    from torch.utils import cpp_extension

    ext_modules.append(
        cpp_extension.CppExtension(
            name="lshn_csrc",
            sources=["csrc/sparse_event_driven.cpp"],
            extra_compile_args=["-O3"] if sys.platform != "win32"
            else ["/O2", "/std:c++17"],
        )
    )
    cmdclass["build_ext"] = cpp_extension.BuildExtension
except ImportError:
    print("WARNING: torch not found, skipping C++ extension build")

# ---------- 主包 ----------
setup(
    name="lshn",
    version="0.1.0",
    description="Liquid Spiking Hypergraph Network — 脑启发持续学习系统",
    author="LSHN Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "experiments*", "scripts*"]),
    install_requires=[
        "torch>=2.0.0",
        "torch_geometric>=2.3.0",
        "numpy",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov",
        ],
        "wavelet": [
            "PyWavelets>=1.4.0",
        ],
        "snn": [
            "snntorch>=0.7.0",
        ],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
