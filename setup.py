from setuptools import setup, find_packages

# ---------- 主包 ----------
setup(
    name="lshn",
    version="0.1.1",
    description="Liquid Spiking Hypergraph Network — 脑启发持续学习系统",
    author="LSHN Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "experiments*", "scripts*"]),
    # configs/ 不含 __init__.py, find_packages 不会收集;
    # 以 data_files 将默认配置随包分发到 sys.prefix/configs/
    data_files=[("configs", ["configs/default.yaml"])],
    # install_requires 与 requirements.txt 保持同步
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "pyyaml>=6.0",
        "typing-extensions",
    ],
    extras_require={
        # 开发/测试依赖
        "dev": ["pytest"],
    },
)
