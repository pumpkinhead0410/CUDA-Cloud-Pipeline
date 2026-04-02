from setuptools import find_packages, setup

setup(
    name="cuda-cloud-pipeline",
    version="1.0.0",
    description="CUDA/TensorRT ML inference pipeline on AWS EKS",
    author="pumpkinhead0410",
    python_requires=">=3.11",
    packages=find_packages(where=".", include=["src*"]),
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "pydantic>=2.7.0",
        "numpy>=1.26.0",
        "numba>=0.59.0",
    ],
    extras_require={
        "gpu": [
            "tensorrt>=10.0.0",
            "pycuda>=2024.1",
            "onnx>=1.16.0",
        ],
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "httpx>=0.27.0",
            "ruff>=0.4.0",
            "mypy>=1.10.0",
        ],
    },
)
