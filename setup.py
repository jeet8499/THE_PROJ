"""
setup.py — optional, for installing as a package
Usage: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name             = "gold-trading-openenv",
    version          = "3.0.0",
    description      = "XAU/USD Hybrid RL + LLM Trading Environment (OpenEnv v1)",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "openai>=1.35.0",
        "httpx>=0.27.0",
        "fastapi==0.111.0",
        "uvicorn[standard]==0.30.1",
        "pydantic==2.7.4",
        "python-dotenv>=1.0.1",
    ],
    entry_points = {
        "console_scripts": [
            "goldtrade-train=train:main",
        ]
    },
    classifiers = [
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
