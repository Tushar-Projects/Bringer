from setuptools import setup, find_packages

setup(
    name="bringer",
    version="1.0.0",
    description="A fully local, lightweight GPU-accelerated RAG system.",
    packages=find_packages(),
    py_modules=["bringer_cli", "config"],
    entry_points={
        "console_scripts": [
            "Bringer=bringer_cli:launch_bringer",
        ]
    },
)
