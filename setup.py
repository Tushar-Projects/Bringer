from pathlib import Path

from setuptools import find_packages, setup


README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")


setup(
    name="bringer",
    version="1.0.0",
    description="A fully local, lightweight GPU-accelerated RAG system.",
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(),
    py_modules=["bringer_cli", "config"],
    install_requires=[
        "chromadb>=0.5.0",
        "httpx>=0.27.0",
        "langchain-text-splitters>=0.2.0",
        "markdown>=3.5",
        "psutil>=5.9.0",
        "pypdf>=4.0.0",
        "python-docx>=1.1.0",
        "python-pptx>=0.6.23",
        "rank-bm25>=0.2.2",
        "rich>=13.0.0",
        "sentence-transformers>=3.0.0",
        "streamlit>=1.35.0",
        "tiktoken>=0.7.0",
        "torch>=2.1.0",
        "watchdog>=4.0.0",
    ],
    entry_points={
        "console_scripts": [
            "Bringer=bringer_cli:launch_bringer",
        ]
    },
)
