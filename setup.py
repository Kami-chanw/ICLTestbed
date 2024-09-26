from setuptools import setup, find_packages

setup(
    name="ICLTestbed",
    version="0.1.0",
    description="The minimal implementation of various popular AI models",
    author=" Kamichanw",
    author_email="865710157@qq.com",
    packages=find_packages(include=["testbed", "testbed.*"]),
    install_requires=[
        "datasets",
        "dill",
        "filelock",
        "inflection",
        "nltk",
        "numpy",
        "Pillow",
        "pycocoevalcap",
        "torch",
        "tqdm",
        "transformers",
        "xxhash",
    ],
    extras_require={
        "faiss-cpu": ["faiss-cpu"],
        "faiss-gpu": ["faiss-gpu"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
