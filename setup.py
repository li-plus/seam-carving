import os
import re

import setuptools

here = os.path.dirname(__file__)

# get the version string
with open(os.path.join(here, "seam_carving", "__init__.py")) as f:
    version = re.search(r'__version__ = "(.*?)"', f.read()).group(1)

setuptools.setup(
    name="seam-carving",
    version=version,
    author="Jiahao Li",
    author_email="liplus17@163.com",
    maintainer="Jiahao Li",
    maintainer_email="liplus17@163.com",
    url="https://github.com/li-plus/seam-carving",
    description="A super-fast Python implementation of seam carving algorithm "
    "for intelligent image resizing.",
    long_description=open(os.path.join(here, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["test"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "seam carving",
        "computer vision",
        "image processing",
        "image resizing",
        "content aware",
    ],
    license="MIT",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
        "numba>=0.56.0",
    ],
    extras_require={
        "dev": [
            "Pillow",
            "pytest",
            "pytest-cov",
            "isort",
            "black",
        ]
    },
)
