"""Setup pose_estimation"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

from itertools import dropwhile
import numpy as np
from os import path


def collect_docstring(lines):
    """Return document docstring if it exists"""
    lines = dropwhile(lambda x: not x.startswith('"""'), lines)
    doc = ""
    for line in lines:
        doc += line
        if doc.endswith('"""\n'):
            break

    return doc[3:-4].replace("\r", "").replace("\n", " ")


def collect_metadata():
    meta = {}
    with open(path.join("pose_estimation", "__init__.py")) as f:
        lines = iter(f)
        meta["description"] = collect_docstring(lines)
        for line in lines:
            if line.startswith("__"):
                key, value = map(lambda x: x.strip(), line.split("="))
                meta[key[2:-2]] = value[1:-1]

    return meta

def get_install_requirements():
    return [
        "numpy",
        "scikit-learn",
        "trimesh==2.38.42",
        "torch",
        "torchvision==0.1.8",
        "progress==1.4",
        "cython",
        "Pillow",
        "pyquaternion",
        "backports.functools_lru_cache",
        "sympy",
        "matplotlib==2.2.5",
        "seaborn",
        "mayavi"
    ]


def setup_package():
    meta = collect_metadata()
    setup(
        name="pose_estimation",
        version=meta["version"],
        maintainer=meta["maintainer"],
        maintainer_email=meta["email"],
        url=meta["url"],
        license=meta["license"],
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
        ],
        install_requires=get_install_requirements()
    )


if __name__ == "__main__":
    setup_package()
