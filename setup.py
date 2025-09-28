#!/usr/bin/env python3
"""
DeepDTA-Pro Setup Script
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

# Read long description from README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "DeepDTA-Pro: Advanced Drug-Target Binding Affinity Prediction using Graph Neural Networks"

setup(
    name="deepdta-pro",
    version="1.0.0",
    author="DeepDTA-Pro Team",
    author_email="contact@deepdta-pro.com",
    description="Advanced Drug-Target Binding Affinity Prediction using Graph Neural Networks",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/deepdta-pro/deepdta-pro",
    
    packages=find_packages(include=['src', 'src.*']),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    entry_points={
        'console_scripts': [
            'deepdta-pro=src.inference.predictor:main',
            'deepdta-train=scripts.train_models:main',
            'deepdta-eval=scripts.evaluate_models:main',
            'deepdta-demo=run_demo:main',
        ],
    },
    
    extras_require={
        'dev': [
            'pytest>=7.2.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'cuda': [
            'cupy-cuda11x>=12.0.0',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    
    package_data={
        'src': ['configs/*.yaml'],
    },
    
    include_package_data=True,
    zip_safe=False,
)