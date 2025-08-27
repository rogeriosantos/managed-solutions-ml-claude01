"""Setup script for CNC ML Project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cnc-ml-project",
    version="0.1.0",
    author="CNC Analytics Team",
    author_email="analytics@company.com",
    description="A comprehensive machine learning application for CNC downtime prediction and operator performance optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/cnc-ml-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "api": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cnc-ml-train=cnc_ml_project.cli:train_models",
            "cnc-ml-predict=cnc_ml_project.cli:make_predictions",
            "cnc-ml-dashboard=cnc_ml_project.cli:start_dashboard",
        ],
    },
    include_package_data=True,
    package_data={
        "cnc_ml_project": ["config/*.yaml", "templates/*.html"],
    },
    keywords="machine learning, manufacturing, CNC, predictive analytics, operator performance",
    project_urls={
        "Bug Reports": "https://github.com/company/cnc-ml-project/issues",
        "Documentation": "https://cnc-ml-project.readthedocs.io/",
        "Source": "https://github.com/company/cnc-ml-project",
    },
)