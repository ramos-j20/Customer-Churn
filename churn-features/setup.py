from setuptools import setup, find_packages

setup(
    name="churn-features",
    version="1.0.0",
    description="Feature engineering package for Customer Churn prediction",
    author="Diogo",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
