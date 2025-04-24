from setuptools import setup, find_packages

setup(
    name="lbce_bert_mvp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "biopython>=1.79",
    ],
    author="LBCE-BERT MVP Team",
    author_email="example@example.com",
    description="A Minimal Viable Project based on the LBCE-BERT paper",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lbce-bert-mvp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
