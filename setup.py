from setuptools import setup, find_packages

setup(
    name="pine_ta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["pandas", "numpy"],
    author="Huzaifa Zahoor",
    author_email="huzaifazahoor654@email.com",
    description="Pure pandas/numpy technical indicators inspired by Pine Script",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huzaifazahoor/pine_ta",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
