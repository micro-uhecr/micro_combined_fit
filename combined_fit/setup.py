from setuptools import setup

setup(
    name="Combined Fit",
    author="Antonio CONDORELLI",
    author_email="antonio.condorelli@ijclab.in2p3.fr",
    url = "https://gitlab.in2p3.fr/micro/micro_cf",
    packages=["combined_fit"],
    description="Combined UHECR fit",
    setup_requires=['setuptools_scm'],
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    use_scm_version={"write_to":"combined_fit/_version.py"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Beta",
        "License :: OSI Approved :: GPL License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        ],
    install_requires=["astropy",
                      "matplotlib",
                      "numpy",
                      "pandas",                      
                      "scipy",
                      "iminuit",
                      "jupyterlab",
                      "numba",
                      "zstandard",
                      "setuptools_scm"]
)