from setuptools import setup, find_packages

setup(
    name="neuropixel_data_exporter",
    version="1.3.0",
    packages=find_packages(where="core"),
    package_dir={"": "core"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "plotly",
        "openpyxl",
        "XlsxWriter",
    ],
    python_requires=">=3.8",
)
