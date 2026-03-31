from setuptools import setup, find_packages

setup(
    name="neuropixel_data_exporter",
    version="2.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.1",
        "plotly>=5.20",
        "openpyxl>=3.1",
        "XlsxWriter>=3.1",
        "scipy>=1.10",
        "PySide6>=6.6",
    ],
    python_requires=">=3.11",
)
