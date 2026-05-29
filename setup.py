from setuptools import find_packages, setup


setup(
    name="adsorbflow",
    version="0.1.0",
    description=(
        "Energy-conditioned deterministic transport for fast adsorbate "
        "placement on catalytic surfaces"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    author="Jiangjie Qiu, Wentao Li, Honghao Chen, Leyi Zhao, Xiaonan Wang",
    url="https://github.com/sunyrain/AdsorbFlow",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
)
