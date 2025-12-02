import json

from setuptools import find_packages, setup


def run_setup():
    with open("setup.json", 'r') as info:
        kwargs = json.load(info)
    setup(
        include_package_data=True,
        packages=find_packages(),
        **kwargs
    )


if __name__ == "__main__":
    run_setup()
