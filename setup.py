from setuptools import find_packages, setup

setup(
    name="dirpa",
    version="0.1",
    description="Dirichlet Prior Adjustment (DirPA) for few-shot crop type classification on the EuroCropsML dataset.",
    author="Joana Reuss",
    author_email="joana.reuss@tum.de",
    packages=find_packages(exclude=("tests*",)),
    python_requires=">=3.10",
    package_data={"dirpa": ["experiments/**/*.yaml", "experiments/**/*/*.yaml"]},
    include_package_data=True,
    entry_points={"console_scripts": ["dirpa-cli=dirpa.cli:cli"]},
)
