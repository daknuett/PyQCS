from setuptools import setup, Extension, find_packages


basic_gates = Extension(
                        "pyqcs.gates.implementations.basic_gates"
                        , sources=["src/pyqcs/gates/implementations/basic_gates.c"]
                        , extra_compile_args=["-fstack-protector", "-Wno-unused-variable"]
                    )
generic_gate = Extension(
                        "pyqcs.gates.implementations.generic_gate"
                        , sources=["src/pyqcs/gates/implementations/generic_gate.c"]
                        , extra_compile_args=["-fstack-protector", "-Wno-unused-variable"])

setup(
        name="pyqcs"
        , version="0.0.16"
        , description="A quantum computing simulator."
        , long_description = open("README.rst").read()
        , ext_modules=[basic_gates
                        , generic_gate]
        , packages=find_packages(where="src")
        , package_dir={"pyqcs": "src/pyqcs"}
        , install_requires=["numpy"]
        , project_urls={
            "Source Code": "https://github.com/daknuett/pyqcs"
        }
        , author="Daniel Knüttel"
        , author_email="daniel.knuettel@daknuett.eu"
    )

