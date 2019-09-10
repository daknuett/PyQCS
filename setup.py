from setuptools import setup, Extension


basic_gates = Extension(
                        "pyqcs.gates.implementations.basic_gates"
                        , sources=["src/pyqcs/gates/implementations/basic_gates.c"]
                        , extra_compile_args=["-fstack-protector", "-Wno-unused-variable"]
                    )

setup(
        name="pyqcs"
        , version="0.0.1"
        , description="A quantum computing simulator."
        , ext_modules=[basic_gates]
        , packages=[ 
            "pyqcs"
            , "pyqcs.gates"
            , "pyqcs.build"
            , "pyqcs.state"
            , "pyqcs.gates.implementations"
        ]
        , package_dir={"pyqcs": "src/pyqcs"}
        , install_requires=["numpy"]
        , project_urls={
            "Source Code": "https://github.com/daknuett/pyqcs"
        }
        , author="Daniel Knüttel"
        , author_email="daniel.knuettel@daknuett.eu"
    )

