from setuptools import setup, Extension, find_packages
import numpy


basic_gates = Extension(
                        "pyqcs.gates.implementations.basic_gates"
                        , sources=["src/pyqcs/gates/implementations/basic_gates.c"]
                        , extra_compile_args=["-fstack-protector"
                                            , "-Wno-unused-variable"
                                            , "-g"
                                            , "-I%s" % numpy.get_include()]
                    )
graph_backend = Extension("pyqcs.graph.backend.raw_state"
                        , sources=["src/pyqcs/graph/backend/raw_state.c"
                                , "src/pyqcs/graph/backend/linked_list.c"
                                , "src/pyqcs/graph/backend/graph_operations.c"
                        ]
                        , extra_compile_args=["-fstack-protector"
                                             , "-Wno-unused-variable"
                                             , "-g"
                                             , "-I%s" % numpy.get_include()]
                    )
generic_gate = Extension(
                        "pyqcs.gates.implementations.generic_gate"
                        , sources=["src/pyqcs/gates/implementations/generic_gate.c"]
                        , extra_compile_args=["-fstack-protector"
                                            , "-Wno-unused-variable"
                                            , "-g"
                                            , "-I%s" % numpy.get_include()])

setup(
        name="pyqcs"
        , version="2.0.6"
        , description="A quantum computing simulator."
        , long_description = open("README.rst").read()
        , ext_modules=[basic_gates
                        , generic_gate
                        , graph_backend]
        , packages=find_packages(where="src")
        , package_dir={"pyqcs": "src/pyqcs"}
        , install_requires=["numpy"]
        , project_urls={
            "Source Code": "https://github.com/daknuett/pyqcs"
        }
        , author="Daniel Knüttel"
        , author_email="daniel.knuettel@daknuett.eu"
        , python_requires=">3.5"
    )

