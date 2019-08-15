from setuptools import setup,Extension


basic_gates = Extension("pyqcs.gates.implementations.basic_gates",
		sources = ["pyqcs/gates/implementations/basic_gates.c"])

setup(name = "pyqcs",
	version = "0.0.1",
	description = "A quantum computing simulator.",
	ext_modules = [basic_gates],
	packages = [
		"pyqcs"
	],
	package_dir = {"pyqcs": "pyqcs"},
        install_requires=["numpy"],
	#url="https://github.com/daknuett/python3-nf",
	author = "Daniel Knüttel",
	author_email = "daniel.knuettel@daknuett.eu")

