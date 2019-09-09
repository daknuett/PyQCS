py_files= src/pyqcs/__init__.py \
		  src/pyqcs/gates/abc.py src/pyqcs/gates/circuits.py \
		  src/pyqcs/gates/executor.py src/pyqcs/gates/__init__.py \
		  src/pyqcs/build/abc.py src/pyqcs/build/__init__.py \
		  src/pyqcs/state/abc.py src/pyqcs/state/state.py src/pyqcs/state/__init__.py \
		  setup.py


c_files= src/pyqcs/gates/implementations/basic_gates.c \
		 src/pyqcs/gates/implementations/generic_setup.h

all: test

clean:
	python3 setup.py clean
	-rm -r build
	-rm -r dist
	-rm -r pyqcs.egg-info

install: Build
	python3 setup.py install

Build: $(py_files) $(c_files)
	python3 setup.py build

test: clean install
	((cd test && python3 -m pytest test/ -vv))
