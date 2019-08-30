py_files= pyqcs/__init__.py \
		  pyqcs/gates/abc.py pyqcs/gates/circuits.py \
		  pyqcs/gates/executor.py pyqcs/gates/__init__.py \
		  pyqcs/build/abc.py pyqcs/build/__init__.py \
		  pyqcs/state/abc.py pyqcs/state/state.py pyqcs/state/__init__.py

c_files= pyqcs/gates/implementations/basic_gates.c \
		 pyqcs/gates/implementations/generic_setup.h

all: test
install: build
	python3 setup.py install

build: $(py_files) $(c_files)
	python3 setup.py build

test: install
	cd test
	python3 -m pytest test/ -vv
