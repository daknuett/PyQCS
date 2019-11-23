#!/bin/bash

tmpdir=tmp-install-sqc

if python3 -c "import sqc"; then
    echo "sqc is already installed"
else
    mkdir $tmpdir
    cd $tmpdir
    git clone https://github.com/daknuett/sqc
    echo "CLONED sqc to temporary directory"
    cd sqc
    echo "RUNNING install ..."
    python3 setup.py install
    echo "DONE"

    cd ../..
    rm -rf $tmpdir
fi
