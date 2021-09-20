project('pyqcs', 'cpp', 'c', version:'3.0.0a')

add_global_arguments('-O3', language:'cpp')
message(meson.source_root())
#####################################################################################
############### This section contains the purely C++ part ###########################
#####################################################################################

incdir_backend = include_directories('src/backend/include/')
incdir_rbtree = include_directories('src/backend/rbtree/include/')
incdir = [incdir_backend, incdir_rbtree]

ll = shared_library('ll'
      , 'src/backend/rbtree/src/ll/ll.cpp'
      , include_directories:incdir)
rbt = shared_library('rbt'
      , 'src/backend/rbtree/src/rbt/tree.cpp'
      , 'src/backend/rbtree/src/rbt/iterator.cpp'
      , install:true
      , include_directories:incdir)
dsv = shared_library('dsv'
      , 'src/backend/src/dsv.cpp'
      , include_directories:incdir)
graphical = shared_library('graphical'
      , 'src/backend/src/graphical.cpp'
      , include_directories:incdir
      , install:true
      , link_with:[rbt])
graphical2dsv = shared_library('graphical2dsv'
      , 'src/backend/src/graphical2dsv.cpp'
      , include_directories:incdir
      , link_with:[graphical
      , dsv])

#####################################################################################
######### This section contains the pure python part and setup ######################
#####################################################################################

pymod = import('python')
python = pymod.find_installation('python3')
pydep = python.dependency('python3')
subdir('src/pyqcs')

#####################################################################################
######## This section contains the (outdated) old backend compilation ###############
#####################################################################################

basic_gates = python.extension_module('basic_gates', 'src/pyqcs/gates/implementations/basic_gates.c', dependencies: pydep, install:true, subdir:'pyqcs/gates/implementations/')
compute_amplitude = python.extension_module('compute_amplitude', 'src/pyqcs/gates/implementations/compute_amplitude.c', dependencies: pydep, install:true, subdir:'pyqcs/gates/implementations/')
generic_gate = python.extension_module('generic_gate', 'src/pyqcs/gates/implementations/generic_gate.c', dependencies: pydep, install:true, subdir:'pyqcs/gates/implementations/')

#####################################################################################
###### This section contains the new backend compilation ############################
#####################################################################################

python.extension_module(
    'rawstate'
    , ['src/pyqcs/graph/rawstate.cpp'
      , 'src/backend/src/graphical.cpp'
      , 'src/backend/rbtree/src/rbt/tree.cpp'
      , 'src/backend/rbtree/src/rbt/iterator.cpp']
    , dependencies: pydep
    , install:true
    , subdir: 'pyqcs/graph'
    , include_directories: incdir)
