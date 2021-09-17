project('pyqcs', 'cpp', version:'3.0.0a')

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
      , include_directories:incdir)
dsv = shared_library('dsv'
      , 'src/backend/src/dsv.cpp'
      , include_directories:incdir)
graphical = shared_library('graphical'
      , 'src/backend/src/graphical.cpp'
      , include_directories:incdir
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
pydep = dependency('python3')
subdir('src/pyqcs')

#####################################################################################
######## This section contains the (outdated) old backend compilation ###############
#####################################################################################

basic_gates = pymod.extension_module('pyqcs.gates.implementations.basic_gates', 'src/pyqcs/gates/implementations/basic_gates.c', dependencies: pydep)
compute_amplitude = pymod.extension_module('pyqcs.gates.implementations.compute_amplitude', 'src/pyqcs/gates/implementations/compute_amplitude.c', dependencies: pydep)
