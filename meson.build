project('pyqcs', 'cpp', 'c', version:'3.0.0')

add_global_arguments('-O3', language:'cpp')
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
pydep = python.dependency()
subdir('src/pyqcs')


#####################################################################################
###### This section contains the new backend compilation ############################
#####################################################################################

numpy_cmd = run_command(python, ['-c', 'import numpy; print(numpy.get_include())'], check : true )
message(numpy_cmd.stdout().strip())
message('include_directories(numpy_cmd.stdout().strip())')
numpy_incdir = include_directories(numpy_cmd.stdout().strip())
numpy_dep = declare_dependency(include_directories: numpy_incdir)

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

python.extension_module(
    'dsv'
    , ['src/pyqcs/state/dsv.cpp'
      , 'src/backend/src/dsv.cpp']
    , dependencies: [pydep, numpy_dep]
    , install:true
    , subdir: 'pyqcs/state'
    , include_directories: incdir)
