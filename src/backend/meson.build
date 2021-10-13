project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')
rbtree_incdir = include_directories('rbtree/include/')

ll = library('ll', 'rbtree/src/ll/ll.cpp', include_directories:rbtree_incdir, install:true)
rbt = library('rbt', 'rbtree/src/rbt/tree.cpp', 'rbtree/src/rbt/iterator.cpp', include_directories:rbtree_incdir, install:true)
rbt_objects = rbt.extract_objects(['rbtree/src/rbt/tree.cpp', 'rbtree/src/rbt/iterator.cpp'])
dsv = library('dsv', 'src/dsv.cpp', include_directories:incdir, install:true)
dsv_objects = dsv.extract_objects('src/dsv.cpp')
graphical = library('graphical', 'src/graphical.cpp', include_directories:[incdir, rbtree_incdir], objects: rbt_objects, install:true)
graphical_objects = graphical.extract_objects('src/graphical.cpp')
graphical2dsv = library('graphical2dsv', 'src/graphical2dsv.cpp', include_directories:[incdir, rbtree_incdir], objects:[rbt_objects, dsv_objects, graphical_objects], install:true)

pkgcfg = import('pkgconfig')
pkgcfg.generate(ll)
pkgcfg.generate(rbt)
pkgcfg.generate(dsv)
pkgcfg.generate(graphical)
pkgcfg.generate(graphical2dsv)
install_headers(['include/dsv.hpp',  'include/graphical2dsv.hpp',
                'include/graphical.hpp',  'include/time_m.hpp'])
install_headers(['rbtree/include/ll/ll.hpp'],subdir: 'll')
install_headers(['rbtree/include/rbt/rbt.hpp'], subdir: 'rbt')

test_filenames = [['test_dsv_HZH_X', 'tests/dsv_HZH_X.cpp']
            , ['test_dsv_S_by_R', 'tests/dsv_S_by_R.cpp']
            , ['test_dsv_HH_I', 'tests/dsv_HH_I.cpp']
            , ['test_graph2dsv_selected_exports', 'tests/graph2dsv_selected_exports.cpp']
            , ['test_graph2dsv_selected_exports_CZ', 'tests/graph2dsv_selected_exports_CZ.cpp']
            , ['test_graph_specific_circuit', 'tests/graphical_specific_circuit.cpp']
            , ['test_graph_copy', 'tests/graphical_copy.cpp']
            ]

tests = []
foreach tfname: test_filenames
    tests += [[tfname[0], executable(tfname[0]
                            , tfname[1]
                            , include_directories: [incdir, rbtree_incdir]
                            , link_with: [ll, dsv, rbt, graphical, graphical2dsv]
                            , install: false)]]
endforeach


foreach t:tests
    test(t[0], t[1])
endforeach

