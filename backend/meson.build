project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')
rbtree_incdir = include_directories('rbtree/include/')

ll = shared_library('ll', 'rbtree/src/ll/ll.cpp', include_directories:rbtree_incdir)
rbt = shared_library('rbt', 'rbtree/src/rbt/tree.cpp', 'rbtree/src/rbt/iterator.cpp', include_directories:rbtree_incdir)
dsv = shared_library('dsv', 'src/dsv.cpp', include_directories:incdir)
graphical = shared_library('graphical', 'src/graphical.cpp', include_directories:[incdir, rbtree_incdir], link_with:[rbt])
graphical2dsv = shared_library('graphical2dsv', 'src/graphical2dsv.cpp', include_directories:[incdir, rbtree_incdir], link_with:[graphical, dsv])

test_filenames = [['test_dsv_HZH_X', 'tests/dsv_HZH_X.cpp']
            , ['test_dsv_S_by_R', 'tests/dsv_S_by_R.cpp']
            , ['test_dsv_HH_I', 'tests/dsv_HH_I.cpp']
            , ['test_graph2dsv_selected_exports', 'tests/graph2dsv_selected_exports.cpp']
            , ['test_graph2dsv_selected_exports_CZ', 'tests/graph2dsv_selected_exports_CZ.cpp']
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

