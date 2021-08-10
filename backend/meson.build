project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')

dsv = shared_library('dsv', 'src/dsv.cpp', include_directories:incdir)
ll = shared_library('ll', 'src/ll/ll.cpp', include_directories:incdir)
rbt = shared_library('rbt', 'src/rbt/rbt.cpp', include_directories:incdir)

main = executable('main', 'main.cpp', link_with:[dsv, ll, rbt], include_directories:incdir)

test_filenames = [['test_rbt_insert', 'tests/rbtree_insert.cpp'] 
                , ['test_ll_insert', 'tests/ll_insert.cpp']
                , ['test_rbt_has_value', 'tests/rbtree_has_value.cpp']
                , ['test_ll_has_value', 'tests/ll_has_value.cpp']]
tests = []
foreach tfname: test_filenames
    tests += [[tfname[0], executable(tfname[0]
                            , tfname[1]
                            , include_directories: [incdir]
                            , link_with: [ll, dsv, rbt]
                            , install: false)]]
endforeach


foreach t:tests
    test(t[0], t[1])
endforeach

