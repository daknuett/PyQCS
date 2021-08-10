project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')

dsv = shared_library('dsv', 'src/dsv.cpp', include_directories:incdir)
ll = shared_library('ll', 'src/ll/ll.cpp', include_directories:incdir)
rbt = shared_library('rbt', 'src/rbt/rbt.cpp', include_directories:incdir)

main = executable('main', 'main.cpp', link_with:[dsv, ll, rbt], include_directories:incdir)
