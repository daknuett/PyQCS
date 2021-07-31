project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')

dsv = shared_library('dsv', 'src/dsv.cpp', include_directories:incdir)

main = executable('main', 'main.cpp', link_with:dsv, include_directories:incdir)
