project('backend_rewrite', 'cpp')
add_global_arguments('-O3', language:'cpp')

incdir = include_directories('include/')

dsv = shared_library('dsv', 'src/dsv.cpp', include_directories:incdir)

main = executable('main', 'main.cpp', link_with:[dsv], include_directories:incdir)

test_filenames = []

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

