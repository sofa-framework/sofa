### These are real modules. If you need components from there you
### need to first add find_package(XXX) in your CMakeLists.txt
sofa_add_plugin(SofaSparseSolver SofaSparseSolver ON)      #
sofa_add_plugin(SofaPreconditioner SofaPreconditioner ON)  # depends on SofaSparseSolver

### All the other targets visible in "modules" are not real module
### and are contained in the SofaGeneral, SofaMisc, SofaAdvanced
