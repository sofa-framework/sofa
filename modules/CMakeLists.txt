# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

if(NOT SOFA_ENABLE_SOFT_DEPS_TO_SOFAPYTHON)
    sofa_add_module(SofaExporter SofaExporter ON)
endif()
sofa_add_module(SofaSparseSolver SofaSparseSolver ON)      #
sofa_add_module(SofaPreconditioner SofaPreconditioner ON)  # depends on SofaSparseSolver
sofa_add_module(SofaHaptics SofaHaptics ON)
sofa_add_module(SofaValidation SofaValidation ON)

find_package(Newmat QUIET)
sofa_add_module(SofaDenseSolver SofaDenseSolver ${Newmat_FOUND})

sofa_add_module(SofaNonUniformFem SofaNonUniformFem ON)

if(SOFA_NO_OPENGL)
    sofa_add_module(SofaOpenglVisual SofaOpenglVisual OFF)
else()
    sofa_add_module(SofaOpenglVisual SofaOpenglVisual ON)
endif()

# Misc
sofa_add_module(SofaMiscTopology SofaMiscTopology ON)
sofa_add_module(SofaMiscExtra SofaMiscExtra ON)
sofa_add_module(SofaMiscForceField SofaMiscForceField ON)
sofa_add_module(SofaMiscEngine SofaMiscEngine ON)
sofa_add_module(SofaMiscSolver SofaMiscSolver ON)
sofa_add_module(SofaMiscFem SofaMiscFem ON)
sofa_add_module(SofaMiscMapping SofaMiscMapping ON)

# All the other targets visible in "modules" are not real module
# and are contained in packages (SofaGeneral, SofaAdvanced, SofaGui).
