### These are real modules. If you need components from there you
### need to first add find_package(XXX) in your CMakeLists.txt
sofa_add_module(SofaExporter SofaExporter ON)
sofa_add_module(SofaSparseSolver SofaSparseSolver ON)      #
sofa_add_module(SofaPreconditioner SofaPreconditioner ON)  # depends on SofaSparseSolver
sofa_add_module(SofaHaptics SofaHaptics ON)

if(SOFA_NO_OPENGL)
    message("SOFA_NO_OPENGL flag prevents from using the SofaCUDA, SofaSimpleGUI and VolumetricRendering plugins")
else()
    sofa_add_module(SofaOpenglVisual SofaOpenglVisual ON)
endif()

### All the other targets visible in "modules" are not real module
### and are contained in the SofaGeneral, SofaMisc, SofaAdvanced
