cmake_minimum_required(VERSION 3.1)
project(PSL)

### Plugin dependencies
find_package(SofaFramework REQUIRED)
find_package(SofaPython QUIET)
if(!SofaPython_FOUND)
    message(FATAL_ERROR "-- PSL requires the SofaPython plugin to be activated.")
endif()


set(HEADER_FILES
    config.h
    SceneLoaderPSL.h
    components/Import.h
    components/Python.h
    components/TestResult.h
    components/PSLVersion.h
    components/Template.h
)

set(SOURCE_FILES
    initPSL.cpp
    SceneLoaderPSL.cpp
    components/Import.cpp
    components/Python.cpp
    components/TestResult.cpp
    components/PSLVersion.cpp
    components/Template.cpp
)

set(PYTHON_FILES
    python/psl/dsl.py
    python/pslloader.py
    python/pslengine.py
    python/pslast.py
    python/pslparserxml.py
    python/pslparserhjson.py
    python/pslparserpickled.py
)

set(EXTRA_FILES
    examples/livedemo1/mytemplates.psl
    examples/livedemo1/sceneinit.psl
    examples/livedemo1/starthere.psl
    examples/livedemo1/shapes.psl
    examples/starthere.psl
    examples/starthere.pslx
    examples/python01.psl
    examples/python02.psl
    examples/python03.psl
    examples/template01.psl
    examples/template02.psl
    examples/template03.psl
    examples/using.psl
    examples/import.psl
    examples/mylibrary.psl
    examples/caduceus.psl
    examples/caduceus.pslx
    pep/pep0_pythoncontroller.psl
    pep/pep1_cythoncontroller.psl
    pep/pep2_pythoninstanciate.psl
    pep/pep3_pythoncomponent.psl
    pep/pep4_properties.psl
    pep/pep5_importdirectory.psl
    pep/pep6_functional.psl
    pep/pep7_reactiveprogramming.psl
    pep/pep8_templateexample.psl
    pep/pep9_pythoncontroller.psl
    pep/pep10_hotreload.psl
    psl-highlighting.xml
    README.md
    TODO.md
)



# Config files and install rules for pythons scripts
sofa_set_python_directory(${PROJECT_NAME} "python")

## Add test project
if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    add_subdirectory(PSL_test)
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationGraph SofaPython)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/..")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PSL")

install(TARGETS ${PROJECT_NAME}
        COMPONENT PSL_libraries
        EXPORT PSLTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)


