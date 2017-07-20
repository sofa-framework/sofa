cmake_minimum_required(VERSION 3.1)
project(PSL)

set(HEADER_FILES
    config.h
    SceneLoaderPSL.h
    components/Undefined.h
    components/TestResult.h
    components/PSLVersion.h
    components/Template.h
)

set(SOURCE_FILES
    initPSL.cpp
    SceneLoaderPSL.cpp
    components/Undefined.cpp
    components/TestResult.cpp
    components/PSLVersion.cpp
    components/Template.cpp
)

find_package(SofaFramework REQUIRED)
find_package(SofaPython REQUIRED)

set(PYTHON_FILES
    python/pslloader.py
    python/pslengine.py
    python/pslparserxml.py
)

set(EXTRA_FILES
    examples/starthere.psl
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
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT PSL_libraries
        EXPORT PSLTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
