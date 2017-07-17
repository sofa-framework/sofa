cmake_minimum_required(VERSION 3.1)
project(PSL)

set(HEADER_FILES
    config.h
    Undefined.h
)

set(SOURCE_FILES
    initPSL.cpp
    Undefined.cpp
)

find_package(SofaFramework REQUIRED)

find_package(SofaPython QUIET)
if( SofaPython_FOUND )
    list(APPEND HEADER_FILES
        python/Binding_BasePrefab.h
        )

    list(APPEND SOURCE_FILES
        python/Binding_BasePrefab.cpp
        )

    # Config files and install rules for pythons scripts
    sofa_set_python_directory(${PROJECT_NAME} "python")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
#set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_PLUGINEXAMPLE")
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationGraph)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT PSL_libraries
        EXPORT PSLTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
