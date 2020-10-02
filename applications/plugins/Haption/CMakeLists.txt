cmake_minimum_required(VERSION 3.1)
project(Haption)


set(HEADER_FILES
    HaptionDriver.h
)

set(SOURCE_FILES
    initHaption.cpp 
    HaptionDriver.cpp
    initHaption.cpp
)

set(README_FILES
    PluginHaption.txt
)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTIONPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT Haption_libraries
        EXPORT HaptionTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY exemple/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
