cmake_minimum_required(VERSION 2.8.12)
project(SofaGuiGlut)

set(HEADER_FILES
    SimpleGUI.h)

set(SOURCE_FILES
    SimpleGUI.cpp)

if(SOFA_HAVE_BOOST)
    list(APPEND HEADER_FILES MultithreadGUI.h)
    list(APPEND SOURCE_FILES MultithreadGUI.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} SofaGuiCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIGLUT")

sofa_install_targets(SofaGui SofaGuiGlut "")
