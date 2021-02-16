cmake_minimum_required(VERSION 3.12)
project(Sofa.Testing LANGUAGES CXX)

# Enable testing features of cmake, like the add_test() command.
enable_testing()

# add gtest library
# add_subdirectory(extlibs/gtest)
# Depends on SofaSimulation
# add_subdirectory(tools/SofaGTestMain)

set(SOFATESTINGSRC_ROOT "src/sofa/testing")

sofa_find_package(Sofa.Config REQUIRED BOTH_SCOPES)

set(HEADER_FILES
    ${SOFATESTINGSRC_ROOT}/config.h.in
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.h
)

set(SOURCE_FILES
    ${SOFATESTINGSRC_ROOT}/initSofa.Testing.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config gtest)


# own include dir, the macro does not handle "independent" package for SofaFramework-style include 
# i.e NOT the name of package first (sofa/testing is wanted, instead of Sofa.Testing/)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}>")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaFramework)
