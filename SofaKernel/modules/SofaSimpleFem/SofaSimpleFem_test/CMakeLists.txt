cmake_minimum_required(VERSION 3.1)

project(SofaSimpleFem_test)

set(SOURCE_FILES
    HexahedronFEMForceField_test.cpp
    TetrahedronFEMForceField_test.cpp
    TetrahedronDiffusionFEMForceField_test.cpp
)


if (WIN32
    AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"  # visual studio
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)  # before  VS2013 ==> VS2012
    find_package(Boost REQUIRED)
    if (Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
        link_libraries(${Boost_LIBRARIES})
    endif()
endif()




add_definitions("-DSOFASIMPLEFEM_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
