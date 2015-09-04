cmake_minimum_required(VERSION 2.8)

project(SofaTest)

set(SOFATEST_MAJOR_VERSION 0)
set(SOFATEST_MINOR_VERSION 0)
set(SOFATEST_BUILD_VERSION 1)

set(SOFATEST_VERSION ${SOFATEST_MAJOR_VERSION}.${SOFATEST_MINOR_VERSION}.${SOFATEST_BUILD_VERSION} CACHE INTERNAL "SofaTest version")

set(HEADER_FILES
    BroadPhase_test.h
    DataEngine_test.h
    Elasticity_test.h
    Elasticity_test.inl
    ForceField_test.h
    InitPlugin_test.h
    Mapping_test.h
    MultiMapping_test.h
    PrimitiveCreation.h
    Sofa_test.h
)

set(SOURCE_FILES
    Elasticity_test.cpp
    InitPlugin_test.cpp
    PrimitiveCreation.cpp
    Sofa_test.cpp
)

if(WIN32)
    # MSVC2012 has some troubles with the way gtest use the STL, this
    # preprocessor macro fix this issue
    add_definitions("-D_VARIADIC_MAX=10")
endif()

set(PYTHON_FILES "")

find_package(SofaFramework REQUIRED)
find_package(SofaMisc REQUIRED)
find_package(SceneCreator REQUIRED)

find_package(SofaPython QUIET)
if(SofaPython_FOUND)
    list(APPEND HEADER_FILES "Python_test.h")
    list(APPEND SOURCE_FILES "Python_test.cpp")
    set(PYTHON_FILES
        python/SofaTest/Macro.py
        python/SofaTest/__init__.py
    )
else()
    message(STATUS "SofaTest: could not find SofaPython, won't build Python_test")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()

target_link_libraries(${PROJECT_NAME} SceneCreator SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} gtest)

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TEST -DSOFA_SRC_DIR=\"\\\"${CMAKE_CURRENT_SOURCE_DIR}/../../..\\\"\"")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaTest_libraries
        EXPORT SofaTestTargets
        PUBLIC_HEADER DESTINATION include/SofaTest
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

include(CMakePackageConfigHelpers)

# SofaTestVersion.cmake
write_basic_package_version_file(${CMAKE_BINARY_DIR}/SofaTest/SofaTestVersion.cmake
                                 VERSION ${SOFATEST_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
install(FILES ${CMAKE_BINARY_DIR}/SofaTest/SofaTestVersion.cmake
        DESTINATION lib/cmake/SofaTest)

# SofaTestTargets.cmake
install(EXPORT SofaTestTargets
        DESTINATION lib/cmake/SofaTest)

# SofaTestConfig.cmake (build directory)
set(SOFATEST_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaTestConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/SofaTest/SofaTestConfig.cmake
                              INSTALL_DESTINATION Nope
                              PATH_VARS SOFATEST_INCLUDE_DIR)

# SofaTestConfig.cmake (install directory)
set(SOFATEST_INCLUDE_DIR include)
configure_package_config_file(SofaTestConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/SofaTest/SofaTestConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaTest
                              PATH_VARS SOFATEST_INCLUDE_DIR)
install(FILES ${CMAKE_BINARY_DIR}/SofaTest/SofaTestConfigInstall.cmake
        DESTINATION lib/cmake/SofaTest
        RENAME SofaTestConfig.cmake)
