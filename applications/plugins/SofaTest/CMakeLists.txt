cmake_minimum_required(VERSION 2.8)

project(SofaTest)

set(HEADER_FILES

    InitPlugin_test.h
    Sofa_test.h
    Mapping_test.h
    MultiMapping_test.h
    Elasticity_test.h
    Elasticity_test.inl
    ForceField_test.h
    BroadPhase_test.h
    PrimitiveCreation.h
	)

set(SOURCE_FILES

    InitPlugin_test.cpp
    Sofa_test.cpp
    Elasticity_test.cpp
    PrimitiveCreation.cpp
	)

if(WIN32)
    # MSVC2012 has some troubles with the way gtest use the STL, this
    # preprocessor macro fix this issue
    add_definitions("-D_VARIADIC_MAX=10")
endif()

set(PYTHON_FILES "")

# Todo

# if(SOFA-PLUGIN_SOFAPYTHON)
#     list(APPEND HEADER_FILES "Python_test.h")
#     list(APPEND SOURCE_FILES "Python_test.cpp")

#     AddLinkerDependencies(SofaPython)

#     # include python files
#     include(./python/python.cmake)
# endif()

find_package(SofaFramework REQUIRED)
find_package(SofaMisc REQUIRED)
find_package(SceneCreator REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

target_link_libraries(${PROJECT_NAME} SceneCreator SofaComponentMisc)
target_link_libraries(${PROJECT_NAME} gtest)

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_TEST -DSOFA_SRC_DIR=\"\\\"${CMAKE_CURRENT_SOURCE_DIR}/../../..\\\"\"")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS ${PROJECT_NAME}
        COMPONENT Flexible_libraries
        EXPORT FlexibleTargets
        PUBLIC_HEADER DESTINATION include/SofaTest
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
