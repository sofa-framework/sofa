cmake_minimum_required(VERSION 2.8)

project(Flexible_test)

set(SOURCE_FILES
   AffinePatch_test.cpp
   AffineDeformationMapping_test.cpp
   FramesBeamMaterial_test.cpp
   HexahedraMaterial_test.cpp
   Material_test.cpp
   MooneyRivlinHexahedraMaterial_test.cpp
   NeoHookeHexahedraMaterial_test.cpp
   PointDeformationMapping_test.cpp
   Patch_test.cpp
   StrainMappings_test.cpp
   RigidDeformationMapping_test.cpp
   ShapeFunction_test.cpp
   StabilizedNeoHookeHexahedraMaterial_test.cpp
   TetrahedraMaterial_test.cpp
   TetrahedronVolumeMapping_test.cpp)

if(WIN32)
    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")
endif()

find_package(SofaPython QUIET)

if(SofaPython_FOUND)
    list(APPEND SOURCE_FILES python_test_list.cpp)
    add_definitions("-DFLEXIBLE_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes/python\"")
    include(./scenes/python/python.cmake) # Defines PYTHON_FILES
endif()

add_definitions("-DFLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

# find_package(image QUIET)
# find_package(Compliant QUIET)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} Flexible SofaTest gtest_main)
target_link_libraries(${PROJECT_NAME} image)
# target_link_libraries(${PROJECT_NAME} DiffusionSolver)

# ???
if(WIN32 AND NOT CMAKE_CL_64)
    add_definitions("-DEIGEN_DONT_ALIGN")
endif()

# #Needed by python test
# AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
