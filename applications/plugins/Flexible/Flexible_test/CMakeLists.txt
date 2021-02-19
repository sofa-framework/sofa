cmake_minimum_required(VERSION 3.12)

project(Flexible_test)

find_package(SofaTest REQUIRED)

set(HEADER_FILES
    StrainMapping_test.h
)

set(SOURCE_FILES
    AffineDeformationMapping_test.cpp
    AffinePatch_test.cpp
    CauchyStrainMapping_test.cpp
    CorotationalStrainMapping_test.cpp
    FramesBeamMaterial_test.cpp
    GreenStrainMapping_test.cpp
    HexahedraMaterial_test.cpp
    InvariantMapping_test.cpp
    Material_test.cpp
    MooneyRivlinHexahedraMaterial_test.cpp
    NeoHookeHexahedraMaterial_test.cpp
    Patch_test.cpp
    PointDeformationMapping_test.cpp
    PrincipalStretchesMapping_test.cpp
    RigidDeformationMapping_test.cpp
    StabilizedNeoHookeHexahedraMaterial_test.cpp
    TetrahedraMaterial_test.cpp
    TetrahedronVolumeMapping_test.cpp
)

set(PYTHON_FILES
    scenes/python/reInitMapping.py
)

if(WIN32)
    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")
endif()

if(SOFATEST_HAVE_SOFAPYTHON)
    list(APPEND SOURCE_FILES python_test_list.cpp)
    add_definitions("-DFLEXIBLE_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes/python\"")
endif()

add_definitions("-DFLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

find_package(CImgPlugin REQUIRED)
find_package(image QUIET)

if(image_FOUND)
    list(APPEND SOURCE_FILES
            Engine_test.cpp
            ShapeFunction_test.cpp
        )
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} Flexible SofaTest SofaGTestMain CImgPlugin)

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
    target_compile_definitions(${PROJECT_NAME} PUBLIC "FLEXIBLE_TEST_WITH_IMAGE")
endif()


# #Needed by python test
# AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
