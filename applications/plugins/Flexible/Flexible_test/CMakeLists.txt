include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
)

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
        )

if(WIN32)
    list(APPEND HEADER_FILES "stdafx.h")
    list(APPEND SOURCE_FILES "stdafx.cpp")
endif()

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

if(SOFA-PLUGIN_SOFAPYTHON)
    AddCompilerDefinitions("FLEXIBLE_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes/python\"")
    list(APPEND SOURCE_FILES "python_test_list.cpp")
    AddLinkerDependencies(SofaPython)
    include(./scenes/python/python.cmake)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})
AddCompilerDefinitions("FLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

if(WIN32 AND NOT CMAKE_CL_64)
    AddCompilerDefinitions("EIGEN_DONT_ALIGN")
endif()

AddLinkerDependencies(Flexible SofaTest image Compliant)

#Needed by python test
AddCompilerDefinitions("SOFA_SRC_DIR=${SOFA_SRC_DIR}")

include(${SOFA_CMAKE_DIR}/postProject.cmake)
