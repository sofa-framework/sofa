include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

   AffinePatch_test.cpp
   PointDeformationMapping_test.cpp
   Patch_test.cpp
   StrainMappings_test.cpp
   RigidDeformationMapping_test.cpp
   InitLinearMapping.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})
AddCompilerDefinitions("FLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
AddLinkerDependencies(Flexible SofaTest)
AddLinkerDependencies(SofaPython)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
