include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    StrainMappings_test.cpp
    AffinePatch_test.cpp
    Patch_test.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})
AddCompilerDefinitions("FLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
AddLinkerDependencies(Flexible SofaTest)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
