include(${SOFA_CMAKE_DIR}/preProject.cmake)

# FF temporarily deactivated InertiaAlign_test, as well as add_executable since it does not accept 0 parameter
set(SOURCE_FILES
#   InertiaAlign_test.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

#if(SOFA-PLUGIN_SOFAPYTHON)
#    AddCompilerDefinitions("REGISTRATION_TEST_PYTHON_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes/python\"")
#    list(APPEND SOURCE_FILES "python_test_list.cpp")
#    AddLinkerDependencies(SofaPython)
#    include(./scenes/python/python.cmake)
#endif()

#add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${RC_FILES} ${PYTHON_FILES})

AddCompilerDefinitions("FLEXIBLE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
AddLinkerDependencies(SofaTest Registration)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

