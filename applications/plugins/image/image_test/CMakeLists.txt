include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
)

set(SOURCE_FILES
#   DataImage_test.cpp
    ImageEngine_test.cpp
        )

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

include_directories("${SOFA_APPLICATIONS_PROJECTS_DIR}/Standard_test")

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})
AddCompilerDefinitions("IMAGE_TEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

if(WIN32 AND NOT CMAKE_CL_64)
    AddCompilerDefinitions("EIGEN_DONT_ALIGN")
endif()

# Define a macro to get the absolute path to a directory;
# absolute paths are perfectly OK for test resources.
AddCompilerDefinitions("IMAGETEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

AddLinkerDependencies(SofaTest image)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
