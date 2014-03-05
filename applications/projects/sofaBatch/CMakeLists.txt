include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    sofaBatch.cpp

    )

if(APPLE)

    set(RC_FILES "runSOFA.icns")

else()

    set(RC_FILES "sofa.rc")

endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${RC_FILES})

AddLinkerDependencies(SofaComponentMain )

if(UNIX)

    AddLinkerDependencies("dl")

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

