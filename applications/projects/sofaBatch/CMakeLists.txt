cmake_minimum_required(VERSION 2.8)

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

set(LINKER_DEPENDENCIES SofaComponentMain )

if(UNIX)

    list(APPEND LINKER_DEPENDENCIES "dl")

endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

