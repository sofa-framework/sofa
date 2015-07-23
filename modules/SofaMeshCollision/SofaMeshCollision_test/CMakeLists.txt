include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

 #alphabetical order
   BaryMapper_test.cpp
)

if(APPLE)
#    set(RC_FILES "runSOFA.icns")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${RC_FILES} ${PYTHON_FILES})

AddLinkerDependencies(
    SofaGTestMain
    SofaTest
)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
