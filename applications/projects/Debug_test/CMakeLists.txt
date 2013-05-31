include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    )

set(SOURCE_FILES

 #   My_test.cpp 
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_PROJECT_TEST")
#AddLinkerDependencies( )

include(${SOFA_CMAKE_DIR}/post.cmake)

