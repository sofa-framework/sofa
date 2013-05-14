include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    )

set(SOURCE_FILES

 #   My_test.cpp 
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_PROJECT_TEST")
#set(LINKER_DEPENDENCIES  )

include(${SOFA_CMAKE_DIR}/post.cmake)

