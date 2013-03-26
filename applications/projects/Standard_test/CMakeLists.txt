cmake_minimum_required(VERSION 2.8)

project("StandardTest")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    Sofa_test.h 
    Matrix_test.inl

    )

set(SOURCE_FILES

    Matrix_test.cpp 
    OBBTest.cpp 
    ProjectToPlaneConstraint_test.cpp 
    ProjectToLineConstraint_test.cpp 
    
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_PROJECT_TEST")
#set(LINKER_DEPENDENCIES  )

include(${SOFA_CMAKE_DIR}/post.cmake)

