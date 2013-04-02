cmake_minimum_required(VERSION 2.8)

project("StandardTest")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

    Sofa_test.h 
    Matrix_test.inl
    Mapping_test.h
    
    )

set(SOURCE_FILES

    RigidMapping_test.cpp 
    Matrix_test.cpp 
    OBBTest.cpp 
    ProjectToPlaneConstraint_test.cpp 
    ProjectToLineConstraint_test.cpp 
    
    )

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_PROJECT_TEST")
set(LINKER_DEPENDENCIES SofaEigen2Solver SofaComponent SofaBaseCollision SofaGraph gtest gtest_main )

include(${SOFA_CMAKE_DIR}/post.cmake)

