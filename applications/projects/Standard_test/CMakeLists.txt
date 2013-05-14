include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
add_test(${PROJECT_NAME} COMMAND ${PROJECT_NAME})

if(UNIT-TESTS_BUILD_GTEST OR WIN32)
    include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
endif()

set(COMPILER_DEFINES "SOFA_PROJECT_TEST")
set(LINKER_DEPENDENCIES ObjectCreator SofaEigen2Solver SofaComponentMain SofaBaseCollision SofaSimulationGraph SofaBoundaryCondition gtest gtest_main)

if(WIN32)
	set(COMPILER_FLAGS ${COMPILER_FLAGS} "/MT")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

