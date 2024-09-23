cmake_minimum_required(VERSION 3.22)

project(Sofa.DefaultType_test)

set(HEADER_FILES
    DataMockup.h
)


set(SOURCE_FILES
    MapMapSparseMatrixEigenUtils_test.cpp
    QuaternionIntegration_test.cpp
    RigidDeriv_test.cpp
    TypeInfoRepository_test.cpp
    TypeInfoRepository_tu1_test.cpp
    TypeInfoRepository_tu2_test.cpp
    TypeInfo_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Testing Sofa.DefaultType)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
