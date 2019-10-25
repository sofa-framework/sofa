cmake_minimum_required(VERSION 3.1)

project(SofaDefaultType_test)

set(SOURCE_FILES
    MapMapSparseMatrixEigenUtils_test.cpp
    MatTypes_test.cpp
    VecTypes_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaDefaultType)

add_test(NAME SofaDefaultType_test COMMAND SofaDefaultType_test)
