cmake_minimum_required(VERSION 2.8)

project(SofaBaseLinearSolver_test)

set(SOURCE_FILES
    Matrix_test.cpp
    Matrix_test.inl
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest SofaBaseLinearSolver)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

if(MSVC)
    AddCompilerDefinitions("EIGEN_DONT_ALIGN")
endif()
