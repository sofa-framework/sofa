cmake_minimum_required(VERSION 3.22)

project(SofaCHOLMOD_test)

set(SOURCE_FILES
    EigenCholmodSupernodalLLT_test.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME}
    Sofa.Testing
    SofaCHOLMOD
    Sofa.Component.LinearSolver.Ordering
    Eigen3::Eigen
)
