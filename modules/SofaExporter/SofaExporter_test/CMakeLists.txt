cmake_minimum_required(VERSION 3.12)

project(SofaExporter_test)

sofa_find_package(SofaGTestMain REQUIRED)
sofa_find_package(SofaExporter REQUIRED)
sofa_find_package(SofaGeneralImplicitOdeSolver REQUIRED)
sofa_find_package(SofaBaseMechanics REQUIRED)
sofa_find_package(SofaBaseLinearSolver REQUIRED)
sofa_find_package(SofaImplicitOdeSolver REQUIRED)
sofa_find_package(SofaBaseUtils REQUIRED)

set(SOURCE_FILES
    OBJExporter_test.cpp
    STLExporter_test.cpp
    MeshExporter_test.cpp
    WriteState_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGTestMain SofaExporter SofaGeneralImplicitOdeSolver SofaBaseMechanics SofaBaseLinearSolver SofaImplicitOdeSolver SofaBaseUtils)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE "SOFAEXPORTER_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\""
    PRIVATE "SOFAEXPORTER_BUILD_DIR=\"${CMAKE_BINARY_DIR}/\""
    )

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
