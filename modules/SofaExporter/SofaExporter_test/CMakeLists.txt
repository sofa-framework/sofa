cmake_minimum_required(VERSION 3.1)

project(SofaExporter_test)

find_package(SofaBase REQUIRED)
find_package(SofaCommon REQUIRED)
find_package(SofaGeneral REQUIRED)
find_package(SofaExporter REQUIRED)
find_package(SofaGTestMain REQUIRED)

set(SOURCE_FILES
    OBJExporter_test.cpp
    STLExporter_test.cpp
    MeshExporter_test.cpp
    WriteState_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGTestMain SofaExporter SofaGeneralImplicitOdeSolver SofaBaseMechanics SofaBaseLinearSolver SofaImplicitOdeSolver SofaBase SofaCommon SofaGeneral)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE "SOFAEXPORTER_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\""
    PRIVATE "SOFAEXPORTER_BUILD_DIR=\"${CMAKE_BINARY_DIR}/\""
    )

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
