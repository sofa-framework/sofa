cmake_minimum_required(VERSION 3.1)
project(SofaExporter VERSION 1.0 LANGUAGES CXX)

find_package(SofaSimulation REQUIRED)

set(HEADER_FILES
    src/SofaExporter/config.h
    src/SofaExporter/initExporter.h
    )

set(SOURCE_FILES
    src/SofaExporter/initExporter.cpp
    )

set(EXTRA_FILES
    README.md
    examples/MeshExporter.scn
    examples/STLExporter.scn
    examples/OBJExporter.scn
    )

list(APPEND HEADER_FILES
    src/SofaExporter/BlenderExporter.h
    src/SofaExporter/BlenderExporter.inl
    src/SofaExporter/MeshExporter.h
    src/SofaExporter/OBJExporter.h
    src/SofaExporter/STLExporter.h
    src/SofaExporter/VTKExporter.h
    src/SofaExporter/WriteState.h
    src/SofaExporter/WriteState.inl
    src/SofaExporter/WriteTopology.h
    src/SofaExporter/WriteTopology.inl
    )

list(APPEND SOURCE_FILES
    src/SofaExporter/BlenderExporter.cpp
    src/SofaExporter/MeshExporter.cpp
    src/SofaExporter/OBJExporter.cpp
    src/SofaExporter/STLExporter.cpp
    src/SofaExporter/VTKExporter.cpp
    src/SofaExporter/WriteState.cpp
    src/SofaExporter/WriteTopology.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_EXPORTER")
target_compile_definitions(${PROJECT_NAME} PUBLIC "-DSOFA_HAVE_SOFAEXPORTER")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")

sofa_generate_package(
    NAME SofaExporter
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src/SofaExporter"
    INCLUDE_INSTALL_DIR "SofaExporter/SofaExporter"
    )

# Enable tests if SofaTest found
# Will pass only in out-of-SOFA builds
# See modules/tests/CMakeLists.txt if building through SOFA
find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(SofaExporter_test)
endif()
