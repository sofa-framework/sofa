cmake_minimum_required(VERSION 3.1)
project(SofaSimulationTree)

set(HEADER_FILES
    ExportDotVisitor.h
    GNode.h
    GNodeMultiMappingElement.h
    GNodeVisitor.h
    TreeSimulation.h
    init.h
)

set(SOURCE_FILES
    ExportDotVisitor.cpp
    GNode.cpp
    GNodeMultiMappingElement.cpp
    GNodeVisitor.cpp
    TreeSimulation.cpp
    init.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_TREE")

sofa_install_targets(SofaSimulation ${PROJECT_NAME} ${PROJECT_NAME})

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/SofaSimulationTree"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
