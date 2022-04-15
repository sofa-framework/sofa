cmake_minimum_required(VERSION 3.12)
project(Sofa.SimulationGraph)

set(SOURCE_FILES
    empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Simulation.Graph instead.")

sofa_find_package(Sofa.Simulation.Graph REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Graph)

# Create alias to support compatibility, starting from v21.06
# This alias will be deleted in v21.12
add_library(SofaSimulationGraph ALIAS Sofa.SimulationGraph)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
