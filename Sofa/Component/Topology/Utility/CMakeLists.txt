cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Topology.Utility LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR "src/sofa/component/topology/utility")

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologicalChangeProcessor.h
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologyBoundingTrasher.h
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologyBoundingTrasher.inl
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologyChecker.h
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologicalChangeProcessor.cpp
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologyBoundingTrasher.cpp
    ${SOFACOMPONENTTOPOLOGYUTILITY_SOURCE_DIR}/TopologyChecker.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(ZLIB BOTH_SCOPES)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.Component.Topology.Container.Dynamic)
if(ZLIB_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_TOPOLOGY_UTILITY_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_TOPOLOGY_UTILITY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
