cmake_minimum_required(VERSION 3.12)
project(SofaBaseCollision LANGUAGES CXX)

set(SOFABASECOLLISION_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASECOLLISION_SRC}/config.h.in
    ${SOFABASECOLLISION_SRC}/fwd.h
    ${SOFABASECOLLISION_SRC}/initSofaBaseCollision.h
    ${SOFABASECOLLISION_SRC}/BaseContactMapper.h
    ${SOFABASECOLLISION_SRC}/ContactListener.h
    ${SOFABASECOLLISION_SRC}/DefaultContactManager.h
    ${SOFABASECOLLISION_SRC}/Intersector.h

)

set(SOURCE_FILES
    ${SOFABASECOLLISION_SRC}/initSofaBaseCollision.cpp
    ${SOFABASECOLLISION_SRC}/BaseContactMapper.cpp
    ${SOFABASECOLLISION_SRC}/ContactListener.cpp
    ${SOFABASECOLLISION_SRC}/DefaultContactManager.cpp
)

sofa_find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore)

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Collision.Model, Sofa.Component.Collision.Detection.Algorithm and Sofa.Component.Collision.Detection.Intersection instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Collision.Model REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Algorithm)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Detection.Intersection)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFABASECOLLISION_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFABASECOLLISION_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaBaseCollision_test)
endif()
