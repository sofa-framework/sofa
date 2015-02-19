cmake_minimum_required(VERSION 2.8)
project(SixenseHydra)

set(SIXENSEHYDRA_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SIXENSEHYDRA_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SIXENSEHYDRA_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    initSixenseHydra.h
    RazerHydraDriver.h)

set(SOURCE_FILES
    initSixenseHydra.cpp
    RazerHydraDriver.cpp)

find_package(SofaGeneral REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Sixense REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIXENSE_HYDRA")
target_link_libraries(${PROJECT_NAME} SofaBaseLinearSolver SofaBaseTopology
    SofaMeshCollision SofaSimpleFem SofaConstraint SofaUserInteraction)
include_directories(${SIXENSE_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${SIXENSE_LIBRARY} ${SIXENSE_UTILS_LIBRARY})

install(TARGETS ${PROJECT_NAME}
    COMPONENT SixenseHydra_libraries
    EXPORT SixenseHydraTargets
    RUNTIME DESTINATION "${SIXENSEHYDRA_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SIXENSEHYDRA_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SIXENSEHYDRA_INSTALL_LIB_DIR}")
