cmake_minimum_required(VERSION 3.1)
project(SixenseHydra)


set(HEADER_FILES
    RazerHydraDriver.h
    config.h
)

set(SOURCE_FILES
    RazerHydraDriver.cpp
    initSixenseHydra.cpp
)

find_package(SofaGeneral REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Sixense REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIXENSE_HYDRA")
target_link_libraries(${PROJECT_NAME} SofaBaseLinearSolver SofaBaseTopology
    SofaMeshCollision SofaSimpleFem SofaConstraint SofaUserInteraction)
include_directories(${SIXENSE_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${SIXENSE_LIBRARY} ${SIXENSE_UTILS_LIBRARY})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT SixenseHydra_libraries
        EXPORT SixenseHydraTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY Scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
