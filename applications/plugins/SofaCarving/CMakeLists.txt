cmake_minimum_required(VERSION 2.8)
project(SofaCarving)


set(HEADER_FILES
    CarvingManager.h
    initSofaCarving.h
)

set(SOURCE_FILES
    CarvingManager.cpp
    initSofaCarving.cpp
)

set(README_FILES SofaCarving.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACARVING")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)

install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaCarving_libraries
    EXPORT SofaCarvingTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
