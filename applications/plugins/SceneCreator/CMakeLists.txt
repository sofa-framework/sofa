cmake_minimum_required(VERSION 2.8.12)
project(SceneCreator)

set(SCENECREATOR_MAJOR_VERSION 0)
set(SCENECREATOR_MINOR_VERSION 0)
set(SCENECREATOR_BUILD_VERSION 1)

set(SCENECREATOR_VERSION ${SCENECREATOR_MAJOR_VERSION}.${SCENECREATOR_MINOR_VERSION}.${SCENECREATOR_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


set(HEADER_FILES
    GetAssembledSizeVisitor.h
    GetVectorVisitor.h
    SceneCreator.h
    initSceneCreator.h
)

set(SOURCE_FILES
    GetAssembledSizeVisitor.cpp
    GetVectorVisitor.cpp
    SceneCreator.cpp
    initSceneCreator.cpp
)

find_package(SofaSimulation REQUIRED)
find_package(SofaMisc REQUIRED)

set(SCENECREATOR_DEPENDENCY_INCLUDE_DIR ${SofaSimulation_INCLUDE_DIRS} ${SofaMisc_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SceneCreatorConfig.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/SceneCreator/SceneCreatorConfig.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SceneCreator/SceneCreatorConfig.h
         DESTINATION "include/SceneCreator")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree SofaComponentMisc SofaSimulationGraph SofaLoader SofaEngine)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SCENECREATOR_DEPENDENCY_INCLUDE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SCENECREATOR_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SCENECREATOR")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


install(TARGETS ${PROJECT_NAME}
        COMPONENT SceneCreator_libraries
        EXPORT SceneCreatorTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")


include(CMakePackageConfigHelpers)
# SceneCreatorConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(SceneCreatorConfigVersion.cmake
                                 VERSION ${SCENECREATOR_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SceneCreatorConfigVersion.cmake
        DESTINATION lib/cmake/SceneCreator)


# SceneCreatorConfig.cmake (build tree)
set(SCENECREATOR_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SceneCreatorConfig.cmake.in
                              SceneCreatorConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS SCENECREATOR_INCLUDE_DIR)

# SceneCreatorConfig.cmake (installed)
set(SCENECREATOR_INCLUDE_DIR include)
configure_package_config_file(SceneCreatorConfig.cmake.in
                              SceneCreatorConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SceneCreator
                              PATH_VARS SCENECREATOR_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/SceneCreatorConfigInstall.cmake
        DESTINATION lib/cmake/SceneCreator
        RENAME SceneCreatorConfig.cmake)

# SceneCreatorTargets.cmake (installed)
install(EXPORT SceneCreatorTargets
        DESTINATION lib/cmake/SceneCreator)
