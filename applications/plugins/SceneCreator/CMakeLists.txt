cmake_minimum_required(VERSION 2.8.12)
project(SceneCreator)

set(SCENECREATOR_MAJOR_VERSION 0)
set(SCENECREATOR_MINOR_VERSION 0)
set(SCENECREATOR_BUILD_VERSION 1)

set(SCENECREATOR_VERSION ${SCENECREATOR_MAJOR_VERSION}.${SCENECREATOR_MINOR_VERSION}.${SCENECREATOR_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SCENECREATOR_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(SCENECREATOR_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(SCENECREATOR_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(SCENECREATOR_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/SceneCreator)

set(SCENECREATOR_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SCENECREATOR_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SCENECREATOR_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SCENECREATOR_INSTALL_CMAKE_DIR   ${SCENECREATOR_INSTALL_LIB_DIR}/cmake/SceneCreator )

set(HEADER_FILES
    initSceneCreator.h
    SceneCreator.h
    GetVectorVisitor.h
    GetAssembledSizeVisitor.h)

set(SOURCE_FILES
    initSceneCreator.cpp
    SceneCreator.cpp
    GetVectorVisitor.cpp
    GetAssembledSizeVisitor.cpp)

find_package(SofaSimulation REQUIRED)
find_package(SofaMisc REQUIRED)

set(SCENECREATOR_DEPENDENCY_INCLUDE_DIR ${SofaSimulation_INCLUDE_DIRS} ${SofaMisc_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SceneCreator.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/SceneCreator/SceneCreator.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/SceneCreator/SceneCreator.h
         DESTINATION "${SCENECREATOR_INSTALL_INCLUDE_DIR}/SceneCreator")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationTree SofaComponentMisc SofaSimulationGraph SofaLoader)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SCENECREATOR_DEPENDENCY_INCLUDE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SCENECREATOR_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SCENECREATOR")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


install(TARGETS ${PROJECT_NAME}
        COMPONENT SceneCreator_libraries
        EXPORT  SceneCreatorTargets
        RUNTIME DESTINATION "${SCENECREATOR_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SCENECREATOR_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SCENECREATOR_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SCENECREATOR_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")


include(CMakePackageConfigHelpers)
# SceneCreatorConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${SCENECREATOR_BUILD_CMAKE_DIR}/SceneCreatorConfigVersion.cmake
                                 VERSION ${SCENECREATOR_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${SCENECREATOR_BUILD_CMAKE_DIR}/SceneCreatorConfigVersion.cmake
        DESTINATION ${SCENECREATOR_INSTALL_CMAKE_DIR})


# SceneCreatorConfig.cmake (build tree)
set(SCENECREATOR_INCLUDE_DIR "${SCENECREATOR_BUILD_INCLUDE_DIR}")
configure_package_config_file(SceneCreatorConfig.cmake.in
                              ${SCENECREATOR_BUILD_CMAKE_DIR}/SceneCreatorConfig.cmake
                              INSTALL_DESTINATION ${SCENECREATOR_BUILD_CMAKE_DIR}
                              PATH_VARS SCENECREATOR_INCLUDE_DIR)

# SceneCreatorConfig.cmake (installed)
set(SCENECREATOR_INCLUDE_DIR "${SCENECREATOR_INSTALL_INCLUDE_DIR}")
configure_package_config_file(SceneCreatorConfig.cmake.in
                              ${SCENECREATOR_BUILD_CMAKE_DIR}/SceneCreatorConfigInstall.cmake
                              INSTALL_DESTINATION ${SCENECREATOR_INSTALL_CMAKE_DIR}
                              PATH_VARS SCENECREATOR_INCLUDE_DIR)

install(FILES ${SCENECREATOR_BUILD_CMAKE_DIR}/SceneCreatorConfigInstall.cmake
        DESTINATION ${SCENECREATOR_INSTALL_CMAKE_DIR}
        RENAME SceneCreatorConfig.cmake)

# SceneCreatorTargets.cmake (installed)
install(EXPORT SceneCreatorTargets
        DESTINATION ${SCENECREATOR_INSTALL_CMAKE_DIR})
