cmake_minimum_required(VERSION 2.8)
project(MeshSTEPLoader)

set(MESHSTEPLOADER_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(MESHSTEPLOADER_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(MESHSTEPLOADER_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    MeshSTEPLoader.h
    ParametricTriangleTopologyContainer.h
    SingleComponent.inl
    SingleComponent.h
    STEPShapeMapping.h
    initMeshSTEPLoader.h)

set(SOURCE_FILES
    MeshSTEPLoader.cpp
    ParametricTriangleTopologyContainer.cpp
    SingleComponent.cpp
    STEPShapeMapping.cpp
    initMeshSTEPLoader.cpp)

set(README_FILES PluginMeshSTEPLoader.txt)

find_package(SofaBase REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCascade REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set(MESHSTEPLOADER_COMPILER_FLAGS "-DSOFA_BUILD_MESHSTEPLOADERPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)
target_link_libraries(${PROJECT_NAME} ${OPENCASCADE_LIBRARIES})
include_directories(${OPENCASCADE_INCLUDE_DIR})

if(WIN32)
    # required for OpenCASCADE
    set(MESHSTEPLOADER_COMPILER_FLAGS "${MESHSTEPLOADER_COMPILER_FLAGS} -DWNT")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_MESHSTEPLOADERPLUGIN")

install(TARGETS ${PROJECT_NAME}
    COMPONENT MeshSTEPLoader_libraries
    EXPORT MeshSTEPLoaderTargets
    RUNTIME DESTINATION "${MESHSTEPLOADER_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${MESHSTEPLOADER_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${MESHSTEPLOADER_INSTALL_LIB_DIR}")

install(DIRECTORY example/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

if(WIN32)
    # Copy OpenCascade DLLs
    file(COPY ${OPENCASCADE_DLLS} DESTINATION bin)
    install(FILES ${OPENCASCADE_DLLS} DESTINATION ${MESHSTEPLOADER_INSTALL_BIN_DIR})
endif()
