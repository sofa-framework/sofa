cmake_minimum_required(VERSION 3.12)
project(MeshSTEPLoader)


set(HEADER_FILES
    MeshSTEPLoader.h
    ParametricTriangleTopologyContainer.h
    STEPShapeMapping.h
    SingleComponent.h
    SingleComponent.inl
    config.h
)

set(SOURCE_FILES
    MeshSTEPLoader.cpp
    ParametricTriangleTopologyContainer.cpp
    STEPShapeMapping.cpp
    SingleComponent.cpp
    initMeshSTEPLoader.cpp
)

set(README_FILES PluginMeshSTEPLoader.txt)

find_package(Sofa.Component.Topology.Container.Constant REQUIRED)
find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCascade REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_${PROJECT_NAME}")
target_link_libraries(${PROJECT_NAME} Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Dynamic)
target_link_libraries(${PROJECT_NAME} ${OPENCASCADE_LIBRARIES})
include_directories(${OPENCASCADE_INCLUDE_DIR})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

if(WIN32)
    # required for OpenCASCADE
    set(MESHSTEPLOADER_COMPILER_FLAGS "${MESHSTEPLOADER_COMPILER_FLAGS} -DWNT")
endif()


install(TARGETS ${PROJECT_NAME}
    COMPONENT MeshSTEPLoader_libraries
    EXPORT MeshSTEPLoaderTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(DIRECTORY example/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

if(WIN32)
    # Copy OpenCascade DLLs
    file(COPY ${OPENCASCADE_DLLS} DESTINATION bin)
    install(FILES ${OPENCASCADE_DLLS} DESTINATION bin)
endif()
