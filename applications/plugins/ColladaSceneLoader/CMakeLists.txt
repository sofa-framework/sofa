cmake_minimum_required(VERSION 3.1)
project(ColladaSceneLoader)

set(HEADER_FILES
    SceneColladaLoader.h
    config.h
)

set(SOURCE_FILES
    SceneColladaLoader.cpp
    initPlugin.cpp
)

set(README_FILES ColladaSceneLoader.txt)

find_package(SofaBase REQUIRED)
find_package(SofaGeneral REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Assimp REQUIRED)
find_package(Flexible QUIET)
find_package(image QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaGeneralRigid SofaLoader SofaSimulationCommon SofaBaseMechanics SofaRigid SofaBoundaryCondition SofaBaseTopology SofaBaseVisual SofaOpenglVisual SofaDeformable SofaMeshCollision)
set(COLLADASCENELOADER_COMPILE_FLAGS "-DSOFA_BUILD_COLLADASCENELOADER")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories(${ASSIMP_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${ASSIMP_LIBRARIES})

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
    set(COLLADASCENELOADER_COMPILE_FLAGS "${COLLADASCENELOADER_COMPILE_FLAGS} -DSOFA_HAVE_PLUGIN_FLEXIBLE")
else()
    message(STATUS "ColladaSceneLoader: could not find Flexible")
endif()

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
    set(COLLADASCENELOADER_COMPILE_FLAGS "${COLLADASCENELOADER_COMPILE_FLAGS} -DSOFA_HAVE_PLUGIN_IMAGE")
else()
    message(STATUS "ColladaSceneLoader: could not find image")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COLLADASCENELOADER_COMPILE_FLAGS}")

install(TARGETS ${PROJECT_NAME}
        COMPONENT ColladaSceneLoader_libraries
        EXPORT ColladaSceneLoaderTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY Demos/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

# Copy Assimp DLLs on Windows
if(WIN32)
    if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            file(COPY ${ASSIMP_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    else()                      # Single-config generator (nmake)
        file(COPY ${ASSIMP_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
    install(FILES ${ASSIMP_DLLS} DESTINATION bin)
endif()
