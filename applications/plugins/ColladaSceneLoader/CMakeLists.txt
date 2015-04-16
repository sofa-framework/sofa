cmake_minimum_required(VERSION 2.8)
project(ColladaSceneLoader)

set(COLLADASCENELOADER_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(COLLADASCENELOADER_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(COLLADASCENELOADER_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
	initPlugin.h
    SceneColladaLoader.h)

set(SOURCE_FILES
	initPlugin.cpp
    SceneColladaLoader.cpp)
	
set(README_FILES
	ColladaSceneLoader.txt)

find_package(SofaBase REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Assimp REQUIRED)
find_package(Flexible QUIET)
find_package(image QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COLLADASCENELOADER")
target_link_libraries(${PROJECT_NAME} SofaLoader SofaSimulationCommon SofaBaseMechanics SofaRigid SofaBoundaryCondition SofaBaseTopology SofaBaseVisual SofaOpenglVisual SofaDeformable SofaMeshCollision)

include_directories(${ASSIMP_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${ASSIMP_LIBRARIES})

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
endif()

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT ColladaSceneLoader_libraries
    EXPORT ColladaSceneLoaderTargets
    RUNTIME DESTINATION "${COLLADASCENELOADER_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${COLLADASCENELOADER_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${COLLADASCENELOADER_INSTALL_LIB_DIR}")

install(DIRECTORY Demos/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

if(WIN32)
    # Copy OpenCascade DLLs
    file(COPY ${ASSIMP_DLLS} DESTINATION bin)
    install(FILES ${ASSIMP_DLLS} DESTINATION ${COLLADASCENELOADER_INSTALL_BIN_DIR})
endif()
