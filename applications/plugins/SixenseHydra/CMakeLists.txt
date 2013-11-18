cmake_minimum_required(VERSION 2.8)

#project("SixenseHydra")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES 
    initSixenseHydra.h 
    RazerHydraDriver.h 
    )

set(SOURCE_FILES
    initSixenseHydra.cpp 
    RazerHydraDriver.cpp 
    )



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set(COMPILER_DEFINES "SOFA_BUILD_SIXENSE_HYDRA")
set(LINKER_DEPENDENCIES SofaPython SofaBaseLinearSolver SofaConstraint SofaUserInteraction SofaSimpleFem SofaBaseTopology SofaGuiCommon SofaMeshCollision SofaH3DBaseNodes H3DAPI HAPI H3DUtil)
IF( H3D_USE_DEPENDENCIES_ONLY )
  # The variables set here must be set by the CMakeLists.txt that sets H3D_USE_DEPENDENCIES_ONLY to true.
  INCLUDE_DIRECTORIES( ${SOFAH3DBASE_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES( ${H3DAPI_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES( ${HAPI_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES( ${H3DUTIL_INCLUDE_DIR} )
  INCLUDE_DIRECTORIES( ${PTHREAD_INCLUDE_DIR} )
ENDIF( H3D_USE_DEPENDENCIES_ONLY )

include_directories("${SOFA_APPLICATIONS_PLUGINS_DIR}/SixenseHydra/include")

#link_directories("${SOFA_APPLICATIONS_PLUGINS_DIR}/SixenseHydra/lib/win32/release_dll")
AddLinkerDependencies("${SOFA_APPLICATIONS_PLUGINS_DIR}/SixenseHydra/lib/win32/release_dll/sixense")
AddLinkerDependencies("${SOFA_APPLICATIONS_PLUGINS_DIR}/SixenseHydra/lib/win32/release_dll/sixense_utils")

include(${SOFA_CMAKE_DIR}/postProject.cmake)