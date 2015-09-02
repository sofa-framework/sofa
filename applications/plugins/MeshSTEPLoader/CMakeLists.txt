include(${SOFA_CMAKE_DIR}/preProject.cmake)

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


## OpenCascade
if(WIN32 AND NOT SOFA-EXTERNAL_OPENCASCADE_PATH STREQUAL "")
    set(OPENCASCADE_INCLUDE_DIR "${SOFA-EXTERNAL_OPENCASCADE_PATH}/ros/inc"
        CACHE INTERNAL "OpenCascade include directory" FORCE)
    set(OPENCASCADE_LIBRARIES_DIR "${SOFA-EXTERNAL_OPENCASCADE_PATH}/ros/win32/vc9/lib"
        CACHE INTERNAL "OpenCascade library directory" FORCE)
    mark_as_advanced(OPENCASCADE_INCLUDE_DIR)
    mark_as_advanced(OPENCASCADE_LIBRARIES_DIR)
else()
    find_package(OpenCascade REQUIRED)
endif()

include_directories(${OPENCASCADE_INCLUDE_DIR})
link_directories(${OPENCASCADE_LIBRARIES_DIR})

if(WIN32)
    #required for OpenCASCADE
    AddCompilerDefinitions("WNT")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_MESHSTEPLOADERPLUGIN")
AddLinkerDependencies(SofaBaseTopology)

set(TKLibraries
    TKBRep
    TKernel
    TKG3d
    TKGeomBase
    TKMath
    TKMesh
    TKPrim
    TKShHealing
    TKSTEP
    TKSTEPBase
    TKTopAlgo
    TKXSBase)

foreach(TKLibrary ${TKLibraries})
    AddLinkerDependencies(${TKLibrary})
endforeach()

# copy dlls on Windows
if(WIN32)
    ## OpenCascade third party
    file(GLOB sharedObjects "${SOFA-EXTERNAL_OPENCASCADE_PATH}/3rdparty/tbb30_018oss/bin/ia32/vc9/*.dll")
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()

    ## OpenCascade core
    file(GLOB sharedObjects "${SOFA-EXTERNAL_OPENCASCADE_PATH}/ros/win32/vc9/bin/*.dll")
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
