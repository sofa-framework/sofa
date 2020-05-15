cmake_minimum_required(VERSION 3.1)

set(EXTLIBS_TARGETS
    newmat
    tinyxml
)

######## SofaKernel/extlibs ########

# Nlohmann JSON (header-only): see SofaKernel/cmake/Modules/FindJson.cmake
set(JSON_ROOT "${SOFA_KERNEL_SOURCE_DIR}/extlibs/json" CACHE PATH "Nlohmann JSON directory")

# Newmat
find_package(Newmat QUIET)
if(NOT Newmat_FOUND)
    add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/newmat newmat)
endif()

# TinyXML
find_package(TinyXML QUIET)
if(NOT TinyXML_FOUND)
    add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/tinyxml tinyxml)
endif()

####################################

## Necessary ones

option(SOFA_BUILD_MINIFLOWVR "Build the miniFlowVR library distributed in the extlibs/ directory." OFF)
if(SOFA_BUILD_MINIFLOWVR)
    add_subdirectory(miniFlowVR)
    list(APPEND EXTLIBS_TARGETS miniFlowVR)
endif()

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()

