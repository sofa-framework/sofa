cmake_minimum_required(VERSION 3.12)

set(EXTLIBS_TARGETS
    tinyxml
)

######## SofaKernel/extlibs ########

# Nlohmann JSON (header-only): see SofaKernel/cmake/Modules/FindJson.cmake
set(JSON_ROOT "${SOFA_KERNEL_SOURCE_DIR}/extlibs/json" CACHE PATH "Nlohmann JSON directory")
# STB (header-only): see SofaKernel/cmake/Modules/FindSTB.cmake
set(STB_ROOT "${SOFA_KERNEL_SOURCE_DIR}/extlibs/stb" CACHE PATH "STB directory")

# TinyXML
find_package(TinyXML QUIET)
if(NOT TinyXML_FOUND)
    add_subdirectory(${SOFA_KERNEL_SOURCE_DIR}/extlibs/tinyxml tinyxml)
endif()

####################################

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()
