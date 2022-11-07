cmake_minimum_required(VERSION 3.12)

set(EXTLIBS_TARGETS
    tinyxml
)

############# extlibs ##############

# Nlohmann JSON (header-only)
set(JSON_ROOT "${CMAKE_CURRENT_LIST_DIR}/json" CACHE PATH "Nlohmann JSON directory")
# STB (header-only)
set(STB_ROOT "${CMAKE_CURRENT_LIST_DIR}/stb" CACHE PATH "STB directory")
# DiffLib
# STB (header-only)
set(DIFFLIB_ROOT "${CMAKE_CURRENT_LIST_DIR}/difflib" CACHE PATH "DiffLib directory")

# TinyXML
find_package(TinyXML QUIET)
if(NOT TinyXML_FOUND)
    add_subdirectory(tinyxml tinyxml)
endif()

####################################

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()
