cmake_minimum_required(VERSION 3.12)


############# extlibs ##############

# Nlohmann JSON (header-only)
set(JSON_ROOT "${CMAKE_CURRENT_LIST_DIR}/json" CACHE PATH "Nlohmann JSON directory")
# STB (header-only)
set(STB_ROOT "${CMAKE_CURRENT_LIST_DIR}/stb" CACHE PATH "STB directory")
# DiffLib
set(DIFFLIB_ROOT "${CMAKE_CURRENT_LIST_DIR}/difflib" CACHE PATH "DiffLib directory")

####################################
