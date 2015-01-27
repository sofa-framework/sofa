# - Try to find Assimp
# This module defines
#  ASSIMP_FOUND        - Assimp was found
#  ASSIMP_INCLUDE_DIR  - Assimp include directories
#  ASSIMP_LIBRARIES    - the Assimp library
# And on Windows:
#  ASSIMP_DLLS         - the DLL of Assimp

if(WIN32)
    # Use the version provided with the plugin.
    set(ASSIMP_FOUND 1)
    set(ASSIMP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assimp/include)
    if(CMAKE_CL_64)
        set(ASSIMP_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/assimp/lib/x64/assimp.lib)
        set(ASSIMP_DLLS ${CMAKE_CURRENT_SOURCE_DIR}/assimp/lib/x64/Assimp64.dll)
    else()
        set(ASSIMP_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/assimp/lib/x86/assimp.lib)
        set(ASSIMP_DLLS ${CMAKE_CURRENT_SOURCE_DIR}/assimp/lib/x86/Assimp32.dll)
    endif()
else()
    find_path(ASSIMP_INCLUDE_DIR assimp/mesh.h)
    find_library(ASSIMP_LIBRARY assimp)

    if(ASSIMP_INCLUDE_DIR AND ASSIMP_LIBRARY)
        set(ASSIMP_FOUND 1)
        set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY})
    endif()

    if(ASSIMP_FOUND)
        message(STATUS "Found Assimp: ${ASSIMP_LIBRARY}")
    else()
        if(ASSIMP_FIND_REQUIRED)
            message(FATAL_ERROR "Could not find libassimp")
        endif()
    endif()
endif()
