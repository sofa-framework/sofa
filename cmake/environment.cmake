
# Path variables
set(SOFA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Path to the Sofa cmake directory")
set(SOFA_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "Path to the Sofa source directory")
set(SOFA_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "Path to the Sofa build directory")
set(SOFA_BIN_DIR "${SOFA_BUILD_DIR}/bin" CACHE INTERNAL "Path to the Sofa bin directory")
set(SOFA_BIN_PLUGINS_DIR "${SOFA_BUILD_DIR}/bin/plugins" CACHE INTERNAL "Path to the plugins directory")
if(WIN32)
    set(SOFA_INC_DIR "${SOFA_SRC_DIR}/include" CACHE INTERNAL "Path to the Sofa include directory")
endif()
set(SOFA_LIB_DIR "${SOFA_BUILD_DIR}/lib" CACHE INTERNAL "Path to the Sofa lib directory")
set(SOFA_EXTLIBS_DIR "${SOFA_SRC_DIR}/extlibs" CACHE INTERNAL "Path to the Sofa extlibs directory")
set(SOFA_SHARE_DIR "${SOFA_SRC_DIR}/share" CACHE INTERNAL "Path to the Sofa share directory")
set(SOFA_FRAMEWORK_DIR "${SOFA_SRC_DIR}/framework" CACHE INTERNAL "Path to the Sofa framework directory")
set(SOFA_MODULES_DIR "${SOFA_SRC_DIR}/modules" CACHE INTERNAL "Path to the Sofa modules directory")
set(SOFA_APPLICATIONS_DIR "${SOFA_SRC_DIR}/applications" CACHE INTERNAL "Path to the Sofa applications directory")
set(SOFA_APPLICATIONS_DEV_DIR "${SOFA_SRC_DIR}/applications-dev" CACHE INTERNAL "Path to the Sofa applications-dev directory")
set(SOFA_APPLICATIONS_PLUGINS_DIR "${SOFA_APPLICATIONS_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications plugins directory")
set(SOFA_APPLICATIONS_DEV_PLUGINS_DIR "${SOFA_APPLICATIONS_DEV_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications-dev plugins directory")
set(SOFA_APPLICATIONS_PROJECTS_DIR "${SOFA_APPLICATIONS_DIR}/projects" CACHE INTERNAL "Path to the Sofa applications projects directory")
set(SOFA_APPLICATIONS_DEV_PROJECTS_DIR "${SOFA_APPLICATIONS_DEV_DIR}/projects" CACHE INTERNAL "Path to the Sofa applications-dev projects directory")
set(SOFA_CUDA_DIR "${SOFA_APPLICATIONS_DIR}/plugins/SofaCUDA" CACHE INTERNAL "Path to the SofaCuda directory")

# CMake modules path, for our FindXXX.cmake modules
list(APPEND CMAKE_MODULE_PATH ${SOFA_CMAKE_DIR}/Modules)

# Misc
set(SOFA_VERSION_NUM "1_0" CACHE STRING "Version number for this build.")
file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/misc/")
file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/misc/include")
file(MAKE_DIRECTORY "${SOFA_BUILD_DIR}/misc/include/sofa")

## OS-specific
if(WIN32)
    if(CMAKE_CL_64)
        set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/win64/" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
    else()
        set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/win32/" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
    endif()
endif()
if(XBOX)
    set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/xbox/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
endif()
if(PS3)
    set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/ps3/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
endif()

# disable every pre-enabled modules
foreach(dependency ${GLOBAL_DEPENDENCIES})
    unset(GLOBAL_PROJECT_ENABLED_${dependency} CACHE)
endforeach()

# Clear the internal cache variables that we regenerate each time
unset(GLOBAL_DEPENDENCIES CACHE)        # dependency database
unset(GLOBAL_COMPILER_DEFINES CACHE)    #
unset(GLOBAL_INCLUDE_DIRECTORIES CACHE) #

## Output directories
# This macro sets the output directory for each build type
macro(sofa_set_output_directory target_type directory)
    set(CMAKE_${target_type}_OUTPUT_DIRECTORY "${directory}")
    set(CMAKE_${target_type}_OUTPUT_DIRECTORY_DEBUG "${directory}")
    set(CMAKE_${target_type}_OUTPUT_DIRECTORY_RELEASE "${directory}")
    set(CMAKE_${target_type}_OUTPUT_DIRECTORY_RELWITHDEBINFO "${directory}")
    set(CMAKE_${target_type}_OUTPUT_DIRECTORY_MINSIZEREL "${directory}")
endmacro()
# Set the output directory for each target type
sofa_set_output_directory(RUNTIME "${SOFA_BIN_DIR}") # Executables
if(UNIX)
    sofa_set_output_directory(LIBRARY "${SOFA_LIB_DIR}") # Dynamic libraries
else()
    sofa_set_output_directory(LIBRARY "${SOFA_BIN_DIR}") # Dynamic libraries
    sofa_set_output_directory(ARCHIVE "${SOFA_LIB_DIR}") # Static libraries
endif()
