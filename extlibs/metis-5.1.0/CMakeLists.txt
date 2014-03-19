include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(METIS_INSTALL FALSE)
set(METIS_LIBRARY_TYPE STATIC)

unset(GKLIB_PATH CACHE)
get_filename_component(GKLIB_PATH "GKlib" ABSOLUTE)
include(GKlib/GKlibSystem.cmake)

include_directories(GKlib)
include_directories(include)
include_directories(libmetis)

file(GLOB metis_sources libmetis/*.c)

add_library(${PROJECT_NAME} STATIC ${metis_sources} ${GKlib_sources})

if(UNIX)
    target_link_libraries(${PROJECT_NAME} m)
endif()

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
