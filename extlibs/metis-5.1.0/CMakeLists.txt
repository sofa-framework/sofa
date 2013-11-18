include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories(GKlib)
include_directories(include)
include_directories(libmetis)

file(GLOB metis_sources libmetis/*.c)
file(GLOB GKlib_sources GKlib/*.c)

add_library(${PROJECT_NAME} STATIC ${metis_sources} ${GKlib_sources})

if(UNIX)
    target_link_libraries(metis m)
endif()

if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
