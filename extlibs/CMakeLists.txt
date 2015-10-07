cmake_minimum_required(VERSION 2.8.12)

set(EXTLIBS_TARGETS
    csparse
    miniFlowVR
    newmat
    tinyxml
)

## Necessary ones
add_subdirectory(SuiteSparse)
add_subdirectory(csparse)
add_subdirectory(eigen-3.2.5 eigen)
add_subdirectory(miniFlowVR)
add_subdirectory(newmat)
add_subdirectory(tinyxml)

## Optional ones
add_subdirectory(CImg)
add_subdirectory(libQGLViewer-2.4.0/QGLViewer QGLViewer)

option(SOFA_BUILD_ARTRACK
"Build the ARTrack library distributed in the extlibs/ directory.  This library
is used only by the ARTrack plugin."  OFF)
if(SOFA_BUILD_ARTRACK)
    add_subdirectory(ARTrack ARTrackLib)
    list(APPEND EXTLIBS_TARGETS ARTrackLib)
endif()

option(SOFA_BUILD_METIS
"Build the metis library distributed in the extlibs/ directory.  It is used only
by the SparseLDLSolver component.  Note that this library is usually available
on Linux distributions." OFF)
if(SOFA_BUILD_METIS)
    add_subdirectory(metis-5.1.0 "metis")
    list(APPEND EXTLIBS_TARGETS metis)
endif()

foreach(target ${EXTLIBS_TARGETS})
    set_target_properties(${target} PROPERTIES DEBUG_POSTFIX "_d")
    set_target_properties(${target} PROPERTIES FOLDER "SofaExtlibs")
endforeach()

# C++11
set(CPP11_EXTLIBS_TARGETS ${EXTLIBS_TARGETS})
list(REMOVE_ITEM CPP11_EXTLIBS_TARGETS csparse)
if(NOT WIN32)
    if(SOFA_USE_C++11)
        foreach(target ${CPP11_EXTLIBS_TARGETS})
            target_compile_options(${target} PRIVATE ${CXX11_FLAG})
            target_compile_options(${target} PRIVATE "-Wno-deprecated-register")
        endforeach()
    endif()
endif()
