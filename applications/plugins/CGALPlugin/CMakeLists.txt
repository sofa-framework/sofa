cmake_minimum_required(VERSION 3.1)
project(CGALPlugin)

set(CGALPLUGIN_MAJOR_VERSION 0)
set(CGALPLUGIN_MINOR_VERSION 1)
set(CGALPLUGIN_VERSION ${CGALPLUGIN_MAJOR_VERSION}.${CGALPLUGIN_MINOR_VERSION})

set(HEADER_FILES
    DecimateMesh.h
    DecimateMesh.inl
    MeshGenerationFromPolyhedron.h
    MeshGenerationFromPolyhedron.inl
    TriangularConvexHull3D.h
    TriangularConvexHull3D.inl
    CylinderMesh.h
    CylinderMesh.inl
)

set(SOURCE_FILES
    DecimateMesh.cpp
    MeshGenerationFromPolyhedron.cpp
    TriangularConvexHull3D.cpp
    initCGALPlugin.cpp
    CylinderMesh.cpp
)

set(README_FILES CGALPlugin.txt)


### Dependencies


find_package(SofaFramework REQUIRED)
find_package(image QUIET)
if(image_FOUND)
    list(APPEND HEADER_FILES MeshGenerationFromImage.h)
    list(APPEND HEADER_FILES MeshGenerationFromImage.inl)
    list(APPEND SOURCE_FILES MeshGenerationFromImage.cpp)
    find_package(CGAL REQUIRED COMPONENTS ImageIO)
else()
    message(STATUS "CGALPlugin: could not find image, won't build MeshGenerationFromImage")
    find_package(CGAL REQUIRED)
endif()
## CGAL dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS system thread)

message(STATUS "CGAL VERSION ${CGAL_MAJOR_VERSION}.${CGAL_MINOR_VERSION}")
set(CGAL_VERSION ${CGAL_MAJOR_VERSION}.${CGAL_MINOR_VERSION})
if (CGAL_VERSION VERSION_GREATER 4.9) # if CGAL >= 4.10
    add_definitions(-DCGAL_MESH_3_VERBOSE=0)
endif()


if(WIN32)
    #add library dir as it is not added automatically on Windows (?)
    link_directories(${CGAL_LIBRARIES_DIR})
	
	if(NOT EXISTS ${CGAL_INSTALL_PREFIX})
		set(CGAL_INSTALL_PREFIX CACHE PATH "Set path to CGAL Install dir root.")
	endif()

    # Gather all CGAL dlls into var CGAL_DLLS
    file(GLOB CGAL_DLLS "${CGAL_INSTALL_PREFIX}/bin/*.dll")
	
	if ("${CGAL_DLLS}" STREQUAL "")
		MESSAGE(FATAL_ERROR "No CGAL_DLLS found at location: {CGAL_INSTALL_PREFIX}/bin/ equal to: ${CGAL_INSTALL_PREFIX}/bin/" )
	endif()
	
    # We suppose that the auxiliary DLLS for CGAL are always used from the CGAL package
    file(GLOB CGAL_AUX_DLLS "${GMP_INCLUDE_DIR}/../lib/*.dll")
    ## Copy DLLs to the build tree
    if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
            file(COPY ${CGAL_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
            file(COPY ${CGAL_AUX_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    else()                      # Single-config generator (nmake)
        file(COPY ${CGAL_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        file(COPY ${CGAL_AUX_DLLS} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    endif()
    ## Install DLLs
    install(FILES ${DLLS} DESTINATION bin)
    install(FILES ${LIBS} DESTINATION lib)
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PUBLIC ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} ${CGAL_INCLUDE_DIRS})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${CGALPLUGIN_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CGALPLUGIN")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_link_libraries(${PROJECT_NAME} SofaCore SofaSimulationCommon ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARY})
if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CGAL_ImageIO_LIBRARY} image)
endif()

if(SOFA_BUILD_TESTS)
   find_package(SofaTest QUIET)
   add_subdirectory(CGALPlugin_test)
endif()

## Install rules for the library and the headers; CMake package configurations files
sofa_create_package(CGALPlugin ${CGALPLUGIN_VERSION} CGALPlugin CGALPlugin)

## Install rules for the resources
install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
