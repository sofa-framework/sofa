cmake_minimum_required(VERSION 2.8.12)
project(CGALPlugin)

set(CGALPLUGIN_MAJOR_VERSION 0)
set(CGALPLUGIN_MINOR_VERSION 0)
set(CGALPLUGIN_BUILD_VERSION 1)

set(CGALPLUGIN_VERSION ${CGALPLUGIN_MAJOR_VERSION}.${CGALPLUGIN_MINOR_VERSION}.${CGALPLUGIN_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(CGALPLUGIN_BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CGALPLUGIN_BUILD_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CGALPLUGIN_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CGALPLUGIN_BUILD_CMAKE_DIR ${CMAKE_BINARY_DIR}/CGALPlugin)

set(CGALPLUGIN_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(CGALPLUGIN_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(CGALPLUGIN_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(CGALPLUGIN_INSTALL_CMAKE_DIR ${CGALPLUGIN_INSTALL_LIB_DIR}/cmake/image)

set(HEADER_FILES
    DecimateMesh.h
    DecimateMesh.inl
    MeshGenerationFromPolyhedron.h
    MeshGenerationFromPolyhedron.inl
    TriangularConvexHull3D.h
    TriangularConvexHull3D.inl
    initCGALPlugin.h
)

set(SOURCE_FILES
    DecimateMesh.cpp
    MeshGenerationFromPolyhedron.cpp
    TriangularConvexHull3D.cpp
    initCGALPlugin.cpp
)

set(README_FILES CGALPlugin.txt)

find_package(SofaFramework REQUIRED)
find_package(CGAL REQUIRED)

# Dependencies required by CGAL
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(MPFR REQUIRED)
find_package(GMP REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS system thread)

find_package(image QUIET)
if(image_FOUND)
    list(APPEND HEADER_FILES MeshGenerationFromImage.h)
    list(APPEND HEADER_FILES MeshGenerationFromImage.inl)
    list(APPEND SOURCE_FILES MeshGenerationFromImage.cpp)
else()
    message(STATUS "CGALPlugin: could not find image, won't build MeshGenerationFromImage")
endif()

set(CGALPLUGIN_DEPENDENCY_INCLUDE_DIR ${SofaFramework_INCLUDE_DIRS}
    ${GMP_INCLUDE_DIR} ${MPFR_INCLUDE_DIR} ${CGAL_DIR}/include)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CGALPlugin.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/CGALPlugin/CGALPlugin.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/CGALPlugin/CGALPlugin.h
         DESTINATION "${CGALPLUGIN_INSTALL_INCLUDE_DIR}/CGALPlugin")

if(WIN32)
    # copy required CGAL dlls
    file(GLOB sharedObjects "${CGAL_DIR}/bin/*.dll")
    foreach(sharedObject ${sharedObjects})
        file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
    endforeach()
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CGALPLUGIN_DEPENDENCY_INCLUDE_DIR}>")

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${CGALPLUGIN_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CGALPLUGIN")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

target_link_libraries(${PROJECT_NAME} SofaCore ${Boost_LIBRARIES} ${GMP_LIBRARIES} ${MPFR_LIBRARIES} ${CGAL_LIBRARY})

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CGAL_ImageIO_LIBRARY} image)
endif()


install(TARGETS ${PROJECT_NAME}
    COMPONENT CGALPlugin_libraries
    EXPORT CGALPluginTargets
    RUNTIME DESTINATION "${CGALPLUGIN_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${CGALPLUGIN_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${CGALPLUGIN_INSTALL_LIB_DIR}"
    PUBLIC_HEADER DESTINATION "${CGALPLUGIN_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")

install(DIRECTORY scenes/ DESTINATION share/sofa/plugins/${PROJECT_NAME})


include(CMakePackageConfigHelpers)
# CGALPluginConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${CGALPLUGIN_BUILD_CMAKE_DIR}/CGALPluginConfigVersion.cmake
                                 VERSION ${CGALPLUGIN_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${CGALPLUGIN_BUILD_CMAKE_DIR}/CGALPluginConfigVersion.cmake
        DESTINATION ${CGALPLUGIN_INSTALL_CMAKE_DIR})


# CGALPluginConfig.cmake (build tree)
set(CGALPLUGIN_INCLUDE_DIR "${CGALPLUGIN_BUILD_INCLUDE_DIR}")
configure_package_config_file(CGALPluginConfig.cmake.in
                              ${CGALPLUGIN_BUILD_CMAKE_DIR}/CGALPluginConfig.cmake
                              INSTALL_DESTINATION ${CGALPLUGIN_BUILD_CMAKE_DIR}
                              PATH_VARS CGALPLUGIN_INCLUDE_DIR)

# CGALPluginConfig.cmake (installed)
set(CGALPLUGIN_INCLUDE_DIR "${CGALPLUGIN_INSTALL_INCLUDE_DIR}")
configure_package_config_file(CGALPluginConfig.cmake.in
                              ${CGALPLUGIN_BUILD_CMAKE_DIR}/CGALPluginConfigInstall.cmake
                              INSTALL_DESTINATION ${CGALPLUGIN_INSTALL_CMAKE_DIR}
                              PATH_VARS CGALPLUGIN_INCLUDE_DIR)

install(FILES ${CGALPLUGIN_BUILD_CMAKE_DIR}/CGALPluginConfigInstall.cmake
        DESTINATION ${CGALPLUGIN_INSTALL_CMAKE_DIR}
        RENAME CGALPluginConfig.cmake)

# CGALPluginTargets.cmake (installed)
install(EXPORT CGALPluginTargets
        DESTINATION ${CGALPLUGIN_INSTALL_CMAKE_DIR})
