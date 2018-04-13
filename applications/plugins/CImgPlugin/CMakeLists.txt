cmake_minimum_required(VERSION 2.8.12)
project(CImgPlugin)


set(${PROJECT_NAME}_MAJOR_VERSION 0)
set(${PROJECT_NAME}_MINOR_VERSION 1)
set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

## RPATH
if(UNIX)
    set(CMAKE_INSTALL_RPATH "../lib")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_MACOSX_RPATH ON)
    endif()
endif(UNIX)

set(HEADER_FILES
    ImageCImg.h
    CImgPlugin.h
    SOFACImg.h
    )

set(SOURCE_FILES
    ImageCImg.cpp
    initCImgPlugin.cpp
    )

add_subdirectory(extlibs/CImg extlibs/CImg)

find_package(CImg REQUIRED)
find_package(SofaFramework REQUIRED)

find_package(JPEG QUIET)
find_package(TIFF QUIET)
find_package(PNG QUIET)

### Configure Paths according to detected packages
if(TIFF_FOUND)
    message(STATUS "CImgPlugin: TIFF Image Format enabled")
    get_filename_component(TIFF_LIB_DIRS ${TIFF_LIBRARIES} PATH)
    list(APPEND CIMG_CFLAGS ${CIMG_TIFF_CCFLAGS} -DCIMGPLUGIN_HAVE_TIFF -Dcimg_use_tiff)
    link_directories(${TIFF_LIB_DIRS})
    include_directories(${TIFF_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${TIFF_LIBRARIES})
else()
    message(STATUS "CImgPlugin: TIFF Image Format disabled")
endif()

if(JPEG_FOUND)
    message(STATUS "CImgPlugin: JPEG Image Format enabled")
    get_filename_component(JPEG_LIB_DIRS ${JPEG_LIBRARIES} PATH)
    list(APPEND CIMG_CFLAGS ${CIMG_JPEG_CCFLAGS} -DCIMGPLUGIN_HAVE_JPEG -Dcimg_use_jpeg)
    if(WIN32)
        # prevent INT32 redefinition on Windows (see jmorecfg.h)
        list(APPEND CIMG_CFLAGS -DXMD_H)
    endif()
    link_directories(${JPEG_LIB_DIRS})
    include_directories(${JPEG_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${JPEG_LIBRARIES})
else()
    message(STATUS "CImgPlugin: JPEG Image Format disabled")
endif()

if (ZLIB_FOUND)
    list(APPEND CIMG_CFLAGS ${CIMG_ZLIB_CCFLAGS})
    link_directories(${ZLIB_LIB_DIRS})
    include_directories(${ZLIB_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${ZLIB_LIBRARIES})

    # PNG requires ZLIB
    if(PNG_FOUND)
        message(STATUS "CImgPlugin: PNG Image Format enabled")
        list(APPEND CIMG_CFLAGS ${CIMG_PNG_CCFLAGS} -DCIMGPLUGIN_HAVE_PNG -Dcimg_use_png)
    if(UNIX)
        # Prevent pngconf error (see https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409)
        list(APPEND CIMG_CFLAGS -DPNG_SKIP_SETJMP_CHECK)
    endif()
        link_directories(${PNG_LIB_DIRS})
        include_directories(${PNG_INCLUDE_DIR} )
        SET( EXTERNAL_LIBS ${EXTERNAL_LIBS} ${PNG_LIBRARIES} )
    else()
        message(STATUS "CImgPlugin: PNG Image Format disabled")
    endif()
else()
    message(STATUS "CImgPlugin: Did not find ZLIB library (required for PNG Image Format)")
endif()

#Remove CImg sys calls
list(APPEND CIMG_CFLAGS -Dcimg_no_system_calls)



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_compile_options(${PROJECT_NAME} PUBLIC ${CIMG_CFLAGS})
target_compile_options(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_CIMGPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaCore ${EXTERNAL_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>/include")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CImg_INCLUDE_DIRS}>")

## Install rules for the library; CMake package configurations files
sofa_create_package(${PROJECT_NAME} ${${PROJECT_NAME}_VERSION} ${PROJECT_NAME} ${PROJECT_NAME})

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(CImgPlugin_test)
    endif()
endif()
