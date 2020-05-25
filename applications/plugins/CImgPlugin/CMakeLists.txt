cmake_minimum_required(VERSION 3.1)
project(CImgPlugin VERSION 0.1)

set(HEADER_FILES
    src/CImgPlugin/ImageCImg.h
    src/CImgPlugin/CImgPlugin.h.in
    src/CImgPlugin/SOFACImg.h
    src/CImgPlugin/CImgData.h
    )

set(SOURCE_FILES
    src/CImgPlugin/ImageCImg.cpp
    src/CImgPlugin/initCImgPlugin.cpp
    )

add_subdirectory(extlibs/CImg)

find_package(SofaFramework REQUIRED)
sofa_find_package(CImg REQUIRED)

# OS X only: if the user installed its own JPEG/PNG lib (typically with homebrew/port),
# it will allow to use those instead of those present in XCode's frameworks
set(CMAKE_FIND_FRAMEWORK LAST)

sofa_find_package(JPEG QUIET)
sofa_find_package(TIFF QUIET)
sofa_find_package(PNG QUIET)
sofa_find_package(ZLIB QUIET)

### Configure Paths according to detected packages
if(TIFF_FOUND AND NOT SOFA_BUILD_RELEASE_PACKAGE)
    message(STATUS "CImgPlugin: TIFF Image Format enabled")
    get_filename_component(TIFF_LIB_DIRS ${TIFF_LIBRARIES} PATH)
    list(APPEND CIMG_CFLAGS -Dcimg_use_tiff)
    link_directories(${TIFF_LIB_DIRS})
    include_directories(${TIFF_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${TIFF_LIBRARIES})
    sofa_install_libraries(TARGETS TIFF::TIFF)
else()
    message(STATUS "CImgPlugin: TIFF Image Format disabled")
endif()

if(JPEG_FOUND)
    message(STATUS "CImgPlugin: JPEG Image Format enabled")
    get_filename_component(JPEG_LIB_DIRS ${JPEG_LIBRARIES} PATH)
    list(APPEND CIMG_CFLAGS -Dcimg_use_jpeg)
    if(WIN32)
        # prevent INT32 redefinition on Windows (see jmorecfg.h)
        list(APPEND CIMG_CFLAGS -DXMD_H)
    endif()
    link_directories(${JPEG_LIB_DIRS})
    include_directories(${JPEG_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${JPEG_LIBRARIES})
    sofa_install_libraries(PATHS ${JPEG_LIBRARIES})
else()
    message(STATUS "CImgPlugin: JPEG Image Format disabled")
endif()

if (ZLIB_FOUND)
    link_directories(${ZLIB_LIB_DIRS})
    include_directories(${ZLIB_INCLUDE_DIR})
    SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${ZLIB_LIBRARIES})
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS ZLIB::ZLIB)
    endif()

    # PNG requires ZLIB
    if(PNG_FOUND)
        message(STATUS "CImgPlugin: PNG Image Format enabled")
        list(APPEND CIMG_CFLAGS -Dcimg_use_png)
        if(UNIX)
            # Prevent pngconf error (see https://bugs.launchpad.net/ubuntu/+source/libpng/+bug/218409)
            list(APPEND CIMG_CFLAGS -DPNG_SKIP_SETJMP_CHECK)
        endif()
        link_directories(${PNG_LIB_DIRS})
        include_directories(${PNG_INCLUDE_DIR} )
        SET( EXTERNAL_LIBS ${EXTERNAL_LIBS} ${PNG_LIBRARIES} )
        sofa_install_libraries(TARGETS PNG::PNG)
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
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${CImg_INCLUDE_DIRS}>")

## Install rules for the library; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "CImgPlugin"
    RELOCATABLE "plugins"
    )

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(CImgPlugin_test)
    endif()
endif()
