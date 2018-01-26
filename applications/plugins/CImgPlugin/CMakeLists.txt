cmake_minimum_required(VERSION 2.8.12)
project(CImgPlugin)


set(${PROJECT_NAME}_MAJOR_VERSION 0)
set(${PROJECT_NAME}_MINOR_VERSION 1)
set(${PROJECT_NAME}_VERSION ${${PROJECT_NAME}_MAJOR_VERSION}.${${PROJECT_NAME}_MINOR_VERSION})

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

find_package(JPEG)
find_package(TIFF)
find_package(PNG)

### Configure Paths according to detected packages
if(TIFF_FOUND)
  message(STATUS "CImgPlugin: Enable TIFF Image Format")
  get_filename_component(TIFF_LIB_DIRS ${TIFF_LIBRARIES} PATH)
  SET(CIMG_CFLAGS "${CIMG_CFLAGS} ${CIMG_TIFF_CCFLAGS} -DCIMGPLUGIN_HAVE_TIFF -Dcimg_use_tiff")
  link_directories(${TIFF_LIB_DIRS})
  include_directories(${TIFF_INCLUDE_DIR})
  SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${TIFF_LIBRARIES})
  if(WIN32)
      sofa_copy_libraries_from_targets(TIFF::TIFF) # Windows dll -> bin
  endif()
  sofa_install_libraries_from_targets(TIFF::TIFF)
else()
  message(STATUS "CImgPlugin: Did not find TIFF library")
endif()

if(JPEG_FOUND)
  message(STATUS "CImgPlugin: Enable JPEG Image Format")
  get_filename_component(JPEG_LIB_DIRS ${JPEG_LIBRARIES} PATH)
  SET(CIMG_CFLAGS "${CIMG_CFLAGS} ${CIMG_JPEG_CCFLAGS} -DCIMGPLUGIN_HAVE_JPEG -Dcimg_use_jpeg")
  if(WIN32)
      # prevent INT32 redefinition on Windows (see jmorecfg.h)
      SET(CIMG_CFLAGS "${CIMG_CFLAGS} -DXMD_H")
  endif()
  link_directories(${JPEG_LIB_DIRS})
  include_directories(${JPEG_INCLUDE_DIR})
  SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${JPEG_LIBRARIES})
  if(WIN32)
      sofa_copy_libraries(${JPEG_LIBRARIES}) # Windows dll -> bin
  endif()
  sofa_install_libraries(${JPEG_LIBRARIES})
else()
  message(STATUS "CImgPlugin: Did not find JPEG library")
endif()

if (ZLIB_FOUND)
  SET(CIMG_CFLAGS "${CIMG_CFLAGS} ${CIMG_ZLIB_CCFLAGS}")
  link_directories(${ZLIB_LIB_DIRS})
  include_directories(${ZLIB_INCLUDE_DIR})
  SET(EXTERNAL_LIBS ${EXTERNAL_LIBS} ${ZLIB_LIBRARIES})

  # PNG requires ZLIB
  if(PNG_FOUND)
    message(STATUS "CImgPlugin: Enable PNG Image Format")
    SET(CIMG_CFLAGS "${CIMG_CFLAGS} ${CIMG_PNG_CCFLAGS} -DCIMGPLUGIN_HAVE_PNG -Dcimg_use_png")
    link_directories(${PNG_LIB_DIRS})
    include_directories(${PNG_INCLUDE_DIR} )
    SET( EXTERNAL_LIBS ${EXTERNAL_LIBS} ${PNG_LIBRARIES} )
    if(WIN32)
        sofa_copy_libraries_from_targets(PNG::PNG) # Windows dll -> bin
    endif()
    sofa_install_libraries_from_targets(PNG::PNG)
  else()
      message(STATUS "CImgPlugin: Did not find PNG library")
  endif()
else()
    message(STATUS "CImgPlugin: Did not find zlib library (required for PNG Image Format)")
endif()

#Remove CImg sys calls
SET(CIMG_CFLAGS "${CIMG_CFLAGS} -Dcimg_no_system_calls")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${CIMG_CFLAGS} -DSOFA_BUILD_CIMGPLUGIN")
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
