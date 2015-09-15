cmake_minimum_required(VERSION 2.8.12)

project(CImg)

install(FILES CImg.h cimgmatlab.h jpeg_buffer.h nlmeans.h skeleton.h SOFACImg.h vrml.h
        DESTINATION include/CImg
        COMPONENT CImg_headers)


include(CMakePackageConfigHelpers)

### CImgConfig.cmake
## Build tree
set(CIMG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
configure_package_config_file(CImgConfig.cmake.in
                              CImgConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_BINARY_DIR}/CImg
                              PATH_VARS CIMG_INCLUDE_DIR)
## Install tree
set(CIMG_INCLUDE_DIR include)
configure_package_config_file(CImgConfig.cmake.in
                              InstalledCImgConfig.cmake
                              INSTALL_DESTINATION lib/cmake/CImg
                              PATH_VARS CIMG_INCLUDE_DIR)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InstalledCImgConfig.cmake
        DESTINATION lib/cmake/CImg
        RENAME CImgConfig.cmake)
