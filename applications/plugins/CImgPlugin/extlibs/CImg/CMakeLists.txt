cmake_minimum_required(VERSION 3.1)

project(CImg)

install(FILES
        CImg.h
        DESTINATION include/CImg
        COMPONENT CImg_headers)

install(DIRECTORY
        plugins
        DESTINATION include/CImg
        COMPONENT CImg_headers)


include(CMakePackageConfigHelpers)

### CImgConfig.cmake
## Build tree
set(CIMG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
configure_package_config_file(CImgConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/cmake/CImgConfig.cmake
                              INSTALL_DESTINATION lib/cmake/CImg
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
