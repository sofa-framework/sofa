cmake_minimum_required(VERSION 2.8.12)

project(CImg)

set(CIMG_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

install(FILES
        CImg.h cimgmatlab.h jpeg_buffer.h nlmeans.h skeleton.h SOFACImg.h vrml.h
        DESTINATION "${CIMG_INSTALL_INCLUDE_DIR}/CImg"
        COMPONENT CImg_headers)
