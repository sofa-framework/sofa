cmake_minimum_required(VERSION 2.8.12)

set(SOFA_INSTALL_CMAKE_DIR share/cmake/Modules CACHE PATH "Installation directory for sofa cmake functions relative to CMAKE_INSTALL_PREFIX variable"  )

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/Modules/CMakeSofaCreateTarget.cmake 
        DESTINATION "${SOFA_INSTALL_CMAKE_DIR}")