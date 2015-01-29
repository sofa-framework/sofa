cmake_minimum_required(VERSION 2.8.12)
project(SuiteSparse)

if(MSVC)
    set(SUITESPARSE_INSTALL_LIB_DIR     lib                 CACHE PATH "Installation directory for libraries")
    set(SUITESPARSE_INSTALL_BIN_DIR     bin                 CACHE PATH "Installation directory for executables")
    set(SUITESPARSE_INSTALL_INCLUDE_DIR include             CACHE PATH "Installation directory for header files")

    # # I think we don't need this, because nobody actually ever
    # # includes a cholmod file in SOFA
    # file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Include/*.h")
    # install(FILES ${headers} DESTINATION "${SUITESPARSE_INSTALL_INCLUDE_DIR}")

    file(GLOB libs "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*.lib")
    file(GLOB dlls "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*.dll")
    # Copy libraries to the build tree
    file(COPY ${libs} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    file(COPY ${dlls} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    # Copy libraries to the install directory
    install(FILES ${libs} DESTINATION "${SUITESPARSE_INSTALL_LIB_DIR}")
    install(FILES ${dlls} DESTINATION "${SUITESPARSE_INSTALL_BIN_DIR}")
endif()
