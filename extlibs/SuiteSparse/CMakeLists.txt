cmake_minimum_required(VERSION 2.8.12)
project(SuiteSparse)

if(MSVC)
    # # I think we don't need this, because nobody actually ever
    # # includes a cholmod file in SOFA
    # file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Include/*.h")
    # install(FILES ${headers} DESTINATION include)

    file(GLOB libs "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*.lib")
    file(GLOB dlls "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*.dll")
    # Copy libraries to the build tree
    file(COPY ${libs} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    file(COPY ${dlls} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    # Copy libraries to the install directory
    install(FILES ${libs} DESTINATION lib)
    install(FILES ${dlls} DESTINATION bin)
endif()
