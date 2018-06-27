cmake_minimum_required(VERSION 3.1)
project(SuiteSparse)

if(MSVC)
    # # I think we don't need this, because nobody actually ever
    # # includes a cholmod file in SOFA
    # file(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Include/*.h")
    # install(FILES ${headers} DESTINATION include)

    file(GLOB libs "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*${CMAKE_STATIC_LIBRARY_SUFFIX}")
    file(GLOB dlls "${CMAKE_CURRENT_SOURCE_DIR}/cholmod/Lib/*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    # Copy libraries to the build tree

	file(COPY ${libs} DESTINATION "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
	if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (MSVC)
        foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
			file(COPY ${dlls} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CONFIG}")
        endforeach()
    else()                      # Single-config generator (nmake)
		file(COPY ${dlls} DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
    endif()
    # Copy libraries to the install directory
    install(FILES ${libs} DESTINATION ${ARCHIVE_OUTPUT_DIRECTORY} COMPONENT libraries)
    install(FILES ${dlls} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY} COMPONENT applications)
endif()
