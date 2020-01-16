cmake_minimum_required(VERSION 3.1)
project(SofaFramework)

include(SofaMacros)

# Clear internal target list (set by the macro sofa_add_generic() )
set_property(GLOBAL PROPERTY __GlobalTargetList__ "")
set_property(GLOBAL PROPERTY __GlobalTargetNameList__ "")

# Options
option(SOFA_DETECTIONOUTPUT_FREEMOTION "Compile Sofa with the DETECTIONOUTPUT_FREEMOTION macro defined." OFF)
option(SOFA_NO_OPENGL "Compile Sofa with no OpenGL support. (This will define the SOFA_NO_OPENGL macro.)" OFF)
option(SOFA_NO_UPDATE_BBOX "Compile Sofa with the SOFA_NO_UPDATE_BBOX macro defined." OFF)
option(SOFA_DUMP_VISITOR_INFO "Compile Sofa with the SOFA_DUMP_VISITOR_INFO macro defined." OFF)
if(MSVC)
    option(SOFA_VECTORIZE "Enable the use of SSE2 instructions by the compiler (Only available for MSVC)." OFF)
endif()

# Variables to expose in configured files
sofa_set_01(SOFA_WITH_THREADING_ VALUE ${SOFA_WITH_THREADING}) # build_option_threading.h.in
sofa_set_01(SOFA_WITH_DEPRECATED_COMPONENTS_ VALUE ${SOFA_WITH_DEPRECATED_COMPONENTS}) # build_option_deprecated_components.h.in
sofa_set_01(SOFA_NO_UPDATE_BBOX_ VALUE ${SOFA_NO_UPDATE_BBOX}) # build_option_bbox.h.in

# Subdirectories
add_subdirectory(../modules/SofaHelper ${CMAKE_CURRENT_BINARY_DIR}/SofaHelper)
add_subdirectory(../modules/SofaDefaultType ${CMAKE_CURRENT_BINARY_DIR}/SofaDefaultType)
add_subdirectory(../modules/SofaCore ${CMAKE_CURRENT_BINARY_DIR}/SofaCore)
add_subdirectory(../modules/SofaSimulationCore ${CMAKE_CURRENT_BINARY_DIR}/SofaSimulationCore)

# Generate & install the different files that contains the defines associated with
# the state of the build option.
set(SOFA_BUILD_OPTIONS_SRC
    sharedlibrary_defines.h
    build_option_deprecated_components.h
    build_option_dump_visitor.h
    build_option_opengl.h
    build_option_threading.h
    build_option_bbox.h
    )
foreach(NAME ${SOFA_BUILD_OPTIONS_SRC})
    configure_file("${NAME}.in" "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}")
    install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config/${NAME}" DESTINATION "include/${PROJECT_NAME}/sofa/config" COMPONENT headers)
endforeach()
configure_file("config.h.in" "${CMAKE_BINARY_DIR}/include/sofa/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/config.h" DESTINATION "include/${PROJECT_NAME}/sofa" COMPONENT headers)

# Version
set(SOFAFRAMEWORK_VERSION "${Sofa_VERSION}")
set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")
configure_file(version.h.in "${CMAKE_BINARY_DIR}/include/sofa/version.h")
install(FILES "${CMAKE_BINARY_DIR}/include/sofa/version.h" DESTINATION include/${PROJECT_NAME}/sofa COMPONENT headers)

# Package install include directory
set(SOFAFRAMEWORK_TARGETS SofaCore SofaDefaultType SofaHelper SofaSimulationCore)
foreach(TARGET ${SOFAFRAMEWORK_TARGETS})
    set_target_properties(${TARGET} PROPERTIES FOLDER "SofaFramework") # IDE folder
    target_include_directories(${TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
    target_include_directories(${TARGET} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
    if(SOFA_BUILD_TESTS)
        target_compile_options(${TARGET} PUBLIC "-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources/tests\"")
    endif()
endforeach()

# CMakeParseLibraryList.cmake
configure_file(CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
install(FILES CMakeParseLibraryList.cmake DESTINATION lib/cmake/SofaFramework COMPONENT headers)

# SofaMacros.cmake
configure_file(SofaMacros.cmake ${CMAKE_BINARY_DIR}/cmake/SofaMacros.cmake COPYONLY)
install(FILES SofaMacros.cmake DESTINATION lib/cmake/SofaFramework COMPONENT headers)

# Package configuration files
sofa_write_package_config_files(SofaFramework ${SOFAFRAMEWORK_VERSION})
