cmake_minimum_required(VERSION 3.12)
project(SofaFramework)

include(SofaMacros)

# Version
set(SOFAFRAMEWORK_VERSION "${Sofa_VERSION}")
set(SOFA_VERSION_STR "\"${Sofa_VERSION}\"")
set(SOFA_VERSION "${Sofa_VERSION_MAJOR}${Sofa_VERSION_MINOR}${Sofa_VERSION_PATCH}")

# Clear internal target list (set by the macro sofa_add_generic() )
set_property(GLOBAL PROPERTY __GlobalTargetList__ "")
set_property(GLOBAL PROPERTY __GlobalTargetNameList__ "")

# Help RELOCATABLE plugins to resolve their dependencies.
# See SofaMacrosInstall.cmake for usage of this property.
define_property(TARGET
    PROPERTY "RELOCATABLE_INSTALL_DIR"
    BRIEF_DOCS "Install directory of RELOCATABLE target"
    FULL_DOCS "Install directory of RELOCATABLE target"
    )

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

# Package install include directory
set(SOFAFRAMEWORK_TARGETS SofaCore SofaDefaultType SofaHelper SofaSimulationCore)
foreach(TARGET ${SOFAFRAMEWORK_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    if(SOFA_BUILD_TESTS)
        target_compile_options(${TARGET} PUBLIC "-DFRAMEWORK_TEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/resources/tests\"")
    endif()
endforeach()

set(SOFAFRAMEWORK_SRC src/sofa)
set(HEADER_FILES
    ${SOFAFRAMEWORK_SRC}/config.h.in
    ${SOFAFRAMEWORK_SRC}/initSofaFramework.h
    ${SOFAFRAMEWORK_SRC}/version.h.in
    )
set(SOURCE_FILES
    ${SOFAFRAMEWORK_SRC}/initSofaFramework.cpp
    )
set(SOFA_BUILD_OPTIONS_SRC
    ${SOFAFRAMEWORK_SRC}/config/sharedlibrary_defines.h.in
    ${SOFAFRAMEWORK_SRC}/config/build_option_deprecated_components.h.in
    ${SOFAFRAMEWORK_SRC}/config/build_option_dump_visitor.h.in
    ${SOFAFRAMEWORK_SRC}/config/build_option_opengl.h.in
    ${SOFAFRAMEWORK_SRC}/config/build_option_threading.h.in
    ${SOFAFRAMEWORK_SRC}/config/build_option_bbox.h.in
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${SOFA_BUILD_OPTIONS_SRC})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAFRAMEWORK_TARGETS})

# CMakeParseLibraryList.cmake
configure_file(CMakeParseLibraryList.cmake ${CMAKE_BINARY_DIR}/cmake/CMakeParseLibraryList.cmake COPYONLY)
install(FILES CMakeParseLibraryList.cmake DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)

# SofaMacros*.cmake
file(GLOB macro_files RELATIVE ${CMAKE_CURRENT_LIST_DIR} LIST_DIRECTORIES FALSE "SofaMacros*.cmake")
foreach(macro_file ${macro_files})
    configure_file(${macro_file} ${CMAKE_BINARY_DIR}/cmake/${macro_file} COPYONLY)
    install(FILES ${macro_file} DESTINATION lib/cmake/${PROJECT_NAME} COMPONENT headers)
endforeach()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
