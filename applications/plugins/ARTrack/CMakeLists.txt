cmake_minimum_required(VERSION 2.8)
project(ARTrack)

set(ARTRACK_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(ARTRACK_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(ARTRACK_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    ARTrackDriver.h
    ARTrackEvent.h
    ARTrackController.h
    ARTrackController.inl)

set(SOURCE_FILES
    ARTrackDriver.cpp
    initARTrack.cpp
    ARTrackEvent.cpp
    ARTrackController.cpp)

find_package(ARTrackLib REQUIRED)
find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ARTRACKPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaUserInteraction)
target_link_libraries(${PROJECT_NAME} ARTrackLib)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} Ws2_32)
endif()

if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT ARTrack_libraries
    EXPORT ARTrackTargets
    RUNTIME DESTINATION "${ARTRACK_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${ARTRACK_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${ARTRACK_INSTALL_LIB_DIR}")
