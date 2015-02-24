cmake_minimum_required(VERSION 2.8)
project(THMPGSpatialHashing)

set(THMPGSPATIALHASHING_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(THMPGSPATIALHASHING_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(THMPGSPATIALHASHING_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    initTHMPGSpatialHashingPlugin.h
    THMPGSpatialHashing.h
    THMPGHashTable.h)

set(SOURCE_FILES
    initTHMPGSpatialHashingPlugin.cpp
    THMPGSpatialHashing.cpp
    THMPGHashTable.cpp)

find_package(SofaBase REQUIRED)
find_package(Boost COMPONENTS system REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_THMPGSPATIALHASHING")
target_link_libraries(${PROJECT_NAME} SofaComponentBase)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})

if(LINUX OR UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} rt)   # for clock_gettime()
endif()

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(THMPGSpatialHashing_test)
endif()

install(TARGETS ${PROJECT_NAME}
    COMPONENT THMPGSpatialHashing_libraries
    EXPORT THMPGSpatialHashingTargets
    RUNTIME DESTINATION "${THMPGSPATIALHASHING_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${THMPGSPATIALHASHING_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${THMPGSPATIALHASHING_INSTALL_LIB_DIR}")
