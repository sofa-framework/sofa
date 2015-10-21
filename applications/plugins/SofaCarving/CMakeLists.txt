cmake_minimum_required(VERSION 2.8)
project(SofaCarving)

set(SOFACARVING_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFACARVING_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFACARVING_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
	CarvingManager.h
	initSofaCarving.h)

set(SOURCE_FILES
	CarvingManager.cpp
    initSofaCarving.cpp)

set(README_FILES
	SofaCarving.txt)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFACARVING")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)

install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaCarving_libraries
    EXPORT SofaCarvingTargets
    RUNTIME DESTINATION "${SOFACARVING_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SOFACARVING_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SOFACARVING_INSTALL_LIB_DIR}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
