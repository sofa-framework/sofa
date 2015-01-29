cmake_minimum_required(VERSION 2.8)
project(Haption)

set(HAPTION_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(HAPTION_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(HAPTION_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    HaptionDriver.h)

set(SOURCE_FILES
    initHaption.cpp 
    HaptionDriver.cpp)

set(README_FILES
	PluginHaption.txt)	

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_HAPTIONPLUGIN")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)

install(TARGETS ${PROJECT_NAME}
    COMPONENT Haption_libraries
    EXPORT HaptionTargets
    RUNTIME DESTINATION "${HAPTION_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${HAPTION_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${HAPTION_INSTALL_LIB_DIR}")
