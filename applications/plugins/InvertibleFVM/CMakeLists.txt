cmake_minimum_required(VERSION 2.8)
project(InvertibleFVM)

set(INVERTIBLEFVM_INSTALL_LIB_DIR     lib     CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(INVERTIBLEFVM_INSTALL_BIN_DIR     bin     CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(INVERTIBLEFVM_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    InvertibleFVMForceField.h
	InvertibleFVMForceField.inl
	initPlugin.h)

set(SOURCE_FILES
    InvertibleFVMForceField.cpp
    initPlugin.cpp)

set(README_FILES
	InvertibleFVM.txt)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_INVERTIBLEFVM")
target_link_libraries(${PROJECT_NAME} SofaBaseTopology)

install(TARGETS ${PROJECT_NAME}
    COMPONENT InvertibleFVM_libraries
    EXPORT InvertibleFVMTargets
    RUNTIME DESTINATION "${INVERTIBLEFVM_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${INVERTIBLEFVM_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INVERTIBLEFVM_INSTALL_LIB_DIR}")
