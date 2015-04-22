cmake_minimum_required(VERSION 2.8)
project(ManualMapping)

set(MANUALMAPPING_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(MANUALMAPPING_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(MANUALMAPPING_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    initManualMapping.h
    ManualLinearMapping.h
    ManualLinearMapping.inl)

set(SOURCE_FILES
    initManualMapping.cpp
    ManualLinearMapping.cpp)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ManualMapping")
target_link_libraries(${PROJECT_NAME} SofaComponentBase)

install(TARGETS ${PROJECT_NAME}
    COMPONENT ManualMapping_libraries
    EXPORT ManualMappingTargets
    RUNTIME DESTINATION "${MANUALMAPPING_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${MANUALMAPPING_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${MANUALMAPPING_INSTALL_LIB_DIR}")
